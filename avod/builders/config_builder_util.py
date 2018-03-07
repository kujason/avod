"""Config file reader utils."""

import os
import shutil

from google.protobuf import text_format

import avod
from avod.protos import model_pb2
from avod.protos import pipeline_pb2


class ConfigObj:
    pass


def proto_to_obj(config):
    """Hack to convert proto config into an object so repeated fields can be
    overwritten

    Args:
        config: proto config

    Returns:
        config_obj: object with same fields as the config
    """
    all_fields = list(config.DESCRIPTOR.fields_by_name)
    config_obj = ConfigObj()
    for field in all_fields:
        field_value = eval('config.{}'.format(field))
        setattr(config_obj, field, field_value)

    return config_obj


def get_model_config_from_file(config_path):
    """Reads model configuration from a configuration file.
       This merges the layer config info with model default configs.
    Args:
        config_path: A path to the config

    Returns:
        layers_config: A configured model_pb2 config
    """

    model_config = model_pb2.ModelConfig()
    with open(config_path, 'r') as f:
        text_format.Merge(f.read(), model_config)
    return model_config


def get_configs_from_pipeline_file(pipeline_config_path,
                                   is_training):
    """Reads model configuration from a pipeline_pb2.NetworkPipelineConfig.
    Args:
        pipeline_config_path: A path directory to the network pipeline config
        is_training: A boolean flag to indicate training stage, used for
            creating the checkpoint directory which must be created at the
            first training iteration.
    Returns:
        model_config: A model_pb2.ModelConfig
        train_config: A train_pb2.TrainConfig
        eval_config: A eval_pb2.EvalConfig
        dataset_config: A kitti_dataset_pb2.KittiDatasetConfig
    """

    pipeline_config = pipeline_pb2.NetworkPipelineConfig()
    with open(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    model_config = pipeline_config.model_config

    # Make sure the checkpoint name matches the config filename
    config_file_name = \
        os.path.split(pipeline_config_path)[1].split('.')[0]
    checkpoint_name = model_config.checkpoint_name
    if config_file_name != checkpoint_name:
        raise ValueError('Config and checkpoint names must match.')

    output_root_dir = avod.root_dir() + '/data/outputs/' + checkpoint_name

    # Construct paths
    paths_config = model_config.paths_config
    if not paths_config.checkpoint_dir:
        checkpoint_dir = output_root_dir + '/checkpoints'

        if is_training:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        paths_config.checkpoint_dir = checkpoint_dir

    if not paths_config.logdir:
        paths_config.logdir = output_root_dir + '/logs/'

    if not paths_config.pred_dir:
        paths_config.pred_dir = output_root_dir + '/predictions'

    train_config = pipeline_config.train_config
    eval_config = pipeline_config.eval_config
    dataset_config = pipeline_config.dataset_config

    if is_training:
        # Copy the config to the experiments folder
        experiment_config_path = output_root_dir + '/' +\
            model_config.checkpoint_name
        experiment_config_path += '.config'
        # Copy this even if the config exists, in case some parameters
        # were modified
        shutil.copy(pipeline_config_path, experiment_config_path)

    return model_config, train_config, eval_config, dataset_config

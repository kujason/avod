"""Tests for avod.core.trainer with a dummy Detection Model"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

from google.protobuf import text_format

from avod.core import trainer
from avod.core import model

from avod.protos import train_pb2
from avod.protos import model_pb2


class FakeBatchNormClassifier(model.DetectionModel):

    def __init__(self, model_config, num_classes=1):
        # Sets model configs (_config and _num_classes)
        super(FakeBatchNormClassifier, self).__init__(model_config)

        self.tf_inputs, self.tf_labels = self.get_input()
        self._train_op = None
        self._loss = None

    def BatchNormClassifier(self, inputs):
        inputs = layers.batch_norm(inputs, decay=0.1, fused=None)
        return layers.fully_connected(inputs, 1, activation_fn=math_ops.sigmoid)

    def get_input(self):
        """Creates an easy training set."""
        np.random.seed(0)

        inputs = np.zeros((16, 4))
        labels = np.random.randint(
                0, 2, size=(16, 1)).astype(
                np.float32)

        for i in range(16):
            j = int(2 * labels[i] + np.random.randint(0, 2))
            inputs[i, j] = 1

        random_seed.set_random_seed(0)
        tf_inputs = constant_op.constant(inputs, dtype=dtypes.float32)
        tf_labels = constant_op.constant(labels, dtype=dtypes.float32)

        return tf_inputs, tf_labels

    def build(self):
        """Prediction tensors from inputs tensor.

        Args:
            preprocessed_inputs: a [batch, 28, 28, channels] float32 tensor.

        Returns:
            prediction_dict: a dictionary holding prediction tensors to be
                             passed to the Loss or Postprocess functions.
        """
        tf_predictions = self.BatchNormClassifier(self.tf_inputs)
        return tf_predictions

    def loss(self,  tf_predictions):
        """Compute scalar loss tensors with respect to provided groundtruth.
        """
        # trainer expects two losses, pass in a dummy one
        dummy_loss_dict = {}
        total_loss = tf.losses.log_loss(self.tf_labels,
                                        tf_predictions,
                                        scope='BatchNormLoss')
        return dummy_loss_dict, total_loss


class ClassifierTrainerTest(tf.test.TestCase):

    def test_batch_norm_class(self):
        # This tests the model and trainer set up
        train_config_text_proto = """
        optimizer {
          gradient_descent {
            learning_rate {
              constant_learning_rate {
                learning_rate: 1.0
              }
            }
          }
        }
        max_iterations: 5
        """
        model_config_text_proto = """
            path_drop_probabilities: [1.0, 1.0]
        """
        train_config = train_pb2.TrainConfig()
        text_format.Merge(train_config_text_proto, train_config)

        model_config = model_pb2.ModelConfig()
        text_format.Merge(model_config_text_proto, model_config)
        train_config.overwrite_checkpoints = True
        test_root_dir = '/tmp/avod_unit_test/'

        paths_config = model_config.paths_config
        paths_config.logdir = test_root_dir + 'logs/'
        paths_config.checkpoint_dir = test_root_dir

        classifier = FakeBatchNormClassifier(model_config)
        trainer.train(classifier,
                      train_config)


if __name__ == '__main__':
    tf.test.main()

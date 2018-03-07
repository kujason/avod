from copy import deepcopy

from google.protobuf import text_format

import avod
from avod.datasets.kitti.kitti_dataset import KittiDataset
from avod.protos import kitti_dataset_pb2
from avod.protos.kitti_dataset_pb2 import KittiDatasetConfig


class DatasetBuilder(object):
    """
    Static class to return preconfigured dataset objects
    """

    KITTI_UNITTEST = KittiDatasetConfig(
        name="unittest-kitti",
        dataset_dir=avod.root_dir() + "/tests/datasets/Kitti/object",
        data_split="train",
        data_split_dir="training",
        has_labels=True,
        cluster_split="train",
        classes=["Car", "Pedestrian", "Cyclist"],
        num_clusters=[2, 1, 1],
    )

    KITTI_TRAIN = KittiDatasetConfig(
        name="kitti",
        data_split="train",
        data_split_dir="training",
        has_labels=True,
        cluster_split="train",
        classes=["Car"],
        num_clusters=[2]
    )

    KITTI_VAL = KittiDatasetConfig(
        name="kitti",
        data_split="val",
        data_split_dir="training",
        has_labels=True,
        cluster_split="train",
        classes=["Car"],
        num_clusters=[2],
    )

    KITTI_TEST = KittiDatasetConfig(
        name="kitti",
        data_split="test",
        data_split_dir="testing",
        has_labels=False,
        cluster_split="train",
        classes=["Car"],
        num_clusters=[2],
    )

    KITTI_TRAINVAL = KittiDatasetConfig(
        name="kitti",
        data_split="trainval",
        data_split_dir="training",
        has_labels=True,
        cluster_split="trainval",
        classes=["Car"],
        num_clusters=[2],
    )

    KITTI_TRAIN_MINI = KittiDatasetConfig(
        name="kitti",
        data_split="train_mini",
        data_split_dir="training",
        has_labels=True,
        cluster_split="train",
        classes=["Car"],
        num_clusters=[2],
    )
    KITTI_VAL_MINI = KittiDatasetConfig(
        name="kitti",
        data_split="val_mini",
        data_split_dir="training",
        has_labels=True,
        cluster_split="train",
        classes=["Car"],
        num_clusters=[2],
    )
    KITTI_TEST_MINI = KittiDatasetConfig(
        name="kitti",
        data_split="test_mini",
        data_split_dir="testing",
        has_labels=False,
        cluster_split="train",
        classes=["Car"],
        num_clusters=[2],
    )

    CONFIG_DEFAULTS_PROTO = \
        """
        bev_source: 'lidar'

        kitti_utils_config {
            area_extents: [-40, 40, -5, 3, 0, 70]
            voxel_size: 0.1
            anchor_strides: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

            bev_generator {
                slices {
                    height_lo: -0.2
                    height_hi: 2.3
                    num_slices: 5
                }
            }

            mini_batch_config {
                density_threshold: 1

                rpn_config {
                    iou_2d_thresholds {
                        neg_iou_lo: 0.0
                        neg_iou_hi: 0.3
                        pos_iou_lo: 0.5
                        pos_iou_hi: 1.0
                    }
                    # iou_3d_thresholds {
                    #     neg_iou_lo: 0.0
                    #     neg_iou_hi: 0.005
                    #     pos_iou_lo: 0.1
                    #     pos_iou_hi: 1.0
                    # }

                    mini_batch_size: 512
                }

                avod_config {
                    iou_2d_thresholds {
                        neg_iou_lo: 0.0
                        neg_iou_hi: 0.55
                        pos_iou_lo: 0.65
                        pos_iou_hi: 1.0
                    }

                    mini_batch_size: 1024
                }
            }
        }
        """

    @staticmethod
    def load_dataset_from_config(dataset_config_path):

        dataset_config = kitti_dataset_pb2.KittiDatasetConfig()
        with open(dataset_config_path, 'r') as f:
            text_format.Merge(f.read(), dataset_config)

        return DatasetBuilder.build_kitti_dataset(dataset_config,
                                                  use_defaults=False)

    @staticmethod
    def copy_config(cfg):
        return deepcopy(cfg)

    @staticmethod
    def merge_defaults(cfg):
        cfg_copy = DatasetBuilder.copy_config(cfg)
        text_format.Merge(DatasetBuilder.CONFIG_DEFAULTS_PROTO, cfg_copy)
        return cfg_copy

    @staticmethod
    def build_kitti_dataset(base_cfg,
                            use_defaults=True,
                            new_cfg=None) -> KittiDataset:
        """Builds a KittiDataset object using the provided configurations

        Args:
            base_cfg: a base dataset configuration
            use_defaults: whether to use the default config values
            new_cfg: (optional) a custom dataset configuration, no default
                values will be used, all config values must be provided

        Returns:
            KittiDataset object
        """
        cfg_copy = DatasetBuilder.copy_config(base_cfg)

        if use_defaults:
            # Use default values
            text_format.Merge(DatasetBuilder.CONFIG_DEFAULTS_PROTO, cfg_copy)

        if new_cfg:
            # Use new config values if provided
            cfg_copy.MergeFrom(new_cfg)

        return KittiDataset(cfg_copy)


def main():
    DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAIN_MINI)


if __name__ == '__main__':
    main()

"""KittiUtil unit test module."""

import numpy as np
import unittest

from wavedata.tools.obj_detection import obj_utils as obj_utils
from avod.builders.dataset_builder import DatasetBuilder


class KittiUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset_config = DatasetBuilder.copy_config(
            DatasetBuilder.KITTI_UNITTEST)

        cls.dataset = DatasetBuilder.build_kitti_dataset(dataset_config)
        cls.label_dir = cls.dataset.label_dir

    def test_create_slice_filter(self):
        # Test slice filtering between 0.2 and 2.0m on three points located
        # at y=[0.0, 1.0, 3.0] with a flat ground plane along y

        # Create fake point cloud
        point_cloud = np.array([[1.0, 1.0, 1.0],
                                [0.0, 1.0, 3.0],
                                [1.0, 1.0, 1.0]])

        area_extents = [[-2, 2], [-5, 5], [-2, 2]]
        ground_plane = [0, 1, 0, 0]

        ground_offset_dist = 0.2
        offset_dist = 2.0

        expected_slice_filter = [False, True, False]

        slice_filter = self.dataset.kitti_utils.create_slice_filter(
            point_cloud, area_extents, ground_plane,
            ground_offset_dist, offset_dist)

        np.testing.assert_equal(slice_filter, expected_slice_filter)

    def test_rotate_map_90_degrees(self):
        # Check that a transpose and flip returns the same ndarray as np.rot90
        # This logic is part of create_bev_images

        np.random.seed(123)
        fake_bev_map = np.random.rand(800, 700)

        # Rotate with a transpose then flip (faster than np.rot90)
        np_transpose_then_flip_out = np.flip(fake_bev_map.transpose(), axis=0)

        # Expected result from np.rot90
        np_rot_90_out = np.rot90(fake_bev_map)

        np.testing.assert_allclose(np_transpose_then_flip_out,
                                   np_rot_90_out)

    def test_filter_labels_by_class(self):

        sample_name = '000007'
        obj_labels = obj_utils.read_labels(self.label_dir,
                                           int(sample_name))
        # This particular sample has 2 valid classes
        exp_num_valid_classes = 2

        filtered_labels = \
            self.dataset.kitti_utils.filter_labels(obj_labels, difficulty=None)
        all_types = []
        for label in filtered_labels:
            if label.type not in all_types:
                all_types.append(label.type)
        self.assertEqual(len(all_types),
                         exp_num_valid_classes,
                         msg='Wrong number of labels after filtering')


if __name__ == '__main__':
    unittest.main()

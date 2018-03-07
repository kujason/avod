import unittest

import numpy as np

from avod.datasets.kitti import kitti_aug


class KittiAugTest(unittest.TestCase):

    def test_flip_boxes_3d(self):

        boxes_3d = np.array([
            [1, 2, 3, 4, 5, 6, np.pi / 4],
            [1, 2, 3, 4, 5, 6, -np.pi / 4]
        ])

        exp_flipped_boxes_3d = np.array([
            [-1, 2, 3, 4, 5, 6, 3 * np.pi / 4],
            [-1, 2, 3, 4, 5, 6, -3 * np.pi / 4]
        ])

        flipped_boxes_3d = kitti_aug.flip_boxes_3d(boxes_3d)

        np.testing.assert_almost_equal(flipped_boxes_3d, exp_flipped_boxes_3d)

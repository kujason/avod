import unittest

import numpy as np

from avod.core import box_3d_projector


class Box3dProjectorTest(unittest.TestCase):
    def test_project_to_bev(self):
        boxes_3d = np.array([[0, 0, 0, 1, 0.5, 1, 0],
                             [0, 0, 0, 1, 0.5, 1, np.pi / 2],
                             [1, 0, 1, 1, 0.5, 1, np.pi / 2]])

        box_points, box_points_norm = \
            box_3d_projector.project_to_bev(boxes_3d, [[-1, 1], [-1, 1]])

        expected_boxes = np.array(
            [[[0.5, 0.25],
              [-0.5, 0.25],
              [-0.5, -0.25],
              [0.5, -0.25]],
             [[0.25, -0.5],
              [0.25, 0.5],
              [-0.25, 0.5],
              [-0.25, -0.5]],
             [[1.25, 0.5],
              [1.25, 1.5],
              [0.75, 1.5],
              [0.75, 0.5]]],
            dtype=np.float32)

        for box, exp_box in zip(box_points, expected_boxes):
            np.testing.assert_allclose(box, exp_box, rtol=1E-5)

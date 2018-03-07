import unittest

import numpy as np
import tensorflow as tf
from avod.core import box_3d_encoder


class Box3dEncoderTest(unittest.TestCase):
    def test_box_3d_to_anchor(self):
        # box_3d format is [x, y, z, l, w, h, ry]
        box_3d = np.asarray([[1, 2, 3, 4, 5, 6, 0],
                             [0, 0, 0, 1, 2, 3, 0],
                             [0, 0, 0, 1, 2, 3, np.pi / 2]],
                            dtype=np.float64)

        # anchor format is [x, y, z, dim_x, dim_y, dim_z]
        expected_anchors = np.asarray([[1, 2, 3, 4, 6, 5],
                                       [0, 0, 0, 1, 3, 2],
                                       [0, 0, 0, 2, 3, 1]],
                                      dtype=np.float64)

        anchors = box_3d_encoder.box_3d_to_anchor(box_3d)
        np.testing.assert_allclose(anchors, expected_anchors)

    def test_box_3d_to_anchor_180_270(self):
        box_3d = np.asarray([[1, 2, 3, 4, 5, 6, np.pi],
                             [1, 2, 3, 4, 5, 6, 3 * np.pi / 2]],
                            dtype=np.float64)

        expected_anchors = np.asarray([[1, 2, 3, 4, 6, 5],
                                       [1, 2, 3, 5, 6, 4]],
                                      dtype=np.float64)

        anchors = box_3d_encoder.box_3d_to_anchor(box_3d)
        np.testing.assert_allclose(anchors, expected_anchors)

    def test_box_3d_to_anchor_rotated(self):
        """
        Check that rotated boxes are rotated to the nearest 90
            and that the dimensions do not change
        """
        # Boxes at ry = 144, 288 should give same results as ry = 180, 270
        box_3d = np.asarray([[1, 2, 3, 4, 5, 6, np.pi * 4 / 5],
                             [1, 2, 3, 4, 5, 6, 8 * np.pi / 5]],
                            dtype=np.float64)

        expected_anchors = np.asarray([[1, 2, 3, 4, 6, 5],
                                       [1, 2, 3, 5, 6, 4]],
                                      dtype=np.float64)

        anchors = box_3d_encoder.box_3d_to_anchor(box_3d, ortho_rotate=True)
        np.testing.assert_allclose(anchors, expected_anchors)

    def test_box_3d_to_anchor_projected(self):
        """
        Check that boxes are projected with ortho_rotate=False,
            and that projected boxes have the correct dimensions
        """
        thetas = np.arange(0, 2 * np.pi, np.pi / 6)

        boxes_3d = []
        for theta in thetas:
            boxes_3d.append([1, 2, 3, 4, 5, 6, theta])
        boxes_3d = np.asarray(boxes_3d, dtype=np.float64)

        cos_thetas = np.abs(np.cos(thetas))
        sin_thetas = np.abs(np.sin(thetas))

        expected_dims_x = 4 * cos_thetas + 5 * sin_thetas
        expected_dims_z = 4 * sin_thetas + 5 * cos_thetas

        expected_anchors = []
        for exp_x, exp_z in zip(expected_dims_x, expected_dims_z):
            expected_anchors.append([1, 2, 3, exp_x, 6, exp_z])
        expected_anchors = np.asarray(expected_anchors, np.float64)

        anchors = box_3d_encoder.box_3d_to_anchor(boxes_3d, ortho_rotate=False)
        np.testing.assert_allclose(anchors, expected_anchors)

    def test_anchor_to_box_3d(self):
        anchors = np.asarray([[-0.59, 1.90, 25.01, 3.2, 1.66, 1.61],
                              [-0.59, 1.90, 25.01, 1.61, 1.66, 3.2]],
                             dtype=np.float32)

        exp_3d_box = np.asarray([[-0.59, 1.90, 25.01, 3.2, 1.61, 1.66, 0],
                                 [-0.59, 1.90, 25.01, 3.2, 1.61, 1.66, -1.57]],
                                dtype=np.float32)

        anchor_boxes_3d = box_3d_encoder.anchors_to_box_3d(anchors,
                                                           fix_lw=True)

        np.testing.assert_almost_equal(anchor_boxes_3d,
                                       exp_3d_box,
                                       decimal=3,
                                       err_msg='Wrong anchor to box3D format')

    def test_anchor_tensor_to_box_3d(self):
        anchors = np.asarray([[-0.59, 1.90, 25.01, 3.2, 1.66, 1.61],
                              [-0.59, 1.90, 25.01, 1.61, 1.66, 3.2]],
                             dtype=np.float32)

        exp_3d_box = np.asarray([[-0.59, 1.90, 25.01, 3.2, 1.61, 1.66, 0],
                                 [-0.59, 1.90, 25.01, 3.2, 1.61, 1.66, -1.57]],
                                dtype=np.float32)

        anchor_tensors = tf.convert_to_tensor(anchors, dtype=tf.float32)

        boxes_3d = \
            box_3d_encoder.anchors_to_box_3d(anchor_tensors,
                                             fix_lw=True)

        sess = tf.Session()
        with sess.as_default():
            boxes_3d_out = boxes_3d.eval()
            np.testing.assert_almost_equal(
                boxes_3d_out, exp_3d_box, decimal=3,
                err_msg='Wrong tensor anchor to box3D format')

    def test_box_3d_tensor_to_anchor(self):
        boxes_3d = np.asarray(
            [[-0.59, 1.90, 25.01, 3.2, 1.61, 1.66, 0],
             [-0.59, 1.90, 25.01, 3.2, 1.6, 1.66, -np.pi / 2]],
            dtype=np.float32)

        exp_anchors = np.asarray(
            [[-0.59, 1.90, 25.01, 3.2, 1.66, 1.61],
             [-0.59, 1.90, 25.01, 1.6, 1.66, 3.20]],
            dtype=np.float32)

        boxes_3d_tensors = tf.convert_to_tensor(boxes_3d,
                                                dtype=tf.float32)

        anchor_boxes_3d = box_3d_encoder.tf_box_3d_to_anchor(boxes_3d_tensors)

        sess = tf.Session()
        with sess.as_default():
            anchors_out = anchor_boxes_3d.eval()
            np.testing.assert_almost_equal(
                anchors_out, exp_anchors, decimal=2,
                err_msg='Wrong tensor anchor to box3D format')

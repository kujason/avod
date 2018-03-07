import unittest
import numpy as np
import tensorflow as tf

from avod.core import box_4c_encoder


class Box4cEncoderTest(unittest.TestCase):

    def test_np_box_3d_to_box_4c(self):
        # Test non-vectorized numpy version on ortho boxes

        # Sideways box
        box_3d_1 = np.asarray([0, 0, 0, 2, 1, 5, 0])
        # Straight box
        box_3d_2 = np.asarray([0, 0, 0, 2, 1, 5, -np.pi / 2])

        # Ground plane facing upwards, at 2m along y axis
        ground_plane = [0, -1, 0, 2]

        exp_box_4c_1 = np.asarray(
            [1.0, 1.0, -1.0, -1.0,
             0.5, -0.5, -0.5, 0.5,
             2.0, 7.0])
        exp_box_4c_2 = np.asarray(
            [0.5, 0.5, -0.5, -0.5,
             1.0, -1.0, -1.0, 1.0,
             2.0, 7.0])

        # Convert box_3d to box_4c
        box_4c_1 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_1, ground_plane)
        box_4c_2 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_2, ground_plane)

        np.testing.assert_almost_equal(box_4c_1, exp_box_4c_1, decimal=3)
        np.testing.assert_almost_equal(box_4c_2, exp_box_4c_2, decimal=3)

    def test_np_box_3d_to_box_4c_rotated_translated(self):
        # Test non-vectorized numpy version on rotated boxes
        box_3d_1 = np.asarray([0.0, 0.0, 0.0, 2.0, 1.0, 5.0, -1 * np.pi / 8])
        box_3d_2 = np.asarray([0.0, 0.0, 0.0, 2.0, 1.0, 5.0, -3 * np.pi / 8])
        box_3d_3 = np.asarray([0.0, 0.0, 0.0, 2.0, 1.0, 5.0, -5 * np.pi / 8])
        box_3d_4 = np.asarray([0.0, 0.0, 0.0, 2.0, 1.0, 5.0, -7 * np.pi / 8])
        box_3d_5 = np.asarray([0.0, 0.0, 0.0, 2.0, 1.0, 5.0, 1 * np.pi / 8])
        box_3d_6 = np.asarray([0.0, 0.0, 0.0, 2.0, 1.0, 5.0, 3 * np.pi / 8])
        box_3d_7 = np.asarray([0.0, 0.0, 0.0, 2.0, 1.0, 5.0, 5 * np.pi / 8])
        box_3d_8 = np.asarray([0.0, 0.0, 0.0, 2.0, 1.0, 5.0, 7 * np.pi / 8])

        # Also test a box translated along xz
        box_3d_translated = box_3d_1 + [10, 0, 10, 0, 0, 0, 0]

        # Ground plane facing upwards, at 2m along y axis
        ground_plane = [0, -1, 0, 2]

        # Convert box_3d to box_4c
        box_4c_1 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_1, ground_plane)
        box_4c_2 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_2, ground_plane)
        box_4c_3 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_3, ground_plane)
        box_4c_4 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_4, ground_plane)
        box_4c_5 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_5, ground_plane)
        box_4c_6 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_6, ground_plane)
        box_4c_7 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_7, ground_plane)
        box_4c_8 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_8, ground_plane)

        box_4c_translated = box_4c_encoder.np_box_3d_to_box_4c(
            box_3d_translated, ground_plane)

        # Expected boxes_4c
        exp_box_4c_1 = [0.733, 1.115, -0.733, -1.115,
                        0.845, -0.079, -0.845, 0.079,
                        2.000, 7.000]
        exp_box_4c_2 = [0.845, 0.079, -0.845, -0.079,
                        0.733, -1.115, -0.733, 1.115,
                        2.000, 7.000]
        exp_box_4c_3 = [0.079, 0.845, -0.079, -0.845,
                        1.115, -0.733, -1.115, 0.733,
                        2.000, 7.000]
        exp_box_4c_4 = [1.115, 0.733, -1.115, -0.733,
                        0.079, -0.845, -0.079, 0.845,
                        2.000, 7.000]
        exp_box_4c_5 = [1.115, 0.733, -1.115, -0.733,
                        0.079, -0.845, -0.079, 0.845,
                        2.000, 7.000]
        exp_box_4c_6 = [0.079, 0.845, -0.079, -0.845,
                        1.115, -0.733, -1.115, 0.733,
                        2.000, 7.000]
        exp_box_4c_7 = [0.845, 0.079, -0.845, -0.079,
                        0.733, -1.115, -0.733, 1.115,
                        2.000, 7.000]
        exp_box_4c_8 = [0.733, 1.115, -0.733, -1.115,
                        0.845, -0.079, -0.845, 0.079,
                        2.000, 7.000]
        exp_box_4c_translated = [10.733, 11.115, 9.267, 8.885,
                                 10.845, 9.921, 9.155, 10.079,
                                 2.000, 7.000]

        np.testing.assert_almost_equal(box_4c_1, exp_box_4c_1, decimal=3)
        np.testing.assert_almost_equal(box_4c_2, exp_box_4c_2, decimal=3)
        np.testing.assert_almost_equal(box_4c_3, exp_box_4c_3, decimal=3)
        np.testing.assert_almost_equal(box_4c_4, exp_box_4c_4, decimal=3)
        np.testing.assert_almost_equal(box_4c_5, exp_box_4c_5, decimal=3)
        np.testing.assert_almost_equal(box_4c_6, exp_box_4c_6, decimal=3)
        np.testing.assert_almost_equal(box_4c_7, exp_box_4c_7, decimal=3)
        np.testing.assert_almost_equal(box_4c_8, exp_box_4c_8, decimal=3)
        np.testing.assert_almost_equal(box_4c_translated,
                                       exp_box_4c_translated, decimal=3)

    def test_np_box_3d_to_box_4c_heights(self):
        # Boxes above, on, or below ground plane
        box_3d_1 = np.asarray([0.0, 3.0, 0.0, 2.0, 1.0, 5.0, 0.0])  # below
        box_3d_2 = np.asarray([0.0, 2.0, 0.0, 2.0, 1.0, 5.0, 0.0])  # on
        box_3d_3 = np.asarray([0.0, 1.0, 0.0, 2.0, 1.0, 5.0, 0.0])  # above

        # Ground plane facing upwards, at 2m along y axis
        ground_plane = [0, -1, 0, 2]

        # Convert box_3d to box_4c
        box_4c_1 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_1, ground_plane)
        box_4c_2 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_2, ground_plane)
        box_4c_3 = box_4c_encoder.np_box_3d_to_box_4c(box_3d_3, ground_plane)

        # Expected boxes_4c
        exp_box_4c_1 = np.asarray([1.0, 1.0, -1.0, -1.0,
                                   0.5, -0.5, -0.5, 0.5,
                                   -1.0, 4.0])
        exp_box_4c_2 = np.asarray([1.0, 1.0, -1.0, -1.0,
                                   0.5, -0.5, -0.5, 0.5,
                                   0.0, 5.0])
        exp_box_4c_3 = np.asarray([1.0, 1.0, -1.0, -1.0,
                                   0.5, -0.5, -0.5, 0.5,
                                   1.0, 6.0])

        np.testing.assert_almost_equal(box_4c_1, exp_box_4c_1)
        np.testing.assert_almost_equal(box_4c_2, exp_box_4c_2)
        np.testing.assert_almost_equal(box_4c_3, exp_box_4c_3)

    def test_tf_box_3d_to_box_4c(self):
        # Test that tf version matches np version
        # (rotations, xz translation, heights)
        boxes_3d = np.asarray([
            # Rotated
            [0.0, 0.0, 0.0, 2.0, 1.0, 5.0, -1 * np.pi / 8],
            [0.0, 0.0, 0.0, 2.0, 1.0, 5.0, -3 * np.pi / 8],
            [0.0, 0.0, 0.0, 2.0, 1.0, 5.0, -5 * np.pi / 8],
            [0.0, 0.0, 0.0, 2.0, 1.0, 5.0, -7 * np.pi / 8],
            [0.0, 0.0, 0.0, 2.0, 1.0, 5.0, 1 * np.pi / 8],
            [0.0, 0.0, 0.0, 2.0, 1.0, 5.0, 3 * np.pi / 8],
            [0.0, 0.0, 0.0, 2.0, 1.0, 5.0, 5 * np.pi / 8],
            [0.0, 0.0, 0.0, 2.0, 1.0, 5.0, 7 * np.pi / 8],

            # Translated along xz
            [10, 0, 5, 2, 1, 5, - 1 * np.pi / 8],

            # Below, on, or above ground plane
            [0.0, 3.0, 0.0, 2.0, 1.0, 5.0, 0.0],
            [0.0, 2.0, 0.0, 2.0, 1.0, 5.0, 0.0],
            [0.0, 1.0, 0.0, 2.0, 1.0, 5.0, 0.0],
        ])

        # Ground plane facing upwards, at 2m along y axis
        ground_plane = [0, -1, 0, 2]

        # Numpy conversion box_3d to box_4c
        np_boxes_4c = np.asarray(
            [box_4c_encoder.np_box_3d_to_box_4c(box_3d, ground_plane)
             for box_3d in boxes_3d])

        # Convert to tensors
        tf_boxes_3d = tf.convert_to_tensor(boxes_3d, dtype=tf.float32)
        tf_ground_plane = tf.convert_to_tensor(ground_plane, dtype=tf.float32)

        # Tensorflow conversion box_3d to box_4c
        tf_boxes_4c = box_4c_encoder.tf_box_3d_to_box_4c(tf_boxes_3d,
                                                         tf_ground_plane)

        sess = tf.Session()
        with sess.as_default():
            tf_boxes_4c_out = tf_boxes_4c.eval()

            # Loop through to show a separate error when box doesn't match
            for box_idx in range(len(np_boxes_4c)):
                np.testing.assert_almost_equal(np_boxes_4c[box_idx],
                                               tf_boxes_4c_out[box_idx],
                                               decimal=5)

    def test_np_box_4c_to_box_3d(self):
        box_4c_1 = np.asarray([1.0, 0.0, -1.0, 0.5,
                               0.5, -1.0, 0.0, 1.0,
                               1.0, 3.0])

        box_4c_2 = np.asarray([1.0, 0.0, -1.0, -0.5,
                               0.0, -1.0, 0.5, 1.0,
                               1.0, 3.0])

        ground_plane = np.asarray([0, -1, 0, 2])

        box_3d_1 = box_4c_encoder.np_box_4c_to_box_3d(box_4c_1, ground_plane)
        box_3d_2 = box_4c_encoder.np_box_4c_to_box_3d(box_4c_2, ground_plane)

        # Expected boxes_3d
        exp_box_3d_1 = [0.125, 1.000, 0.125, 1.768, 1.414, 2.000, -0.785]
        exp_box_3d_2 = [-0.125, 1.000, 0.125, 1.768, 1.414, 2.000, 0.785]

        np.testing.assert_almost_equal(box_3d_1, exp_box_3d_1, decimal=3)
        np.testing.assert_almost_equal(box_3d_2, exp_box_3d_2, decimal=3)

    def test_tf_box_4c_to_box_3d(self):
        np_boxes_4c = np.asarray(
            [
                [1.0, 0.0, -1.0, 0.5, 0.5, -1.0, 0.0, 1.0, 1.0, 3.0],
                [1.0, 0.0, -1.0, -0.5, 0.0, -1.0, 0.5, 1.0, 1.0, 3.0],
                [1.0, 0.0, -1.0, -0.5, 0.0, -1.0, 0.5, 1.0, 1.0, 3.0],
                [1.0, 0.0, -1.0, -0.5, 0.0, -1.0, 0.5, 1.0, 1.0, 3.0],
                [1.0, 0.0, -1.0, -0.5, 0.0, -1.0, 0.5, 1.0, 1.0, 3.0],
            ])

        np_ground_plane = np.asarray([0, -1, 0, -1])

        np_boxes_3d = [box_4c_encoder.np_box_4c_to_box_3d(box_4c,
                                                          np_ground_plane)
                       for box_4c in np_boxes_4c]

        tf_boxes_4c = tf.convert_to_tensor(np_boxes_4c,
                                           dtype=tf.float32)
        tf_ground_plane = tf.convert_to_tensor(np_ground_plane,
                                               dtype=tf.float32)

        tf_boxes_3d = box_4c_encoder.tf_box_4c_to_box_3d(tf_boxes_4c,
                                                         tf_ground_plane)

        sess = tf.Session()
        with sess.as_default():
            tf_boxes_3d_out = tf_boxes_3d.eval()

            for box_idx in range(len(np_boxes_3d)):
                np.testing.assert_almost_equal(np_boxes_3d[box_idx],
                                               tf_boxes_3d_out[box_idx],
                                               decimal=3)

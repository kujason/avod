import unittest
import numpy as np
import tensorflow as tf

from avod.core import anchor_encoder


class AnchorEncoderTest(unittest.TestCase):

    def test_anchor_to_offset(self):

        # anchor format is [x, y, z, dim_x, dim_y, dim_z]
        anchors = np.asarray([[1, 2, 3, 4, 6, 5],
                              [0, 0, 0, 2, 3, 1]], dtype=np.float32)

        # same formatting goes for the labels
        # which are also in anchor format
        anchors_gt =\
            np.array([2.0,  1.5,  7.0,  1.0,  0.5,  1.8])

        expected_offsets = np.array(
            [[0.25, -0.083, 0.8, -1.386, -2.484, -1.022],
             [1., 0.5, 7., -0.693, -1.791, 0.588]],
            dtype=np.float32)

        anchor_offsets = anchor_encoder.anchor_to_offset(anchors,
                                                         anchors_gt)
        np.testing.assert_almost_equal(anchor_offsets,
                                       expected_offsets,
                                       decimal=3)

    def test_anchor_tensor_to_offset(self):

        # anchor format is [x, y, z, dim_x, dim_y, dim_z]
        anchors = np.asarray([[1, 2, 3, 4, 6, 5],
                              [0, 0, 0, 2, 3, 1]], dtype=np.float32)

        anchors_tensor = \
            tf.convert_to_tensor(anchors, dtype=tf.float32)

        # we expect this in matrix format for the tensor version
        # of this function. In this case, it's just a repeated
        # gt associated with each anchor
        anchors_gt =\
            np.array([[2.0,  1.5,  7.0,  1.0,  0.5,  1.8],
                      [2.0,  1.5,  7.0,  1.0,  0.5,  1.8]])

        anchors_gt_tensor = \
            tf.convert_to_tensor(anchors_gt, dtype=tf.float32)

        expected_offsets = np.array(
            [[0.25, -0.083, 0.8, -1.386, -2.484, -1.022],
             [1., 0.5, 7., -0.693, -1.791, 0.588]],
            dtype=np.float32)

        # test in tensor space
        anchor_offsets = anchor_encoder.tf_anchor_to_offset(anchors_tensor,
                                                            anchors_gt_tensor)

        sess = tf.Session()
        with sess.as_default():
            anchor_offsets_out = anchor_offsets.eval()
            np.testing.assert_almost_equal(anchor_offsets_out,
                                           expected_offsets,
                                           decimal=3)

    def test_offset_to_anchor(self):

        # anchor format is [x, y, z, dim_x, dim_y, dim_z]
        anchors = np.asarray([[1, 2, 3, 4, 6, 5],
                              [0, 0, 0, 2, 3, 1]], dtype=np.float32)

        # anchor offset prediction is [tx, ty, tz, tdim_x, tdim_y, tdim_z]
        anchor_offsets = np.array(
            [[0.5, 0.02, 0.01, 0.1, 0.4, 0.03],
             [0.04, 0.1, 0.03, 0.001, 0.3, 0.03]],
            dtype=np.float32)

        expected_anchors = np.array(
            [[3.0, 2.12, 3.05, 4.420, 8.9509, 5.152],
             [0.08, 0.3, 0.03, 2.002, 4.05, 1.03]],
            dtype=np.float32)

        anchors = anchor_encoder.offset_to_anchor(anchors,
                                                  anchor_offsets)
        np.testing.assert_almost_equal(anchors,
                                       expected_anchors,
                                       decimal=3)

    def test_offset_tensor_to_anchor(self):

        # anchor format is [x, y, z, dim_x, dim_y, dim_z]
        anchors = np.asarray([[1, 2, 3, 4, 6, 5],
                              [0, 0, 0, 2, 3, 1]], dtype=np.float32)

        anchor_tensor = \
            tf.convert_to_tensor(anchors, dtype=tf.float32)

        # anchor offset prediction is [tx, ty, tz, tdim_x, tdim_y, tdim_z]
        anchor_offsets = np.array(
            [[0.5, 0.02, 0.01, 0.1, 0.4, 0.03],
             [0.04, 0.1, 0.03, 0.001, 0.3, 0.03]],
            dtype=np.float32)

        anchor_offset_tensor = \
            tf.convert_to_tensor(anchor_offsets, dtype=tf.float32)

        expected_anchors = np.array(
            [[3.0, 2.12, 3.05, 4.420, 8.9509, 5.152],
             [0.08, 0.3, 0.03, 2.002, 4.05, 1.03]],
            dtype=np.float32)

        anchors_tensor = anchor_encoder.offset_to_anchor(
            anchor_tensor, anchor_offset_tensor)

        sess = tf.Session()
        with sess.as_default():
            anchors = anchors_tensor.eval()

            np.testing.assert_almost_equal(anchors,
                                           expected_anchors,
                                           decimal=3)


if __name__ == '__main__':
    unittest.main()

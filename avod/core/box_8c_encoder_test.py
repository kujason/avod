import unittest
import numpy as np
import tensorflow as tf

from avod.core import box_8c_encoder
from avod.core import box_3d_encoder


class Box8cEncoderTest(unittest.TestCase):

    def test_box_3d_to_box_8co(self):
        # Tests the numpy version of the anchors_to_box_3d
        # function. This is the non-vectorized version.

        # Sample ground-truth in box3D format
        gt_box_3d = np.asarray(
            [-0.69, 1.69, 25.01, 3.2, 1.66, 1.61, -1.59],
            dtype=np.float32)
        # Sample box in anchor format
        anchors = np.asarray([[-0.59, 1.90, 25.01, 3.2, 1.61, 1.66]],
                             dtype=np.float32)
        # Convert the anchor to box3D format
        anchor_box_3d = box_3d_encoder.anchors_to_box_3d(anchors,
                                                         fix_lw=True)

        exp_gt_box_8co = np.asarray(
            [[-1.55, 0.10, 0.17, -1.49, -1.55, 0.11, 0.17, -1.49],
             [1.69, 1.69, 1.69, 1.69, 0.08, 0.08, 0.08, 0.08],
             [26.59, 26.62, 23.43, 23.39, 26.59, 26.62, 23.42, 23.39]])

        exp_anchor_box_8co = np.asarray(
            [[1.01, 1.01, -2.19, -2.19, 1.01, 1.01, -2.19, -2.19],
             [1.89, 1.89, 1.89, 1.89, 0.24, 0.24, 0.24, 0.24],
             [25.81, 24.21, 24.21, 25.82, 25.82, 24.21, 24.21, 25.82]])

        # convert to 8 corners
        gt_box_8co = box_8c_encoder.np_box_3d_to_box_8co(gt_box_3d)
        # the numpy version takes a single box
        anchor_box_8co = \
            box_8c_encoder.np_box_3d_to_box_8co(anchor_box_3d[0])

        np.testing.assert_almost_equal(exp_gt_box_8co,
                                       gt_box_8co,
                                       decimal=2,
                                       err_msg='GT corner encoding mismatch')

        np.testing.assert_almost_equal(
            exp_anchor_box_8co, anchor_box_8co, decimal=1,
            err_msg='Anchor corner encoding mismatch')

    def test_box_3d_tensor_to_box_8co(self):
        # Tests the tensor version of the anchors_to_box_3d
        # function. This is the vectorized version.

        anchors = np.asarray([
            [-0.59, 1.90, 25.01, 3.2, 1.66, 1.61],
            [-0.80, 1.50, 22.01, 1.2, 1.70, 1.50]
        ])
        anchor_boxes_3d = box_3d_encoder.anchors_to_box_3d(anchors,
                                                           fix_lw=True)

        # convert each box to corner using the numpy version
        boxes_8c_1 = \
            box_8c_encoder.np_box_3d_to_box_8co(anchor_boxes_3d[0])
        boxes_8c_2 = \
            box_8c_encoder.np_box_3d_to_box_8co(anchor_boxes_3d[1])

        exp_anchor_box_8co = np.stack((boxes_8c_1,
                                       boxes_8c_2),
                                      axis=0)

        anchors_box3d_tensor = tf.convert_to_tensor(anchor_boxes_3d,
                                                    dtype=tf.float32)
        # convert to 8 corners
        anchor_box_corner_tensor = \
            box_8c_encoder.tf_box_3d_to_box_8co(anchors_box3d_tensor)

        sess = tf.Session()
        with sess.as_default():
            anchor_box_corner_out = anchor_box_corner_tensor.eval()

        np.testing.assert_almost_equal(
            exp_anchor_box_8co[0], anchor_box_corner_out[0], decimal=2,
            err_msg='Anchor tensor corner encoding mismatch')

        np.testing.assert_almost_equal(
            exp_anchor_box_8co[1], anchor_box_corner_out[1], decimal=2,
            err_msg='Anchor tensor corner encoding mismatch')

    def test_gt_box_3d_tensor_to_8c(self):
        # This test is slightly different from above as it tests
        # the conversion of ground-truth boxes with *any* orientation
        # to 8-corner format.
        # Sample ground-truth in box3D format
        gt_boxes_3d = np.asarray([
            [-0.69, 1.69, 25.01, 3.2, 1.66, 1.61, -1.59],
            [-7.43, 1.88, 47.55, 3.7, 1.51, 1.4, 1.55]],
            dtype=np.float32)

        gt_boxes_3d_tensor = tf.convert_to_tensor(gt_boxes_3d,
                                                  dtype=tf.float32)

        boxes_8c_1 = box_8c_encoder.np_box_3d_to_box_8co(gt_boxes_3d[0])
        boxes_8c_2 = box_8c_encoder.np_box_3d_to_box_8co(gt_boxes_3d[1])

        exp_boxes_8c_gt = np.stack((boxes_8c_1,
                                    boxes_8c_2),
                                   axis=0)

        boxes_c8_gt = \
            box_8c_encoder.tf_box_3d_to_box_8co(gt_boxes_3d_tensor)

        sess = tf.Session()
        with sess.as_default():
            corner_box_8co_out = boxes_c8_gt.eval()

        np.testing.assert_almost_equal(
            exp_boxes_8c_gt[0], corner_box_8co_out[0], decimal=3,
            err_msg='Gt box 3D tensor to corner encoding mismatch')

        np.testing.assert_almost_equal(
            exp_boxes_8c_gt[1], corner_box_8co_out[1], decimal=3,
            err_msg='Gt box 3D tensor to corner encoding mismatch')

    def test_align_box_8co(self):
        # Given 8 corners of an irregular shape, tests whether
        # they get aligned correctly
        irregular_box_8co = \
            np.asarray([[[6.0, 6.0, 3.0, 4.0, 5.0, 5.0, 3.0, 2.0],
                         [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
                         [4.0, 1.0, 1.0, 6.0, 4.0, 1.0, 1.0, 4.0]]])

        exp_box_8co = np.asarray([[6.0, 6.0, 2.0, 2.0, 6.0, 6.0, 2.0, 2.0],
                                  [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                  [6.0, 1.0, 1.0, 6.0, 6.0, 1.0, 1.0, 6.0]])

        irregular_box_8co_tensor = tf.convert_to_tensor(irregular_box_8co,
                                                        dtype=tf.float32)

        aligned_box_8co = box_8c_encoder.align_boxes_8c(
            irregular_box_8co_tensor)

        sess = tf.Session()
        with sess.as_default():
            aligned_box_8co_out = aligned_box_8co.eval()

        np.testing.assert_array_equal(exp_box_8co,
                                      aligned_box_8co_out[0],
                                      err_msg='Wrong aligned corners.')

    def test_box3d_to_box8c_and_back(self):

        boxes_3d = np.asarray([
            [-0.59, 1.9, 25.01, 3.2, 1.61, 1.66, 0.],
            [-0.69, 1.69, 25.01, 3.2, 1.66, 1.61, -1.59]],
            dtype=np.float32)

        box_8co_1 = box_8c_encoder.np_box_3d_to_box_8co(boxes_3d[0])
        box_8co_2 = box_8c_encoder.np_box_3d_to_box_8co(boxes_3d[1])

        boxes_8c = np.stack((box_8co_1,
                             box_8co_2),
                            axis=0)

        boxes_c8_tensor = tf.convert_to_tensor(boxes_8c,
                                               dtype=tf.float32)

        boxes_3d_tensor = box_8c_encoder.box_8c_to_box_3d(boxes_c8_tensor)

        sess = tf.Session()
        with sess.as_default():
            boxes_3d_out = boxes_3d_tensor.eval()

        np.testing.assert_almost_equal(boxes_3d[0],
                                       boxes_3d_out[0],
                                       decimal=2)
        np.testing.assert_almost_equal(boxes_3d[1],
                                       boxes_3d_out[1],
                                       decimal=2)

    def test_box_8co_and_box3d(self):

        # These corners are slightly skewed to test the
        # corner alignment during transformation
        exp_box_3d = np.asarray(
            [-0.59, 1.9, 25.01, 3.2, 1.61, 1.66, 0.],
            dtype=np.float32)

        box_8co = box_8c_encoder.np_box_3d_to_box_8co(exp_box_3d)

        # skew the corners slightly
        box_8co[0, 2] += 0.4
        box_8co[0, 6] += 0.9

        box_c8_tensor = tf.convert_to_tensor(box_8co,
                                             dtype=tf.float32)
        box_c8_tensor = tf.expand_dims(box_c8_tensor, axis=0)

        box_3d = box_8c_encoder.box_8c_to_box_3d(box_c8_tensor)

        sess = tf.Session()
        with sess.as_default():
            box_3d_out = box_3d.eval()

        np.testing.assert_almost_equal(box_3d_out[0],
                                       exp_box_3d,
                                       decimal=2)

    def test_tf_box_8co_diagonal_length(self):
        # Tests the calculation of diagonal for both box_3d
        # and box_8co
        boxes_3d = np.asarray([
            [-0.59, 1.9, 25.01, 3.2, 1.61, 1.66, 0.],
            [-0.69, 1.69, 25.01, 2.2, 1.70, 1.61, -1.59]],
            dtype=np.float32)

        exp_diagonal_box1 = 3.9481
        exp_diagonal_box2 = 3.2128

        boxes_3d_tensor = tf.convert_to_tensor(boxes_3d,
                                               dtype=tf.float32)

        boxes_c8_tensor = \
            box_8c_encoder.tf_box_3d_to_box_8co(boxes_3d_tensor)

        boxes_8c_diagonals = \
            box_8c_encoder.tf_box_8c_diagonal_length(boxes_c8_tensor)
        boxes_3d_diagonals = \
            box_3d_encoder.tf_box_3d_diagonal_length(boxes_3d_tensor)

        sess = tf.Session()
        with sess.as_default():
            boxes_8c_diagonals_out = boxes_8c_diagonals.eval()
            boxes_3d_diagonals_out = boxes_3d_diagonals.eval()

        # Test box_8co diagonals
        self.assertAlmostEqual(boxes_8c_diagonals_out[0],
                               exp_diagonal_box1,
                               places=4,
                               msg='Diagonal mistmatch')

        self.assertAlmostEqual(boxes_8c_diagonals_out[1],
                               exp_diagonal_box2,
                               places=4,
                               msg='Diagonal mistmatch')
        # Test box_3d diagonals
        self.assertAlmostEqual(boxes_3d_diagonals_out[0],
                               exp_diagonal_box1,
                               places=4,
                               msg='Diagonal mistmatch')

        self.assertAlmostEqual(boxes_3d_diagonals_out[1],
                               exp_diagonal_box2,
                               places=4,
                               msg='Diagonal mistmatch')

    def test_box_8co_to_offset(self):
        # tests corners to normalized offsets
        boxes_3d = np.asarray([
            [-0.59, 1.9, 25.01, 3.2, 1.61, 1.66, 0.]],
            dtype=np.float32)
        boxes_3d_gt = np.asarray([
            [-0.69, 1.69, 25.01, 3.2, 1.62, 1.61, -1.59]],
            dtype=np.float32)

        exp_normalized_offsets = np.asarray([
            [[-0.6434, -0.233, 0.592, 0.182,
              -0.643, -0.233, 0.592, 0.182],
             [-0.053, -0.0531, -0.0531, -0.0531,
              -0.0405, -0.0405, -0.0405, -0.0405],
             [0.197, 0.613, -0.197, -0.613,
              0.197, 0.613, -0.197, -0.613]]
        ])

        boxes_3d_tensor = tf.convert_to_tensor(boxes_3d,
                                               dtype=tf.float32)

        boxes_3d_gt_tensor = tf.convert_to_tensor(boxes_3d_gt,
                                                  dtype=tf.float32)

        boxes_8c = \
            box_8c_encoder.tf_box_3d_to_box_8co(boxes_3d_tensor)

        boxes_8c_gt = \
            box_8c_encoder.tf_box_3d_to_box_8co(boxes_3d_gt_tensor)

        normalized_offsets = box_8c_encoder.tf_box_8c_to_offsets(
            boxes_8c, boxes_8c_gt)

        sess = tf.Session()
        with sess.as_default():
            normalized_offsets_out = normalized_offsets.eval()

        np.testing.assert_almost_equal(normalized_offsets_out,
                                       exp_normalized_offsets,
                                       decimal=3,
                                       err_msg='box_8co to offsets mistmatch')

    def test_box_offset_to_8c(self):
        # tests corners to normalized offsets
        boxes_3d = np.asarray([
            [-0.59, 1.9, 25.01, 3.2, 1.61, 1.66, 0.],
            [-0.49, 1.8, 25.01, 3.2, 1.61, 1.78, 0.]],
            dtype=np.float32)
        boxes_3d_gt = np.asarray([
            [-0.69, 1.69, 25.01, 3.2, 1.62, 1.61, -1.59],
            [-0.49, 1.78, 25.01, 3.2, 1.63, 1.61, -1.57]],
            dtype=np.float32)

        boxes_3d_tensor = tf.convert_to_tensor(boxes_3d,
                                               dtype=tf.float32)

        boxes_3d_gt_tensor = tf.convert_to_tensor(boxes_3d_gt,
                                                  dtype=tf.float32)

        boxes_8c = \
            box_8c_encoder.tf_box_3d_to_box_8co(boxes_3d_tensor)

        boxes_8c_gt = \
            box_8c_encoder.tf_box_3d_to_box_8co(boxes_3d_gt_tensor)

        # Convert to normalized offsets
        normalized_offsets = box_8c_encoder.tf_box_8c_to_offsets(
            boxes_8c, boxes_8c_gt)

        # Convert the offsets back to boxes_8c
        # This should gives us the gt boxes back
        boxes_8c_gt_back = box_8c_encoder.tf_offsets_to_box_8c(
            boxes_8c, normalized_offsets)

        sess = tf.Session()
        with sess.as_default():
            boxes_8c_out = boxes_8c_gt_back.eval()
            boxes_8c_gt_out = boxes_8c_gt.eval()

        np.testing.assert_almost_equal(boxes_8c_out,
                                       boxes_8c_gt_out,
                                       decimal=3,
                                       err_msg='Offset to box_8co mistmatch')

    def test_np_box_8c_conversion(self):
        # Box at 180 orientation
        boxes_3d = np.asarray([
            [-0.59, 1.9, 25.01, 3.2, 1.61, 1.66, 3.14]],
            dtype=np.float32)
        # Box at 0 orientation
        boxes_3d_gt = np.asarray([
            [-0.59, 1.9, 25.01, 3.2, 1.61, 1.66, 0.]],
            dtype=np.float32)

        # Convert to box_8c
        box_8c = box_8c_encoder.np_box_3d_to_box_8c(boxes_3d[0])
        box_8c_gt = box_8c_encoder.np_box_3d_to_box_8c(boxes_3d_gt[0])

        np.testing.assert_almost_equal(
            box_8c, box_8c_gt, decimal=2,
            err_msg='Wrong box_8c nonordered corners.')

    def test_tf_box_8c_conversion(self):
        # Box at 180 orientation
        boxes_3d = np.asarray([
            [-0.59, 1.9, 25.01, 3.2, 1.61, 1.66, 3.14]],
            dtype=np.float32)
        # Box at 0 orientation
        boxes_3d_gt = np.asarray([
            [-0.59, 1.9, 25.01, 3.2, 1.61, 1.66, 0.]],
            dtype=np.float32)

        boxes_3d_tensor = tf.convert_to_tensor(boxes_3d,
                                               dtype=tf.float32)

        boxes_3d_gt_tensor = tf.convert_to_tensor(boxes_3d_gt,
                                                  dtype=tf.float32)

        # Convert to box_8c
        boxes_8c = box_8c_encoder.tf_box_3d_to_box_8c(boxes_3d_tensor)
        boxes_8c_gt = box_8c_encoder.tf_box_3d_to_box_8c(boxes_3d_gt_tensor)

        sess = tf.Session()
        with sess.as_default():
            boxes_8c_out = boxes_8c.eval()
            boxes_8c_gt_out = boxes_8c_gt.eval()

        np.testing.assert_almost_equal(
            boxes_8c_out, boxes_8c_gt_out, decimal=2,
            err_msg='Wrong box_8c nonordered corners.')


if __name__ == '__main__':
    unittest.main()

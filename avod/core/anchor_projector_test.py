import unittest

import numpy as np
import tensorflow as tf

from wavedata.tools.core import calib_utils

import avod.tests as tests
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import anchor_projector


class AnchorProjectorTest(unittest.TestCase):
    def test_project_to_bev(self):
        anchors = np.asarray([[1, 0, 3, 2, 0, 6],
                              [3, 0, 3, 2, 0, 2]],
                             dtype=np.float64)

        bev_extents = [[0, 5], [0, 10]]
        bev_extents_range = np.diff(bev_extents, axis=1)
        bev_extents_range = np.stack([bev_extents_range,
                                      bev_extents_range]).flatten()

        expected_boxes = np.asarray([[0, 4, 2, 10],
                                     [2, 6, 4, 8]],
                                    dtype=np.float64)
        expected_boxes_norm = expected_boxes / bev_extents_range

        boxes, boxes_norm = \
            anchor_projector.project_to_bev(anchors, bev_extents)

        # Loop through cases to see errors separately
        for box, box_norm, \
            exp_box, exp_box_norm in zip(boxes, boxes_norm,
                                         expected_boxes,
                                         expected_boxes_norm):
            np.testing.assert_allclose(box, exp_box, rtol=1E-5)
            np.testing.assert_allclose(box_norm, exp_box_norm, rtol=1E-5)

    def test_project_to_bev_extents(self):
        anchors = np.asarray([[0, 0, 3, 2, 0, 6],
                              [3, 0, 3, 2, 0, 2]],
                             dtype=np.float64)

        bev_extents = [[-5, 5], [0, 10]]
        bev_extents_range = np.diff(bev_extents, axis=1)
        bev_extents_range = np.stack([bev_extents_range,
                                      bev_extents_range]).flatten()

        expected_boxes = np.asarray([[0 - (-5) - 1, 4, 0 - (-5) + 1, 10],
                                     [3 - (-5) - 1, 6, 3 - (-5) + 1, 8]],
                                    dtype=np.float64)
        expected_boxes_norm = expected_boxes / bev_extents_range

        boxes, boxes_norm = \
            anchor_projector.project_to_bev(anchors, bev_extents)

        # Loop through cases to see errors separately
        for box, box_norm, \
            exp_box, exp_box_norm in zip(boxes, boxes_norm,
                                         expected_boxes,
                                         expected_boxes_norm):
            np.testing.assert_allclose(box, exp_box, rtol=1E-5)
            np.testing.assert_allclose(box_norm, exp_box_norm, rtol=1E-5)

    def test_project_to_bev_outside_extents(self):
        anchors = np.asarray([[0, 0, 0, 10, 0, 2]],
                             dtype=np.float64)

        bev_extents = [[-3, 3], [0, 10]]
        bev_extents_range = np.diff(bev_extents, axis=1)
        bev_extents_range = np.stack([bev_extents_range,
                                      bev_extents_range]).flatten()

        expected_boxes = np.asarray([[0 - (-3) - 5, 9, 0 - (-3) + 5, 11]],
                                    dtype=np.float64)
        expected_boxes_norm = expected_boxes / bev_extents_range

        boxes, boxes_norm = \
            anchor_projector.project_to_bev(anchors, bev_extents)

        # Loop through cases to see errors separately
        for box, box_norm, \
            exp_box, exp_box_norm in zip(boxes, boxes_norm,
                                         expected_boxes,
                                         expected_boxes_norm):
            np.testing.assert_allclose(box, exp_box, rtol=1E-5)
            np.testing.assert_allclose(box_norm, exp_box_norm, rtol=1E-5)

    def test_project_to_bev_tensors(self):
        anchors = np.asarray([[0, 0, 3, 2, 0, 6],
                              [3, 0, 3, 2, 0, 2]],
                             dtype=np.float64)
        tf_anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)

        bev_extents = [[-5, 5], [0, 10]]
        tf_bev_extents = tf.convert_to_tensor(bev_extents, dtype=tf.float32)

        bev_extents_range = np.diff(bev_extents, axis=1)
        bev_extents_range = np.stack([bev_extents_range,
                                      bev_extents_range]).flatten()

        expected_boxes = np.asarray([[0 - (-5) - 1, 4, 0 - (-5) + 1, 10],
                                     [3 - (-5) - 1, 6, 3 - (-5) + 1, 8]],
                                    dtype=np.float64)
        expected_boxes_norm = expected_boxes / bev_extents_range

        tf_boxes, tf_boxes_norm = \
            anchor_projector.project_to_bev(tf_anchors, tf_bev_extents)

        np_boxes, np_boxes_norm = \
            anchor_projector.project_to_bev(anchors, bev_extents)

        sess = tf.Session()
        with sess.as_default():
            tf_boxes_out = tf_boxes.eval()
            tf_boxes_norm_out = tf_boxes_norm.eval()

            np.testing.assert_allclose(tf_boxes_out, expected_boxes)
            np.testing.assert_allclose(tf_boxes_norm_out, expected_boxes_norm)

            # Check that tensor calculations match numpy ones
            np.testing.assert_allclose(tf_boxes_out, np_boxes)
            np.testing.assert_allclose(tf_boxes_norm_out, np_boxes_norm)

    def test_3d_to_2d_point_projection(self):

        anchor_corners = np.asarray([[1., 1., -1., -1., 1., 1., -1., -1.,
                                      4., 4., 2., 2., 4., 4., 2., 2.],
                                     [0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0.],
                                     [6., 0., 0., 6., 6., 0., 0., 6.,
                                      4., 2., 2., 4., 4., 2., 2., 4.]])

        stereo_calib_p2 = \
            np.asarray([[7.21537700e+02, 0.0, 6.09559300e+02, 4.48572800e+01],
                        [0.0, 7.21537700e+02, 1.72854000e+02, 2.16379100e-01],
                        [0.0, 0.0, 1.0, 2.74588400e-03]])

        # Do projection in numpy space
        points_2d = calib_utils.project_to_image(
            anchor_corners, stereo_calib_p2)

        # Do projection in tensor space
        tf_anchor_corners = tf.convert_to_tensor(anchor_corners,
                                                 dtype=tf.float32)
        tf_stereo_calib_p2 = tf.convert_to_tensor(stereo_calib_p2,
                                                  dtype=tf.float32)
        tf_points_2d = anchor_projector.project_to_image_tensor(
            tf_anchor_corners, tf_stereo_calib_p2)

        sess = tf.Session()
        with sess.as_default():
            points_2d_out = tf_points_2d.eval()
            np.testing.assert_allclose(
                points_2d, points_2d_out,
                err_msg='Incorrect tensor 3D->2D projection')

    def test_project_to_image_space_tensors(self):

        anchors = np.asarray([[0, 0, 3, 2, 0, 6],
                              [3, 0, 3, 2, 0, 2]],
                             dtype=np.float64)
        img_idx = int('000217')
        img_shape = [375, 1242]

        dataset_config = DatasetBuilder.copy_config(
            DatasetBuilder.KITTI_UNITTEST)

        dataset_config.data_split = 'train'
        dataset_config.dataset_dir = tests.test_path() + \
            "/datasets/Kitti/object"

        dataset = DatasetBuilder().build_kitti_dataset(dataset_config)

        stereo_calib_p2 = calib_utils.read_calibration(
            dataset.calib_dir, img_idx).p2

        # Project the 3D points in numpy space
        img_corners, img_corners_norm = anchor_projector.project_to_image_space(
            anchors, stereo_calib_p2, img_shape)

        # convert the required params to tensors
        tf_stereo_calib_p2 = tf.convert_to_tensor(stereo_calib_p2,
                                                  dtype=tf.float32)
        tf_anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
        tf_img_shape = tf.convert_to_tensor(img_shape, dtype=tf.float32)

        # Project the 3D points in tensor space
        img_corners_tensor, img_corners_norm_tensor = \
            anchor_projector.tf_project_to_image_space(tf_anchors,
                                                       tf_stereo_calib_p2,
                                                       tf_img_shape)

        sess = tf.Session()
        with sess.as_default():
            img_corners_out = img_corners_tensor.eval()
            img_corners_norm_out = img_corners_norm_tensor.eval()
            np.testing.assert_allclose(img_corners,
                                       img_corners_out,
                                       atol=1e-04,
                                       err_msg='Incorrect corner projection')
            np.testing.assert_allclose(
                img_corners_norm, img_corners_norm_out, atol=1e-04,
                err_msg='Incorrect normalized corner projection')

    def test_reorder_projected_boxes(self):

        box_corners = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        expected_tf_corners = np.array([[2, 1, 4, 3], [6, 5, 8, 7]])

        box_corner_tensor = tf.convert_to_tensor(box_corners)

        tf_corners = \
            anchor_projector.reorder_projected_boxes(box_corner_tensor)
        sess = tf.Session()
        with sess.as_default():
            tf_corners_out = tf_corners.eval()
            np.testing.assert_array_equal(
                tf_corners_out,
                expected_tf_corners,
                err_msg='Incorrect corner reordering')


if __name__ == '__main__':
    unittest.main()

import numpy as np
import tensorflow as tf

from avod.core import box_list
from avod.core import box_list_ops
from avod.builders.dataset_builder import DatasetBuilder


class MiniBatchUtilsTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = DatasetBuilder.build_kitti_dataset(
            DatasetBuilder.KITTI_UNITTEST)

        cls.mb_utils = cls.dataset.kitti_utils.mini_batch_utils

    def test_get_anchors_info(self):

        # Take the first non empty sample
        sample = self.dataset.sample_names[1]

        # Check the anchors info for first class type
        anchors_info = self.mb_utils.get_anchors_info(
            self.dataset.classes_name,
            self.dataset.kitti_utils.anchor_strides,
            sample)

        anchor_indices = anchors_info[0]
        anchor_ious = anchors_info[1]
        anchor_offsets = anchors_info[2]
        anchor_classes = anchors_info[3]

        # Lengths should all be the same
        self.assertTrue(len(anchor_indices), len(anchor_ious))
        self.assertTrue(len(anchor_indices), len(anchor_offsets))
        self.assertTrue(len(anchor_indices), len(anchor_classes))

        # Indices, IOUs, and classes values should all be >= 0
        self.assertTrue((anchor_indices >= 0).all())
        self.assertTrue((anchor_ious >= 0).all())
        self.assertTrue((anchor_classes >= 0).all())

        # Offsets should be (N, 6)
        self.assertTrue(len(anchor_offsets.shape) == 2)
        self.assertTrue(anchor_offsets.shape[1] == 6)

    def test_iou_mask_ops(self):
        # corners are in [y1, x1, y2, x2] format
        corners_pred = tf.constant(
            [[4.0, 3.0, 7.0, 5.0],
             [14.0, 14.0, 16.0, 16.0],
             [0.0, 0.0, 21.0, 19.0],
             [3.0, 4.0, 5.0, 7.0]])
        corners_gt = tf.constant(
            [[4.0, 3.0, 7.0, 6.0],
             [14.0, 14.0, 15.0, 15.0],
             [0.0, 0.0, 20.0, 20.0]])
        # 3 classes
        class_indices = tf.constant([1., 2., 3.])

        exp_ious = [[0.66666669, 0., 0.02255639, 0.15384616],
                    [0., 0.25, 0.00250627, 0.],
                    [0.015, 0.01, 0.90692127, 0.015]]

        exp_max_ious = np.array([0.66666669, 0.25, 0.90692127, 0.15384616])
        exp_max_indices = np.array([0, 1, 2, 0])

        exp_pos_mask = np.array([True, False, True, False])

        exp_class_and_background_indices = np.array([1, 0, 3, 0])

        # Convert to box_list format
        boxes_pred = box_list.BoxList(corners_pred)
        boxes_gt = box_list.BoxList(corners_gt)
        # Calculate IoU
        iou = box_list_ops.iou(boxes_gt,
                               boxes_pred)

        # Get max IoU, the dimension should match the anchors we are
        # evaluating
        max_ious = tf.reduce_max(iou, axis=0)
        max_iou_indices = tf.argmax(iou, axis=0)

        # Sample a mini-batch from anchors with highest IoU match
        mini_batch_size = 4

        # Custom positive/negative iou ranges
        neg_2d_iou_range = [0.0, 0.3]
        pos_2d_iou_range = [0.6, 0.7]

        mb_mask, mb_pos_mask = \
            self.mb_utils.sample_mini_batch(max_ious,
                                            mini_batch_size,
                                            neg_2d_iou_range,
                                            pos_2d_iou_range)

        mb_class_indices = self.mb_utils.mask_class_label_indices(
            mb_pos_mask, mb_mask, max_iou_indices, class_indices)

        with self.test_session() as sess:
            iou_out = sess.run(iou)
            max_ious_out, max_iou_indices_out = sess.run([max_ious,
                                                          max_iou_indices])
            mb_mask_out, mb_pos_mask_out = sess.run([mb_mask,
                                                     mb_pos_mask])
            class_indices_out = sess.run(mb_class_indices)

            self.assertAllClose(iou_out, exp_ious)
            self.assertAllClose(max_ious_out, exp_max_ious)
            self.assertAllEqual(max_iou_indices_out, exp_max_indices)
            self.assertAllEqual(exp_pos_mask, mb_pos_mask_out)
            self.assertAllEqual(class_indices_out,
                                exp_class_and_background_indices)


if __name__ == '__main__':
    tf.test.main()

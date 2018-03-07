# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for object_detection.core.box_list_ops."""
import tensorflow as tf

from avod.core import box_list
from avod.core import box_list_ops


class BoxListOpsTest(tf.test.TestCase):
    """Tests for common bounding box operations."""

    def test_area(self):
        corners = tf.constant([[0.0, 0.0, 10.0, 20.0], [1.0, 2.0, 3.0, 4.0]])
        exp_output = [200.0, 4.0]
        boxes = box_list.BoxList(corners)
        areas = box_list_ops.area(boxes)
        with self.test_session() as sess:
            areas_output = sess.run(areas)
            self.assertAllClose(areas_output, exp_output)

    def test_height_width(self):
        corners = tf.constant([[0.0, 0.0, 10.0, 20.0], [1.0, 2.0, 3.0, 4.0]])
        exp_output_heights = [10., 2.]
        exp_output_widths = [20., 2.]
        boxes = box_list.BoxList(corners)
        heights, widths = box_list_ops.height_width(boxes)
        with self.test_session() as sess:
            output_heights, output_widths = sess.run([heights, widths])
            self.assertAllClose(output_heights, exp_output_heights)
            self.assertAllClose(output_widths, exp_output_widths)

    def test_scale(self):
        corners = tf.constant([[0, 0, 100, 200], [50, 120, 100, 140]],
                              dtype=tf.float32)
        boxes = box_list.BoxList(corners)
        boxes.add_field('extra_data', tf.constant([[1], [2]]))

        y_scale = tf.constant(1.0 / 100)
        x_scale = tf.constant(1.0 / 200)
        scaled_boxes = box_list_ops.scale(boxes, y_scale, x_scale)
        exp_output = [[0, 0, 1, 1], [0.5, 0.6, 1.0, 0.7]]
        with self.test_session() as sess:
            scaled_corners_out = sess.run(scaled_boxes.get())
            self.assertAllClose(scaled_corners_out, exp_output)
            extra_data_out = sess.run(scaled_boxes.get_field('extra_data'))
            self.assertAllEqual(extra_data_out, [[1], [2]])

    def test_intersection(self):
        corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                                [0.0, 0.0, 20.0, 20.0]])
        exp_output = [[2.0, 0.0, 6.0], [1.0, 0.0, 5.0]]
        boxes1 = box_list.BoxList(corners1)
        boxes2 = box_list.BoxList(corners2)
        intersect = box_list_ops.intersection(boxes1, boxes2)
        with self.test_session() as sess:
            intersect_output = sess.run(intersect)
            self.assertAllClose(intersect_output, exp_output)

    def test_matched_intersection(self):
        corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        corners2 = tf.constant(
            [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
        exp_output = [2.0, 0.0]
        boxes1 = box_list.BoxList(corners1)
        boxes2 = box_list.BoxList(corners2)
        intersect = box_list_ops.matched_intersection(boxes1, boxes2)
        with self.test_session() as sess:
            intersect_output = sess.run(intersect)
            self.assertAllClose(intersect_output, exp_output)

    def test_iou(self):
        corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                                [0.0, 0.0, 20.0, 20.0]])
        exp_output = [
            [2.0 / 16.0, 0, 6.0 / 400.0],
            [1.0 / 16.0, 0.0, 5.0 / 400.0]]
        boxes1 = box_list.BoxList(corners1)
        boxes2 = box_list.BoxList(corners2)
        iou = box_list_ops.iou(boxes1, boxes2)
        with self.test_session() as sess:
            iou_output = sess.run(iou)
            self.assertAllClose(iou_output, exp_output)

    def test_matched_iou(self):
        corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        corners2 = tf.constant(
            [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
        exp_output = [2.0 / 16.0, 0]
        boxes1 = box_list.BoxList(corners1)
        boxes2 = box_list.BoxList(corners2)
        iou = box_list_ops.matched_iou(boxes1, boxes2)
        with self.test_session() as sess:
            iou_output = sess.run(iou)
            self.assertAllClose(iou_output, exp_output)

    def test_iouworks_on_empty_inputs(self):
        corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                                [0.0, 0.0, 20.0, 20.0]])
        boxes1 = box_list.BoxList(corners1)
        boxes2 = box_list.BoxList(corners2)
        boxes_empty = box_list.BoxList(tf.zeros((0, 4)))
        iou_empty_1 = box_list_ops.iou(boxes1, boxes_empty)
        iou_empty_2 = box_list_ops.iou(boxes_empty, boxes2)
        iou_empty_3 = box_list_ops.iou(boxes_empty, boxes_empty)
        with self.test_session() as sess:
            iou_output_1, iou_output_2, iou_output_3 = sess.run(
                [iou_empty_1, iou_empty_2, iou_empty_3])
            self.assertAllEqual(iou_output_1.shape, (2, 0))
            self.assertAllEqual(iou_output_2.shape, (0, 3))
            self.assertAllEqual(iou_output_3.shape, (0, 0))

    def test_ioa(self):
        corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                                [0.0, 0.0, 20.0, 20.0]])
        exp_output_1 = [[2.0 / 12.0, 0, 6.0 / 400.0],
                        [1.0 / 12.0, 0.0, 5.0 / 400.0]]
        exp_output_2 = [[2.0 / 6.0, 1.0 / 5.0],
                        [0, 0],
                        [6.0 / 6.0, 5.0 / 5.0]]
        boxes1 = box_list.BoxList(corners1)
        boxes2 = box_list.BoxList(corners2)
        ioa_1 = box_list_ops.ioa(boxes1, boxes2)
        ioa_2 = box_list_ops.ioa(boxes2, boxes1)
        with self.test_session() as sess:
            ioa_output_1, ioa_output_2 = sess.run([ioa_1, ioa_2])
            self.assertAllClose(ioa_output_1, exp_output_1)
            self.assertAllClose(ioa_output_2, exp_output_2)

    def test_prune_non_overlapping_boxes(self):
        corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                                [0.0, 0.0, 20.0, 20.0]])
        boxes1 = box_list.BoxList(corners1)
        boxes2 = box_list.BoxList(corners2)
        minoverlap = 0.5

        exp_output_1 = boxes1
        exp_output_2 = box_list.BoxList(tf.constant(0.0, shape=[0, 4]))
        output_1, keep_indices_1 = box_list_ops.prune_non_overlapping_boxes(
            boxes1, boxes2, min_overlap=minoverlap)
        output_2, keep_indices_2 = box_list_ops.prune_non_overlapping_boxes(
            boxes2, boxes1, min_overlap=minoverlap)
        with self.test_session() as sess:
            (output_1_,
             keep_indices_1_,
             output_2_,
             keep_indices_2_,
             exp_output_1_,
             exp_output_2_) = sess.run([output_1.get(),
                                        keep_indices_1,
                                        output_2.get(),
                                        keep_indices_2,
                                        exp_output_1.get(),
                                        exp_output_2.get()])
            self.assertAllClose(output_1_, exp_output_1_)
            self.assertAllClose(output_2_, exp_output_2_)
            self.assertAllEqual(keep_indices_1_, [0, 1])
            self.assertAllEqual(keep_indices_2_, [])

    def test_prune_small_boxes(self):
        boxes = tf.constant([[4.0, 3.0, 7.0, 5.0],
                             [5.0, 6.0, 10.0, 7.0],
                             [3.0, 4.0, 6.0, 8.0],
                             [14.0, 14.0, 15.0, 15.0],
                             [0.0, 0.0, 20.0, 20.0]])
        exp_boxes = [[3.0, 4.0, 6.0, 8.0],
                     [0.0, 0.0, 20.0, 20.0]]
        boxes = box_list.BoxList(boxes)
        pruned_boxes = box_list_ops.prune_small_boxes(boxes, 3)
        with self.test_session() as sess:
            pruned_boxes = sess.run(pruned_boxes.get())
            self.assertAllEqual(pruned_boxes, exp_boxes)

    def test_prune_small_boxes_prunes_boxes_with_negative_side(self):
        boxes = tf.constant([[4.0, 3.0, 7.0, 5.0],
                             [5.0, 6.0, 10.0, 7.0],
                             [3.0, 4.0, 6.0, 8.0],
                             [14.0, 14.0, 15.0, 15.0],
                             [0.0, 0.0, 20.0, 20.0],
                             [2.0, 3.0, 1.5, 7.0],  # negative height
                             [2.0, 3.0, 5.0, 1.7]])  # negative width
        exp_boxes = [[3.0, 4.0, 6.0, 8.0],
                     [0.0, 0.0, 20.0, 20.0]]
        boxes = box_list.BoxList(boxes)
        pruned_boxes = box_list_ops.prune_small_boxes(boxes, 3)
        with self.test_session() as sess:
            pruned_boxes = sess.run(pruned_boxes.get())
            self.assertAllEqual(pruned_boxes, exp_boxes)

    def test_change_coordinate_frame(self):
        corners = tf.constant([[0.25, 0.5, 0.75, 0.75], [0.5, 0.0, 1.0, 1.0]])
        window = tf.constant([0.25, 0.25, 0.75, 0.75])
        boxes = box_list.BoxList(corners)

        expected_corners = tf.constant(
            [[0, 0.5, 1.0, 1.0], [0.5, -0.5, 1.5, 1.5]])
        expected_boxes = box_list.BoxList(expected_corners)
        output = box_list_ops.change_coordinate_frame(boxes, window)

        with self.test_session() as sess:
            output_, expected_boxes_ = sess.run(
                [output.get(), expected_boxes.get()])
            self.assertAllClose(output_, expected_boxes_)

    def test_ioaworks_on_empty_inputs(self):
        corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
        corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                                [0.0, 0.0, 20.0, 20.0]])
        boxes1 = box_list.BoxList(corners1)
        boxes2 = box_list.BoxList(corners2)
        boxes_empty = box_list.BoxList(tf.zeros((0, 4)))
        ioa_empty_1 = box_list_ops.ioa(boxes1, boxes_empty)
        ioa_empty_2 = box_list_ops.ioa(boxes_empty, boxes2)
        ioa_empty_3 = box_list_ops.ioa(boxes_empty, boxes_empty)
        with self.test_session() as sess:
            ioa_output_1, ioa_output_2, ioa_output_3 = sess.run(
                [ioa_empty_1, ioa_empty_2, ioa_empty_3])
            self.assertAllEqual(ioa_output_1.shape, (2, 0))
            self.assertAllEqual(ioa_output_2.shape, (0, 3))
            self.assertAllEqual(ioa_output_3.shape, (0, 0))

    def test_pairwise_distances(self):
        corners1 = tf.constant([[0.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 0.0, 2.0]])
        corners2 = tf.constant([[3.0, 4.0, 1.0, 0.0],
                                [-4.0, 0.0, 0.0, 3.0],
                                [0.0, 0.0, 0.0, 0.0]])
        exp_output = [[26, 25, 0], [18, 27, 6]]
        boxes1 = box_list.BoxList(corners1)
        boxes2 = box_list.BoxList(corners2)
        dist_matrix = box_list_ops.sq_dist(boxes1, boxes2)
        with self.test_session() as sess:
            dist_output = sess.run(dist_matrix)
            self.assertAllClose(dist_output, exp_output)

    def test_boolean_mask(self):
        corners = tf.constant(
            [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
        indicator = tf.constant([True, False, True, False, True], tf.bool)
        expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
        boxes = box_list.BoxList(corners)
        subset = box_list_ops.boolean_mask(boxes, indicator)
        with self.test_session() as sess:
            subset_output = sess.run(subset.get())
            self.assertAllClose(subset_output, expected_subset)

    def test_boolean_mask_with_field(self):
        corners = tf.constant(
            [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
        indicator = tf.constant([True, False, True, False, True], tf.bool)
        weights = tf.constant([[.1], [.3], [.5], [.7], [.9]], tf.float32)
        expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
        expected_weights = [[.1], [.5], [.9]]

        boxes = box_list.BoxList(corners)
        boxes.add_field('weights', weights)
        subset = box_list_ops.boolean_mask(boxes, indicator, ['weights'])
        with self.test_session() as sess:
            subset_output, weights_output = sess.run(
                [subset.get(), subset.get_field('weights')])
            self.assertAllClose(subset_output, expected_subset)
            self.assertAllClose(weights_output, expected_weights)

    def test_gather(self):
        corners = tf.constant(
            [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
        indices = tf.constant([0, 2, 4], tf.int32)
        expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
        boxes = box_list.BoxList(corners)
        subset = box_list_ops.gather(boxes, indices)
        with self.test_session() as sess:
            subset_output = sess.run(subset.get())
            self.assertAllClose(subset_output, expected_subset)

    def test_gather_with_field(self):
        corners = tf.constant(
            [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
        indices = tf.constant([0, 2, 4], tf.int32)
        weights = tf.constant([[.1], [.3], [.5], [.7], [.9]], tf.float32)
        expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
        expected_weights = [[.1], [.5], [.9]]

        boxes = box_list.BoxList(corners)
        boxes.add_field('weights', weights)
        subset = box_list_ops.gather(boxes, indices, ['weights'])
        with self.test_session() as sess:
            subset_output, weights_output = sess.run(
                [subset.get(), subset.get_field('weights')])
            self.assertAllClose(subset_output, expected_subset)
            self.assertAllClose(weights_output, expected_weights)

    def test_gather_with_invalid_field(self):
        corners = tf.constant([4 * [0.0], 4 * [1.0]])
        indices = tf.constant([0, 1], tf.int32)
        weights = tf.constant([[.1], [.3]], tf.float32)

        boxes = box_list.BoxList(corners)
        boxes.add_field('weights', weights)
        with self.assertRaises(ValueError):
            box_list_ops.gather(boxes, indices, ['foo', 'bar'])

    def test_gather_with_invalid_inputs(self):
        corners = tf.constant(
            [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
        indices_float32 = tf.constant([0, 2, 4], tf.float32)
        boxes = box_list.BoxList(corners)
        with self.assertRaises(ValueError):
            _ = box_list_ops.gather(boxes, indices_float32)
        indices_2d = tf.constant([[0, 2, 4]], tf.int32)
        boxes = box_list.BoxList(corners)
        with self.assertRaises(ValueError):
            _ = box_list_ops.gather(boxes, indices_2d)

    def test_gather_with_dynamic_indexing(self):
        corners = tf.constant(
            [4 * [0.0],
             4 * [1.0],
             4 * [2.0],
             4 * [3.0],
             4 * [4.0]])
        weights = tf.constant([.5, .3, .7, .1, .9], tf.float32)
        indices = tf.reshape(tf.where(tf.greater(weights, 0.4)), [-1])
        expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
        expected_weights = [.5, .7, .9]

        boxes = box_list.BoxList(corners)
        boxes.add_field('weights', weights)
        subset = box_list_ops.gather(boxes, indices, ['weights'])
        with self.test_session() as sess:
            subset_output, weights_output = sess.run(
                [subset.get(), subset.get_field('weights')])
            self.assertAllClose(subset_output, expected_subset)
            self.assertAllClose(weights_output, expected_weights)


if __name__ == '__main__':
    tf.test.main()

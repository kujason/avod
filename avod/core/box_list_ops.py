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

"""Bounding Box List operations.

Example box operations that are supported:
  * areas: compute bounding box areas
  * iou: pairwise intersection-over-union scores
  * sq_dist: pairwise distances between bounding boxes

Whenever box_list_ops functions output a BoxList, the fields of the incoming
BoxList are retained unless documented otherwise.
"""
import tensorflow as tf

from avod.core import box_list


class SortOrder(object):
    """Enum class for sort order.

    Attributes:
      ascend: ascend order.
      descend: descend order.
    """
    ascend = 1
    descend = 2


def area(boxlist, scope=None):
    """Computes area of boxes.

    Args:
      boxlist: BoxList holding N boxes
      scope: name scope.

    Returns:
      a tensor with shape [N] representing box areas.
    """
    with tf.name_scope(scope, 'Area'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.get(), num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def height_width(boxlist, scope=None):
    """Computes height and width of boxes in boxlist.

    Args:
      boxlist: BoxList holding N boxes
      scope: name scope.

    Returns:
      Height: A tensor with shape [N] representing box heights.
      Width: A tensor with shape [N] representing box widths.
    """
    with tf.name_scope(scope, 'HeightWidth'):
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.get(), num_or_size_splits=4, axis=1)
        return tf.squeeze(y_max - y_min, [1]), tf.squeeze(x_max - x_min, [1])


def scale(boxlist, y_scale, x_scale, scope=None):
    """scale box coordinates in x and y dimensions.

    Args:
      boxlist: BoxList holding N boxes
      y_scale: (float) scalar tensor
      x_scale: (float) scalar tensor
      scope: name scope.

    Returns:
      boxlist: BoxList holding N boxes
    """
    with tf.name_scope(scope, 'Scale'):
        y_scale = tf.cast(y_scale, tf.float32)
        x_scale = tf.cast(x_scale, tf.float32)
        y_min, x_min, y_max, x_max = tf.split(
            value=boxlist.get(), num_or_size_splits=4, axis=1)
        y_min = y_scale * y_min
        y_max = y_scale * y_max
        x_min = x_scale * x_min
        x_max = x_scale * x_max
        scaled_boxlist = box_list.BoxList(
            tf.concat([y_min, x_min, y_max, x_max], 1))
        return _copy_extra_fields(scaled_boxlist, boxlist)


def intersection(boxlist1, boxlist2, scope=None):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'Intersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            value=boxlist1.get(), num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            value=boxlist2.get(), num_or_size_splits=4, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(
            0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(
            0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths


def matched_intersection(boxlist1, boxlist2, scope=None):
    """Compute intersection areas between corresponding boxes in two boxlists.

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding N boxes
      scope: name scope.

    Returns:
      a tensor with shape [N] representing pairwise intersections
    """
    with tf.name_scope(scope, 'MatchedIntersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            value=boxlist1.get(), num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            value=boxlist2.get(), num_or_size_splits=4, axis=1)
        min_ymax = tf.minimum(y_max1, y_max2)
        max_ymin = tf.maximum(y_min1, y_min2)
        intersect_heights = tf.maximum(0.0, min_ymax - max_ymin)
        min_xmax = tf.minimum(x_max1, x_max2)
        max_xmin = tf.maximum(x_min1, x_min2)
        intersect_widths = tf.maximum(0.0, min_xmax - max_xmin)
        return tf.reshape(intersect_heights * intersect_widths, [-1])


def iou(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'IOU'):
        intersections = intersection(boxlist1, boxlist2)
        areas1 = area(boxlist1)
        areas2 = area(boxlist2)
        unions = (
            tf.expand_dims(
                areas1,
                1) +
            tf.expand_dims(
                areas2,
                0) -
            intersections)
        return tf.where(
            tf.equal(intersections, 0.0),
            tf.zeros_like(intersections), tf.truediv(intersections, unions))


def matched_iou(boxlist1, boxlist2, scope=None):
    """Compute intersection-over-union between corresponding boxes in boxlists.

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding N boxes
      scope: name scope.

    Returns:
      a tensor with shape [N] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'MatchedIOU'):
        intersections = matched_intersection(boxlist1, boxlist2)
        areas1 = area(boxlist1)
        areas2 = area(boxlist2)
        unions = areas1 + areas2 - intersections
        return tf.where(
            tf.equal(intersections, 0.0),
            tf.zeros_like(intersections), tf.truediv(intersections, unions))


def ioa(boxlist1, boxlist2, scope=None):
    """Computes pairwise intersection-over-area between box collections.

    intersection-over-area (IOA) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, ioa(box1, box2) != ioa(box2, box1).

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise ioa scores.
    """
    with tf.name_scope(scope, 'IOA'):
        intersections = intersection(boxlist1, boxlist2)
        areas = tf.expand_dims(area(boxlist2), 0)
        return tf.truediv(intersections, areas)


def prune_non_overlapping_boxes(
        boxlist1, boxlist2, min_overlap=0.0, scope=None):
    """Prunes the boxes in boxlist1 that overlap less than thresh with boxlist2.

    For each box in boxlist1, we want its IOA to be more than minoverlap with
    at least one of the boxes in boxlist2. If it does not, we remove it.

    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.
      min_overlap: Minimum required overlap between boxes, to count them as
                  overlapping.
      scope: name scope.

    Returns:
      new_boxlist1: A pruned boxlist with size [N', 4].
      keep_inds: A tensor with shape [N'] indexing kept bounding boxes in the
        first input BoxList `boxlist1`.
    """
    with tf.name_scope(scope, 'PruneNonOverlappingBoxes'):
        ioa_ = ioa(boxlist2, boxlist1)  # [M, N] tensor
        ioa_ = tf.reduce_max(ioa_, reduction_indices=[0])  # [N] tensor
        keep_bool = tf.greater_equal(ioa_, tf.constant(min_overlap))
        keep_inds = tf.squeeze(tf.where(keep_bool), squeeze_dims=[1])
        new_boxlist1 = gather(boxlist1, keep_inds)
        return new_boxlist1, keep_inds


def prune_small_boxes(boxlist, min_side, scope=None):
    """Prunes small boxes in the boxlist which have a side smaller than min_side.

    Args:
      boxlist: BoxList holding N boxes.
      min_side: Minimum width AND height of box to survive pruning.
      scope: name scope.

    Returns:
      A pruned boxlist.
    """
    with tf.name_scope(scope, 'PruneSmallBoxes'):
        height, width = height_width(boxlist)
        is_valid = tf.logical_and(tf.greater_equal(width, min_side),
                                  tf.greater_equal(height, min_side))
        return gather(boxlist, tf.reshape(tf.where(is_valid), [-1]))


def change_coordinate_frame(boxlist, window, scope=None):
    """Change coordinate frame of the boxlist to be relative to window's frame.

    Given a window of the form [ymin, xmin, ymax, xmax],
    changes bounding box coordinates from boxlist to be relative to this window
    (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

    An example use case is data augmentation: where we are given groundtruth
    boxes (boxlist) and would like to randomly crop the image to some
    window (window). In this case we need to change the coordinate frame of
    each groundtruth box to be relative to this new window.

    Args:
      boxlist: A BoxList object holding N boxes.
      window: A rank 1 tensor [4].
      scope: name scope.

    Returns:
      Returns a BoxList object with N boxes.
    """
    with tf.name_scope(scope, 'ChangeCoordinateFrame'):
        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        boxlist_new = scale(box_list.BoxList(
            boxlist.get() - [window[0], window[1], window[0], window[1]]),
            1.0 / win_height, 1.0 / win_width)
        boxlist_new = _copy_extra_fields(boxlist_new, boxlist)
        return boxlist_new


def sq_dist(boxlist1, boxlist2, scope=None):
    """Computes the pairwise squared distances between box corners.

    This op treats each box as if it were a point in a 4d Euclidean space and
    computes pairwise squared distances.

    Mathematically, we are given two matrices of box coordinates X and Y,
    where X(i,:) is the i'th row of X, containing the 4 numbers defining the
    corners of the i'th box in boxlist1. Similarly Y(j,:) corresponds to
    boxlist2.  We compute
    Z(i,j) = ||X(i,:) - Y(j,:)||^2
           = ||X(i,:)||^2 + ||Y(j,:)||^2 - 2 X(i,:)' * Y(j,:),

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise distances
    """
    with tf.name_scope(scope, 'SqDist'):
        sqnorm1 = tf.reduce_sum(tf.square(boxlist1.get()), 1, keep_dims=True)
        sqnorm2 = tf.reduce_sum(tf.square(boxlist2.get()), 1, keep_dims=True)
        innerprod = tf.matmul(boxlist1.get(), boxlist2.get(),
                              transpose_a=False, transpose_b=True)
        return sqnorm1 + tf.transpose(sqnorm2) - 2.0 * innerprod


def boolean_mask(boxlist, indicator, fields=None, scope=None):
    """Select boxes from BoxList according to indicator and return new BoxList.

    `boolean_mask` returns the subset of boxes that are marked as "True" by the
    indicator tensor. By default, `boolean_mask` returns boxes corresponding to
    the input index list, as well as all additional fields stored in the boxlist
    (indexing into the first dimension).  However one can optionally only draw
    from a subset of fields.

    Args:
      boxlist: BoxList holding N boxes
      indicator: a rank-1 boolean tensor
      fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.
      scope: name scope.

    Returns:
      subboxlist: a BoxList corresponding to the subset of the input BoxList
        specified by indicator
    Raises:
      ValueError: if `indicator` is not a rank-1 boolean tensor.
    """
    with tf.name_scope(scope, 'BooleanMask'):
        if indicator.shape.ndims != 1:
            raise ValueError('indicator should have rank 1')
        if indicator.dtype != tf.bool:
            raise ValueError('indicator should be a boolean tensor')
        subboxlist = box_list.BoxList(
            tf.boolean_mask(boxlist.get(), indicator))
        if fields is None:
            fields = boxlist.get_extra_fields()
        for field in fields:
            if not boxlist.has_field(field):
                raise ValueError('boxlist must contain all specified fields')
            subfieldlist = tf.boolean_mask(boxlist.get_field(field), indicator)
            subboxlist.add_field(field, subfieldlist)
        return subboxlist


def gather(boxlist, indices, fields=None, scope=None):
    """Gather boxes from BoxList according to indices and return new BoxList.

    By default, `gather` returns boxes corresponding to the input index list, as
    well as all additional fields stored in the boxlist (indexing into the
    first dimension).  However one can optionally only gather from a
    subset of fields.

    Args:
      boxlist: BoxList holding N boxes
      indices: a rank-1 tensor of type int32 / int64
      fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.
      scope: name scope.

    Returns:
      subboxlist: a BoxList corresponding to the subset of the input BoxList
      specified by indices
    Raises:
      ValueError: if specified field is not contained in boxlist or if the
        indices are not of type int32
    """
    with tf.name_scope(scope, 'Gather'):
        if len(indices.shape.as_list()) != 1:
            raise ValueError('indices should have rank 1')
        if indices.dtype != tf.int32 and indices.dtype != tf.int64:
            raise ValueError('indices should be an int32 / int64 tensor')
        subboxlist = box_list.BoxList(tf.gather(boxlist.get(), indices))
        if fields is None:
            fields = boxlist.get_extra_fields()
        for field in fields:
            if not boxlist.has_field(field):
                raise ValueError('boxlist must contain all specified fields')
            subfieldlist = tf.gather(boxlist.get_field(field), indices)
            subboxlist.add_field(field, subfieldlist)
        return subboxlist


def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):
    """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.
    Args:
      boxlist_to_copy_to: BoxList to which extra fields are copied.
      boxlist_to_copy_from: BoxList from which fields are copied.
    Returns:
      boxlist_to_copy_to with extra fields.
    """
    for field in boxlist_to_copy_from.get_extra_fields():
        boxlist_to_copy_to.add_field(
            field, boxlist_to_copy_from.get_field(field))
    return boxlist_to_copy_to

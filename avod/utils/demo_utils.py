import copy
import numpy as np
import tensorflow as tf

from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.obj_detection import evaluation

from avod.core import anchor_projector
from avod.core import box_3d_encoder


COLOUR_SCHEME_PREDICTIONS = {
    "Easy GT": (255, 255, 0),     # Yellow
    "Medium GT": (255, 128, 0),   # Orange
    "Hard GT": (255, 0, 0),       # Red

    "Prediction": (50, 255, 50),  # Green
}


def get_gts_based_on_difficulty(dataset, img_idx):
    """Returns lists of ground-truth based on difficulty.
    """
    # Get all ground truth labels
    all_gt_objs = obj_utils.read_labels(dataset.label_dir, img_idx)

    # Filter to dataset classes
    gt_objs = dataset.kitti_utils.filter_labels(all_gt_objs)

    # Filter objects to desired difficulty
    easy_gt_objs = dataset.kitti_utils.filter_labels(
        copy.deepcopy(gt_objs), difficulty=0)
    medium_gt_objs = dataset.kitti_utils.filter_labels(
        copy.deepcopy(gt_objs), difficulty=1)
    hard_gt_objs = dataset.kitti_utils.filter_labels(
        copy.deepcopy(gt_objs), difficulty=2)

    for gt_obj in easy_gt_objs:
        gt_obj.type = 'Easy GT'
    for gt_obj in medium_gt_objs:
        gt_obj.type = 'Medium GT'
    for gt_obj in hard_gt_objs:
        gt_obj.type = 'Hard GT'

    return easy_gt_objs, medium_gt_objs, hard_gt_objs, all_gt_objs


def get_max_ious_3d(all_gt_boxes_3d, pred_boxes_3d):
    """Helper function to calculate 3D IoU for the given predictions.

    Args:
        all_gt_boxes_3d: A list of the same ground-truth boxes in box_3d
            format.
        pred_boxes_3d: A list of predictions in box_3d format.
    """

    # Only calculate ious if there are predictions
    if pred_boxes_3d:
        # Convert to iou format
        gt_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(
            all_gt_boxes_3d)
        pred_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(
            pred_boxes_3d)

        max_ious_3d = np.zeros(len(all_gt_boxes_3d))
        for gt_obj_idx in range(len(all_gt_boxes_3d)):

            gt_obj_iou_fmt = gt_objs_iou_fmt[gt_obj_idx]

            ious_3d = evaluation.three_d_iou(gt_obj_iou_fmt,
                                             pred_objs_iou_fmt)

            max_ious_3d[gt_obj_idx] = np.amax(ious_3d)
    else:
        # No detections, all ious = 0
        max_ious_3d = np.zeros(len(all_gt_boxes_3d))

    return max_ious_3d


def tf_project_to_image_space(anchors, calib_p2, image_shape, img_idx):
    """Helper function to convert data to tensors and project
       to image space using the tf projection function.
    """

    anchors_tensor = tf.convert_to_tensor(anchors, tf.float32)
    calib_p2_tensor = tf.convert_to_tensor(calib_p2, tf.float32)
    image_shape_tensor = tf.convert_to_tensor(image_shape, tf.float32)

    projected_boxes_tensor, _ = \
        anchor_projector.tf_project_to_image_space(
            anchors_tensor,
            calib_p2_tensor,
            image_shape_tensor)
    sess = tf.Session()

    with sess.as_default():
        projected_boxes = projected_boxes_tensor.eval()

    return projected_boxes

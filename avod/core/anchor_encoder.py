import numpy as np
import tensorflow as tf

import avod.core.format_checker as fc


def anchor_to_offset(anchors, ground_truth):
    """Encodes the anchor regression predictions with the
    ground truth.

    Args:
        anchors: A numpy array of shape (N, 6) representing
            the generated anchors.
        ground_truth: A numpy array of shape (6,) containing
            the label boxes in the anchor format.

    Returns:
        anchor_offsets: A numpy array of shape (N, 6)
            encoded/normalized with the ground-truth, representing the
            offsets.
    """

    fc.check_anchor_format(anchors)

    anchors = np.asarray(anchors).reshape(-1, 6)
    ground_truth = np.reshape(ground_truth, (6,))

    # t_x_gt = (x_gt - x_anch)/dim_x_anch
    t_x_gt = (ground_truth[0] - anchors[:, 0]) / anchors[:, 3]
    # t_y_gt = (y_gt - y_anch)/dim_y_anch
    t_y_gt = (ground_truth[1] - anchors[:, 1]) / anchors[:, 4]
    # t_z_gt = (z_gt - z_anch)/dim_z_anch
    t_z_gt = (ground_truth[2] - anchors[:, 2]) / anchors[:, 5]
    # t_dx_gt = log(dim_x_gt/dim_x_anch)
    t_dx_gt = np.log(ground_truth[3] / anchors[:, 3])
    # t_dy_gt = log(dim_y_gt/dim_y_anch)
    t_dy_gt = np.log(ground_truth[4] / anchors[:, 4])
    # t_dz_gt = log(dim_z_gt/dim_z_anch)
    t_dz_gt = np.log(ground_truth[5] / anchors[:, 5])
    anchor_offsets = np.stack((t_x_gt,
                               t_y_gt,
                               t_z_gt,
                               t_dx_gt,
                               t_dy_gt,
                               t_dz_gt), axis=1)
    return anchor_offsets


def tf_anchor_to_offset(anchors, ground_truth):
    """Encodes the anchor regression predictions with the
    ground truth.

    This function assumes the ground_truth tensor has been arranged
    in a way that each corresponding row in ground_truth, is matched
    with that anchor according to the highest IoU.
    For instance, the ground_truth might be a matrix of shape (256, 6)
    of repeated entries for the original ground truth of shape (x, 6),
    where each entry has been selected as the highest IoU match with that
    anchor. This is different from the same function in numpy format, where
    we loop through all the ground truth anchors, and calculate IoUs for
    each and then select the match with the highest IoU.

    Args:
        anchors: A tensor of shape (N, 6) representing
            the generated anchors.
        ground_truth: A tensor of shape (N, 6) containing
            the label boxes in the anchor format. Each ground-truth entry
            has been matched with the anchor in the same entry as having
            the highest IoU.

    Returns:
        anchor_offsets: A tensor of shape (N, 6)
            encoded/normalized with the ground-truth, representing the
            offsets.
    """

    fc.check_anchor_format(anchors)

    # Make sure anchors and anchor_gts have the same shape
    dim_cond = tf.equal(tf.shape(anchors), tf.shape(ground_truth))

    with tf.control_dependencies([dim_cond]):
        t_x_gt = (ground_truth[:, 0] - anchors[:, 0]) / anchors[:, 3]
        t_y_gt = (ground_truth[:, 1] - anchors[:, 1]) / anchors[:, 4]
        t_z_gt = (ground_truth[:, 2] - anchors[:, 2]) / anchors[:, 5]
        t_dx_gt = tf.log(ground_truth[:, 3] / anchors[:, 3])
        t_dy_gt = tf.log(ground_truth[:, 4] / anchors[:, 4])
        t_dz_gt = tf.log(ground_truth[:, 5] / anchors[:, 5])
        anchor_offsets = tf.stack((t_x_gt,
                                   t_y_gt,
                                   t_z_gt,
                                   t_dx_gt,
                                   t_dy_gt,
                                   t_dz_gt), axis=1)

        return anchor_offsets


def offset_to_anchor(anchors, offsets):
    """Decodes the anchor regression predictions with the
    anchor.

    Args:
        anchors: A numpy array or a tensor of shape [N, 6]
            representing the generated anchors.
        offsets: A numpy array or a tensor of shape
            [N, 6] containing the predicted offsets in the
            anchor format  [x, y, z, dim_x, dim_y, dim_z].

    Returns:
        anchors: A numpy array of shape [N, 6]
            representing the predicted anchor boxes.
    """

    fc.check_anchor_format(anchors)
    fc.check_anchor_format(offsets)

    # x = dx * dim_x + x_anch
    x_pred = (offsets[:, 0] * anchors[:, 3]) + anchors[:, 0]
    # y = dy * dim_y + x_anch
    y_pred = (offsets[:, 1] * anchors[:, 4]) + anchors[:, 1]
    # z = dz * dim_z + z_anch
    z_pred = (offsets[:, 2] * anchors[:, 5]) + anchors[:, 2]

    tensor_format = isinstance(anchors, tf.Tensor)
    if tensor_format:
        # dim_x = exp(log(dim_x) + dx)
        dx_pred = tf.exp(tf.log(anchors[:, 3]) + offsets[:, 3])
        # dim_y = exp(log(dim_y) + dy)
        dy_pred = tf.exp(tf.log(anchors[:, 4]) + offsets[:, 4])
        # dim_z = exp(log(dim_z) + dz)
        dz_pred = tf.exp(tf.log(anchors[:, 5]) + offsets[:, 5])
        anchors = tf.stack((x_pred,
                            y_pred,
                            z_pred,
                            dx_pred,
                            dy_pred,
                            dz_pred), axis=1)
    else:
        dx_pred = np.exp(np.log(anchors[:, 3]) + offsets[:, 3])
        dy_pred = np.exp(np.log(anchors[:, 4]) + offsets[:, 4])
        dz_pred = np.exp(np.log(anchors[:, 5]) + offsets[:, 5])
        anchors = np.stack((x_pred,
                            y_pred,
                            z_pred,
                            dx_pred,
                            dy_pred,
                            dz_pred), axis=1)

    return anchors

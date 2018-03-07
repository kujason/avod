import numpy as np
import tensorflow as tf

from avod.core import format_checker
from avod.core import box_3d_encoder


def np_box_3d_to_box_8co(box_3d):
    """Computes the 3D bounding box corner positions from Box3D format.

    The order of corners are preserved during this conversion.

    Args:
        box_3d: 1 x 7 ndarray of box_3d in the format
            [x, y, z, l, w, h, ry]
    Returns:
        corners_3d: An ndarray or a tensor of shape (3 x 8) representing
            the box as corners in the following format ->
            [[x1,...,x8], [y1...,y8], [z1,...,z8]].
    """

    format_checker.check_box_3d_format(box_3d)

    ry = box_3d[6]
    # Compute transform matrix
    # This includes rotation and translation
    rot = np.array([[np.cos(ry), 0, np.sin(ry), box_3d[0]],
                    [0, 1, 0, box_3d[1]],
                    [-np.sin(ry), 0, np.cos(ry), box_3d[2]]])

    length = box_3d[3]
    width = box_3d[4]
    height = box_3d[5]

    # 3D BB corners
    x_corners = np.array([length / 2, length / 2,
                          -length / 2, -length / 2,
                          length / 2, length / 2,
                          -length / 2, -length / 2])

    y_corners = np.array([0.0, 0.0, 0.0, 0.0,
                          -height, -height, -height, -height])

    z_corners = np.array([width / 2, -width / 2,
                          -width / 2, width / 2,
                          width / 2, -width / 2,
                          -width / 2, width / 2])

    # Create a ones column
    ones_col = np.ones(x_corners.shape)

    # Append the column of ones to be able to multiply
    box_8c = np.dot(rot, np.array([x_corners,
                                   y_corners,
                                   z_corners,
                                   ones_col]))
    # Ignore the fourth column
    box_8c = box_8c[0:3]

    return box_8c


def tf_box_3d_to_box_8co(boxes_3d):
    """Computes the 3D bounding box corner positions from Box3D format.

    The order of corners are preserved during this conversion.

    Args:
        boxes_3d: N x 7 tensor of box_3d in the format
            [x, y, z, l, w, h, ry]
    Returns:
        corners_3d: An ndarray or a tensor of shape (N x 3 x 8) representing
            the box as corners in following format -> [[[x1,...,x8],[y1...,y8],
            [z1,...,z8]]].
    """

    format_checker.check_box_3d_format(boxes_3d)

    all_rys = boxes_3d[:, 6]
    ry_sin = tf.sin(all_rys)
    ry_cos = tf.cos(all_rys)

    zeros = tf.zeros_like(all_rys, dtype=tf.float32)
    ones = tf.ones_like(all_rys, dtype=tf.float32)

    # Rotation matrix
    rot_mats = tf.stack([tf.stack([ry_cos, zeros, ry_sin], axis=1),
                         tf.stack([zeros, ones, zeros], axis=1),
                         tf.stack([-ry_sin, zeros, ry_cos], axis=1)],
                        axis=2)

    length = boxes_3d[:, 3]
    width = boxes_3d[:, 4]
    height = boxes_3d[:, 5]

    half_length = length / 2
    half_width = width / 2

    x_corners = tf.stack([half_length, half_length,
                          -half_length, -half_length,
                          half_length, half_length,
                          -half_length, -half_length], axis=1)

    y_corners = tf.stack([zeros, zeros, zeros, zeros,
                          -height, -height, -height, -height], axis=1)

    z_corners = tf.stack([half_width, -half_width,
                          -half_width, half_width,
                          half_width, -half_width,
                          -half_width, half_width], axis=1)

    corners = tf.stack([x_corners,
                        y_corners,
                        z_corners], axis=1)

    boxes_8c = tf.matmul(rot_mats, corners,
                         transpose_a=True,
                         transpose_b=False)

    # Translate the corners
    corners_3d_x = boxes_8c[:, 0] + tf.reshape(boxes_3d[:, 0], (-1, 1))
    corners_3d_y = boxes_8c[:, 1] + tf.reshape(boxes_3d[:, 1], (-1, 1))
    corners_3d_z = boxes_8c[:, 2] + tf.reshape(boxes_3d[:, 2], (-1, 1))

    boxes_8c = tf.stack([corners_3d_x,
                         corners_3d_y,
                         corners_3d_z], axis=1)

    return boxes_8c


def np_box_3d_to_box_8c(box_3d):
    """Computes the 3D bounding box corner positions from box_3d format.

    This function does not preserve corners order but rather the corners
    are rotated to the nearest 90 degree angle. This helps in calculating
    the closest corner to corner when comparing the corners to the ground-
    truth boxes.

    Args:
        box_3d: ndarray of size (7,) representing box_3d in the format
            [x, y, z, l, w, h, ry]
    Returns:
        corners_3d: An ndarray or a tensor of shape (3 x 8) representing
            the box as corners in following format -> [[x1,...,x8],[y1...,y8],
            [z1,...,z8]].
    """

    format_checker.check_box_3d_format(box_3d)

    # This function is vectorized and returns an ndarray
    anchor = box_3d_encoder.box_3d_to_anchor(box_3d, ortho_rotate=True)[0]

    centroid_x = anchor[0]
    centroid_y = anchor[1]
    centroid_z = anchor[2]
    dim_x = anchor[3]
    dim_y = anchor[4]
    dim_z = anchor[5]

    half_dim_x = dim_x / 2
    half_dim_z = dim_z / 2

    # 3D BB corners
    x_corners = np.array([half_dim_x, half_dim_x,
                          -half_dim_x, -half_dim_x,
                          half_dim_x, half_dim_x,
                          -half_dim_x, -half_dim_x])

    y_corners = np.array([0.0, 0.0, 0.0, 0.0,
                          -dim_y, -dim_y, -dim_y, -dim_y])

    z_corners = np.array([half_dim_z, -half_dim_z,
                          -half_dim_z, half_dim_z,
                          half_dim_z, -half_dim_z,
                          -half_dim_z, half_dim_z])

    ry = box_3d[6]

    # Find nearest 90 degree
    half_pi = np.pi / 2
    ortho_ry = np.round(ry / half_pi) * half_pi

    # Find rotation to make the box ortho aligned
    ry_diff = ry - ortho_ry

    # Compute transform matrix
    # This includes rotation and translation
    rot = np.array([[np.cos(ry_diff), 0, np.sin(ry_diff), centroid_x],
                    [0, 1, 0, centroid_y],
                    [-np.sin(ry_diff), 0, np.cos(ry_diff), centroid_z]])

    # Create a ones column
    ones_col = np.ones(x_corners.shape)

    # Append the column of ones to be able to multiply
    box_8c = np.dot(rot, np.array([x_corners,
                                   y_corners,
                                   z_corners,
                                   ones_col]))
    # Ignore the fourth column
    box_8c = box_8c[0:3]

    return box_8c


def tf_box_3d_to_box_8c(boxes_3d):
    """Computes the 3D bounding box corner positions from box_3d format.

    This function does not preserve corners order during conversion from
    box_3d -> box_8c. Instead of using the box_3d's orientation, 'ry',
    nearest 90 degree angle is selected to create an axis-aligned box.
    This helps in calculating the closest corner to corner when comparing
    the corners to the ground-truth boxes.

    Args:
        boxes_3d: N x 7 tensor of box_3d in the format
            [x, y, z, l, w, h, ry]
    Returns:
        corners_3d: A tensor of shape (N x 3 x 8) representing
            the box as corners in following format -> [[[x1,...,x8],[y1...,y8],
            [z1,...,z8]]].
    """

    format_checker.check_box_3d_format(boxes_3d)
    anchors = box_3d_encoder.tf_box_3d_to_anchor(boxes_3d)

    centroid_x = anchors[:, 0]
    centroid_y = anchors[:, 1]
    centroid_z = anchors[:, 2]
    dim_x = anchors[:, 3]
    dim_y = anchors[:, 4]
    dim_z = anchors[:, 5]

    all_rys = boxes_3d[:, 6]

    # Find nearest 90 degree
    half_pi = np.pi / 2
    ortho_rys = tf.round(all_rys / half_pi) * half_pi

    ry_diff = all_rys - ortho_rys

    ry_sin = tf.sin(ry_diff)
    ry_cos = tf.cos(ry_diff)

    zeros = tf.zeros_like(ry_diff, dtype=tf.float32)
    ones = tf.ones_like(ry_diff, dtype=tf.float32)

    # Rotation matrix
    rot_mats = tf.stack([tf.stack([ry_cos, zeros, ry_sin], axis=1),
                         tf.stack([zeros, ones, zeros], axis=1),
                         tf.stack([-ry_sin, zeros, ry_cos], axis=1)],
                        axis=2)

    half_dim_x = dim_x / 2
    half_dim_z = dim_z / 2

    x_corners = tf.stack([half_dim_x, half_dim_x,
                          -half_dim_x, -half_dim_x,
                          half_dim_x, half_dim_x,
                          -half_dim_x, -half_dim_x], axis=1)

    y_corners = tf.stack([zeros, zeros, zeros, zeros,
                          -dim_y, -dim_y, -dim_y, -dim_y], axis=1)

    z_corners = tf.stack([half_dim_z, -half_dim_z,
                          -half_dim_z, half_dim_z,
                          half_dim_z, -half_dim_z,
                          -half_dim_z, half_dim_z], axis=1)

    corners = tf.stack([x_corners,
                        y_corners,
                        z_corners], axis=1)

    boxes_8c = tf.matmul(rot_mats, corners,
                         transpose_a=True,
                         transpose_b=False)

    # Translate the corners
    corners_3d_x = boxes_8c[:, 0] + tf.reshape(centroid_x, (-1, 1))
    corners_3d_y = boxes_8c[:, 1] + tf.reshape(centroid_y, (-1, 1))
    corners_3d_z = boxes_8c[:, 2] + tf.reshape(centroid_z, (-1, 1))

    boxes_8c = tf.stack([corners_3d_x,
                         corners_3d_y,
                         corners_3d_z], axis=1)

    return boxes_8c


def align_boxes_8c(boxes_8c):
    """Finds the min/max of each corner to align irregular corners.

    In the case where the regressed corners might be skewed, it tries to
    align each face corners of the box to line up, resulting to an aligned
    3D box shape. It finds the min/max of corners for each axis, and re-assigns
    the corners. Note this assumes *certain order* of corners.

    Args:
        boxes_8c: An ndarray or a tensor of shape (N x 3 x 8) representing
            the box corners.
    Returns
        aligned_boxes_8c: An ndarray or a tensor of shape (N x 3 x 8)
            representing the box corners.
    """

    format_checker.check_box_8c_format(boxes_8c)

    x_corners = boxes_8c[:, 0]
    y_corners = boxes_8c[:, 1]
    z_corners = boxes_8c[:, 2]

    min_x = tf.reduce_min(x_corners, axis=1)

    ##########################
    # X-Corners P3, P4, P7, P8
    ##########################
    corner_x3 = min_x
    corner_x4 = min_x
    corner_x7 = min_x
    corner_x8 = min_x

    ##########################
    # X-Corners P1, P2, P5, P6
    ##########################
    max_x = tf.reduce_max(x_corners, axis=1)

    corner_x1 = max_x
    corner_x2 = max_x
    corner_x5 = max_x
    corner_x6 = max_x

    ##########################
    # Z-Corners P2, P3, P6, P7
    ##########################
    min_z = tf.reduce_min(z_corners, axis=1)

    corner_z2 = min_z
    corner_z3 = min_z
    corner_z6 = min_z
    corner_z7 = min_z

    ##########################
    # Z-Corners P1, P4, P5, P6
    ##########################
    max_z = tf.reduce_max(z_corners, axis=1)

    corner_z1 = max_z
    corner_z4 = max_z
    corner_z5 = max_z
    corner_z8 = max_z

    ##########################
    # Y-Corners P1, P2, P3, P4
    ##########################
    # Take the max of the four top y-corners
    # This is because y-axis is facing downwards
    corner_max_y = tf.reduce_max(y_corners, axis=1)
    corner_y1 = corner_y2 = corner_y3 = corner_y4 = corner_max_y

    ##########################
    # Y-Corners P5, P6, P7, P8
    ##########################
    # Take the min of the four bottom y-corners
    corner_min_y = tf.reduce_min(y_corners, axis=1)
    corner_y5 = corner_y6 = corner_y7 = corner_y8 = corner_min_y

    x_corners = tf.stack([corner_x1, corner_x2, corner_x3,
                          corner_x4, corner_x5, corner_x6,
                          corner_x7, corner_x8], axis=1)
    y_corners = tf.stack([corner_y1, corner_y2, corner_y3,
                          corner_y4, corner_y5, corner_y6,
                          corner_y7, corner_y8], axis=1)
    z_corners = tf.stack([corner_z1, corner_z2, corner_z3,
                          corner_z4, corner_z5, corner_z6,
                          corner_z7, corner_z8], axis=1)

    aligned_boxes_8c = tf.stack([x_corners, y_corners, z_corners], axis=1)

    return aligned_boxes_8c


def box_8c_to_box_3d(box_8c):
    """Computes the 3D bounding box corner positions from 8 corners.

    To go back from 8-corner representation to box3D, we need to reverse
    the transformation done in 'box_3d_to_box_8c'. The first thing we need
    is orientation, this is estimated by calculating the midpoints of
    P1 -> P2 and P3 -> P4. Connecting these midpoints, results to a vector
    which gives us the direction of the corners. However note that y-axis
    is facing downwards and hence we negate this orientation.

    Next we calculate the centroids by taking the average of four corners
    for x and z axes. We then translate the centroids back to the origin
    and then multiply by the rotation matrix, however now we are rotating
    the opposite direction, so the angle signs are reversed. After rotation
    we can translate the corners back however, there is one additional step
    before translation. Since we plan to regress corners, it is expected
    for the corners to be skewed, i.e. resulting to non-rectangular shapes.
    Hence we attempt to align the corners (by min/maxing the corners and
    aligning them by the min and max values for each corner. After this step
    we can translate back, and calculate length, width and height.

    Args:
        box_8c: An ndarray or a tensor of shape (N x 3 x 8) representing
            the box corners.
    Returns:
        corners_3d: An ndarray or a tensor of shape (3 x 8) representing
            the box as corners in this format -> [[x1,...,x8],[y1...,y8],
            [z1,...,z8]].
    """
    format_checker.check_box_8c_format(box_8c)

    #######################
    # calculate orientation
    #######################
    x_corners = box_8c[:, 0]
    y_corners = box_8c[:, 1]
    z_corners = box_8c[:, 2]

    x12_midpoint = (x_corners[:, 0] + x_corners[:, 1]) / 2
    z12_midpoint = (z_corners[:, 0] + z_corners[:, 1]) / 2

    x34_midpoint = (x_corners[:, 2] + x_corners[:, 3]) / 2
    z34_midpoint = (z_corners[:, 2] + z_corners[:, 3]) / 2

    # We use the midpoints to get a vector to figure out
    # the orientation
    delta_x = x12_midpoint - x34_midpoint
    delta_z = z12_midpoint - z34_midpoint
    # negate the orientation since y is downwards
    rys = -tf.atan2(delta_z, delta_x)

    # Calcuate the centroid by averaging four corners
    center_x = tf.reduce_mean(x_corners[:, 0:4], axis=1)
    center_z = tf.reduce_mean(z_corners[:, 0:4], axis=1)

    # Translate the centroid to the origin before rotation
    translated_x = box_8c[:, 0] - tf.reshape(center_x, (-1, 1))
    translated_z = box_8c[:, 2] - tf.reshape(center_z, (-1, 1))

    # The sign for the angle needs to be flipped because we
    # want to rotate back i.e. reverse rotation op we did during
    # transforming box_3d -> box_8c
    ry_sin = tf.sin(-rys)
    ry_cos = tf.cos(-rys)

    zeros = tf.zeros_like(rys, dtype=tf.float32)
    ones = tf.ones_like(rys, dtype=tf.float32)

    rotation_mats = tf.stack([
        tf.stack([ry_cos, zeros, ry_sin], axis=1),
        tf.stack([zeros, ones, zeros], axis=1),
        tf.stack([-ry_sin, zeros, ry_cos], axis=1)], axis=2)

    corners = tf.stack([translated_x,
                        y_corners,
                        translated_z], axis=1)
    # Rotate the corners
    corners_3d = tf.matmul(rotation_mats, corners,
                           transpose_a=True,
                           transpose_b=False)

    # Align the corners in case they are skewed
    aligned_corners = align_boxes_8c(corners_3d)

    # Translate the corners back
    aligned_corners_x = aligned_corners[:, 0] + tf.reshape(center_x, (-1, 1))
    aligned_corners_z = aligned_corners[:, 2] + tf.reshape(center_z, (-1, 1))

    new_x_corners = aligned_corners_x
    new_y_corners = aligned_corners[:, 1]
    new_z_corners = aligned_corners_z

    x_b_right = new_x_corners[:, 1]
    x_b_left = new_x_corners[:, 2]

    z_b_left = new_z_corners[:, 2]
    z_t_left = new_z_corners[:, 3]

    corner_y1 = new_y_corners[:, 0]
    corner_y5 = new_y_corners[:, 4]

    length = x_b_right - x_b_left
    width = z_t_left - z_b_left
    height = corner_y1 - corner_y5

    # Re-calculate the centroid
    center_x = tf.reduce_mean(new_x_corners[:, 0:4], axis=1)
    center_z = tf.reduce_mean(new_z_corners[:, 0:4], axis=1)
    center_y = corner_y1

    box_3d = tf.stack([center_x, center_y, center_z,
                       length, width, height, rys], axis=1)
    return box_3d


def tf_box_8c_to_offsets(boxes_8c,
                         boxes_8c_gt):
    """Converts corner boxes to corner offsets.
    It subtracts the ground-truth box corners from the predicted corners
    and normalizes the offsets by the diagonal of the proposed boxes.

    Args:
        boxes_8c: A tensor of shape (N x 3 x 8) representing the box corners.
        boxes_8c_gt: A tensor of shape (N x 3 x 8) representing the box
            corners ground-truth.
    Returns:
        A tensor of dim (N x 3 x 8) representing the offsets.
    """

    # Get the diagonal of the boxes
    diagonals = tf_box_8c_diagonal_length(boxes_8c)
    offsets = tf.subtract(boxes_8c_gt, boxes_8c)

    # Reshape the offsets to a (24 x N) vector
    reshaped_offsets = tf.reshape(offsets, (24, -1))
    ones = tf.ones_like(reshaped_offsets)
    # This gives diagonals of shape (24 x N)
    # This now enables us to divide acorss N batches
    diagonals_mult = tf.multiply(ones, diagonals)
    # Normalize the offsets by the box_8c diagonal
    offsets_norm = tf.divide(reshaped_offsets, diagonals_mult)

    reshaped_offsets_norm = tf.reshape(offsets_norm,
                                       [-1, 3, 8])

    return reshaped_offsets_norm


def tf_offsets_to_box_8c(boxes_8c,
                         offsets):
    """Converts corner ofsets to box corners.

    It multiplies the diagonals with the offsets and then adds it back
    to the box corners.

    Args:
        box_8c: A tensor of shape (N x 3 x 8) representing the box corners.
        offsets: A tensor vector of shape (N x 3 x 8) representing the corner
            offsets.
    Returns:
        A tensor of dim (N x 3 x 8) representing the corners.
    """
    # Get the diagonal of the boxes
    diagonals = tf_box_8c_diagonal_length(boxes_8c)
    # Reshape the offsets to a (24 x N) vector
    reshaped_offsets = tf.reshape(offsets, (24, -1))
    ones = tf.ones_like(reshaped_offsets)
    # This gives diagonals of shape (24 x N)
    diagonals_mult = tf.multiply(ones, diagonals)

    offsets_back = tf.multiply(reshaped_offsets, diagonals_mult)
    reshaped_offsets_back = tf.reshape(offsets_back,
                                       [-1, 3, 8])

    # Multiply the offsets by the normalization factor i.e. diagonals
    return tf.add(reshaped_offsets_back, boxes_8c)


def tf_box_8c_diagonal_length(boxes_8c):
    """Returns the diagonal lengths of box_8c

    Args:
        boxes_3d: An tensor of shape (N x 3 x 8) of boxes in box_8c
            format.
    Returns:
        Diagonal of all boxes, a tensor of (N,) shape.
    """

    # Grab two opposite corners
    p1 = boxes_8c[:, :, 0]
    p7 = boxes_8c[:, :, 6]

    x_diffs = tf.square((p1[:, 0] - p7[:, 0]))
    y_diffs = tf.square((p1[:, 1] - p7[:, 1]))
    z_diffs = tf.square((p1[:, 2] - p7[:, 2]))

    return tf.sqrt(x_diffs + y_diffs + z_diffs)

"""
Projects anchors into bird's eye view and image space.
Returns the minimum and maximum box corners, and will only work
for anchors rotated at 0 or 90 degrees
"""

import numpy as np
import tensorflow as tf

from wavedata.tools.core import calib_utils


def project_to_bev(anchors, bev_extents):
    """
    Projects an array of 3D anchors into bird's eye view

    Args:
        anchors: list of anchors in anchor format (N x 6):
            N x [x, y, z, dim_x, dim_y, dim_z],
            can be a numpy array or tensor
        bev_extents: xz extents of the 3d area
            [[min_x, max_x], [min_z, max_z]]

    Returns:
          box_corners_norm: corners as a percentage of the map size, in the
            format N x [x1, y1, x2, y2]. Origin is the top left corner
    """
    tensor_format = isinstance(anchors, tf.Tensor)

    if not tensor_format:
        anchors = np.asarray(anchors)

    x = anchors[:, 0]
    z = anchors[:, 2]
    half_dim_x = anchors[:, 3] / 2.0
    half_dim_z = anchors[:, 5] / 2.0

    # Calculate extent ranges
    bev_x_extents_min = bev_extents[0][0]
    bev_z_extents_min = bev_extents[1][0]
    bev_x_extents_max = bev_extents[0][1]
    bev_z_extents_max = bev_extents[1][1]
    bev_x_extents_range = bev_x_extents_max - bev_x_extents_min
    bev_z_extents_range = bev_z_extents_max - bev_z_extents_min

    # 2D corners (top left, bottom right)
    x1 = x - half_dim_x
    x2 = x + half_dim_x
    # Flip z co-ordinates (origin changes from bottom left to top left)
    z1 = bev_z_extents_max - (z + half_dim_z)
    z2 = bev_z_extents_max - (z - half_dim_z)

    # Stack into (N x 4)
    if tensor_format:
        bev_box_corners = tf.stack([x1, z1, x2, z2], axis=1)
    else:
        bev_box_corners = np.stack([x1, z1, x2, z2], axis=1)

    # Convert from original xz into bev xz, origin moves to top left
    bev_extents_min_tiled = [bev_x_extents_min, bev_z_extents_min,
                             bev_x_extents_min, bev_z_extents_min]
    bev_box_corners = bev_box_corners - bev_extents_min_tiled

    # Calculate normalized box corners for ROI pooling
    extents_tiled = [bev_x_extents_range, bev_z_extents_range,
                     bev_x_extents_range, bev_z_extents_range]
    bev_box_corners_norm = bev_box_corners / extents_tiled

    return bev_box_corners, bev_box_corners_norm


def project_to_image_space(anchors, stereo_calib_p2, image_shape):
    """
    Projects 3D anchors into image space

    Args:
        anchors: list of anchors in anchor format N x [x, y, z,
            dim_x, dim_y, dim_z]
        stereo_calib_p2: stereo camera calibration p2 matrix
        image_shape: dimensions of the image [h, w]

    Returns:
        box_corners: corners in image space - N x [x1, y1, x2, y2]
        box_corners_norm: corners as a percentage of the image size -
            N x [x1, y1, x2, y2]
    """
    if anchors.shape[1] != 6:
        raise ValueError("Invalid shape for anchors {}, should be "
                         "(N, 6)".format(anchors.shape[1]))

    # Figure out box mins and maxes
    x = (anchors[:, 0])
    y = (anchors[:, 1])
    z = (anchors[:, 2])

    dim_x = (anchors[:, 3])
    dim_y = (anchors[:, 4])
    dim_z = (anchors[:, 5])

    dim_x_half = dim_x / 2.
    dim_z_half = dim_z / 2.

    # Calculate 3D BB corners
    x_corners = np.array([x + dim_x_half,
                          x + dim_x_half,
                          x - dim_x_half,
                          x - dim_x_half,
                          x + dim_x_half,
                          x + dim_x_half,
                          x - dim_x_half,
                          x - dim_x_half]).T.reshape(1, -1)

    y_corners = np.array([y,
                          y,
                          y,
                          y,
                          y - dim_y,
                          y - dim_y,
                          y - dim_y,
                          y - dim_y]).T.reshape(1, -1)

    z_corners = np.array([z + dim_z_half,
                          z - dim_z_half,
                          z - dim_z_half,
                          z + dim_z_half,
                          z + dim_z_half,
                          z - dim_z_half,
                          z - dim_z_half,
                          z + dim_z_half]).T.reshape(1, -1)

    anchor_corners = np.vstack([x_corners, y_corners, z_corners])

    # Apply the 2D image plane transformation
    pts_2d = calib_utils.project_to_image(anchor_corners, stereo_calib_p2)

    # Get the min and maxes of image coordinates
    i_axis_min_points = np.amin(pts_2d[0, :].reshape(-1, 8), axis=1)
    j_axis_min_points = np.amin(pts_2d[1, :].reshape(-1, 8), axis=1)

    i_axis_max_points = np.amax(pts_2d[0, :].reshape(-1, 8), axis=1)
    j_axis_max_points = np.amax(pts_2d[1, :].reshape(-1, 8), axis=1)

    box_corners = np.vstack([i_axis_min_points, j_axis_min_points,
                             i_axis_max_points, j_axis_max_points]).T

    # Normalize
    image_shape_h = image_shape[0]
    image_shape_w = image_shape[1]

    image_shape_tiled = [image_shape_w, image_shape_h,
                         image_shape_w, image_shape_h]

    box_corners_norm = box_corners / image_shape_tiled

    return np.array(box_corners, dtype=np.float32), \
        np.array(box_corners_norm, dtype=np.float32)


def tf_project_to_image_space(anchors, stereo_calib_p2, image_shape):
    """
    Projects 3D tensor anchors into image space

    Args:
        anchors: a tensor of anchors in the shape [N, 6].
            The anchors are in the format [x, y, z, dim_x, dim_y, dim_z]
        stereo_calib_p2: tensor [3, 4] stereo camera calibration p2 matrix
        image_shape: a float32 tensor of shape [2]. This is dimension of
            the image [h, w]

    Returns:
        box_corners: a float32 tensor corners in image space -
            N x [x1, y1, x2, y2]
        box_corners_norm: a float32 tensor corners as a percentage
            of the image size - N x [x1, y1, x2, y2]
    """
    if anchors.shape[1] != 6:
        raise ValueError("Invalid shape for anchors {}, should be "
                         "(N, 6)".format(anchors.shape[1]))

    # Figure out box mins and maxes
    x = (anchors[:, 0])
    y = (anchors[:, 1])
    z = (anchors[:, 2])

    dim_x = (anchors[:, 3])
    dim_y = (anchors[:, 4])
    dim_z = (anchors[:, 5])

    dim_x_half = dim_x / 2.
    dim_z_half = dim_z / 2.

    # Calculate 3D BB corners
    x_corners = tf.reshape(tf.transpose(tf.stack([x + dim_x_half,
                                                  x + dim_x_half,
                                                  x - dim_x_half,
                                                  x - dim_x_half,
                                                  x + dim_x_half,
                                                  x + dim_x_half,
                                                  x - dim_x_half,
                                                  x - dim_x_half])), (1, -1))

    y_corners = tf.reshape(tf.transpose(tf.stack([y,
                                                  y,
                                                  y,
                                                  y,
                                                  y - dim_y,
                                                  y - dim_y,
                                                  y - dim_y,
                                                  y - dim_y])), (1, -1))

    z_corners = tf.reshape(tf.transpose(tf.stack([z + dim_z_half,
                                                  z - dim_z_half,
                                                  z - dim_z_half,
                                                  z + dim_z_half,
                                                  z + dim_z_half,
                                                  z - dim_z_half,
                                                  z - dim_z_half,
                                                  z + dim_z_half])), (1, -1))

    anchor_corners = tf.concat([x_corners, y_corners, z_corners], axis=0)

    # Apply the 2D image plane transformation
    pts_2d = project_to_image_tensor(anchor_corners, stereo_calib_p2)

    # Get the min and maxes of image coordinates
    i_axis_min_points = tf.reduce_min(
        tf.reshape(pts_2d[0, :], (-1, 8)), axis=1)
    j_axis_min_points = tf.reduce_min(
        tf.reshape(pts_2d[1, :], (-1, 8)), axis=1)

    i_axis_max_points = tf.reduce_max(
        tf.reshape(pts_2d[0, :], (-1, 8)), axis=1)
    j_axis_max_points = tf.reduce_max(
        tf.reshape(pts_2d[1, :], (-1, 8)), axis=1)

    box_corners = tf.transpose(
        tf.stack(
            [i_axis_min_points, j_axis_min_points, i_axis_max_points,
             j_axis_max_points],
            axis=0))

    # Normalize
    image_shape_h = image_shape[0]
    image_shape_w = image_shape[1]

    image_shape_tiled = tf.stack([image_shape_w, image_shape_h,
                                  image_shape_w, image_shape_h], axis=0)

    box_corners_norm = tf.divide(box_corners, image_shape_tiled)

    return box_corners, box_corners_norm


def reorder_projected_boxes(box_corners):
    """Helper function to reorder image corners.

    This reorders the corners from [x1, y1, x2, y2] to
    [y1, x1, y2, x2] which is required by the tf.crop_and_resize op.

    Args:
        box_corners: tensor image corners in the format
            N x [x1, y1, x2, y2]

    Returns:
        box_corners_reordered: tensor image corners in the format
            N x [y1, x1, y2, x2]
    """
    boxes_reordered = tf.stack([box_corners[:, 1],
                                box_corners[:, 0],
                                box_corners[:, 3],
                                box_corners[:, 2]],
                               axis=1)
    return boxes_reordered


def project_to_image_tensor(points_3d, cam_p2_matrix):
    """Projects 3D points to 2D points in image space.

    Args:
        points_3d: a list of float32 tensor of shape [3, None]
        cam_p2_matrix: a float32 tensor of shape [3, 4] representing
            the camera matrix.

    Returns:
        points_2d: a list of float32 tensor of shape [2, None]
            This is the projected 3D points into 2D .i.e. corresponding
            3D points in image coordinates.
    """
    ones_column = tf.ones([1, tf.shape(points_3d)[1]])

    # Add extra column of ones
    points_3d_concat = tf.concat([points_3d, ones_column], axis=0)

    # Multiply camera matrix by the 3D points
    points_2d = tf.matmul(cam_p2_matrix, points_3d_concat)

    # 'Tensor' object does not support item assignment
    # so instead get the result of each division and stack
    # the results
    points_2d_c1 = points_2d[0, :] / points_2d[2, :]
    points_2d_c2 = points_2d[1, :] / points_2d[2, :]
    stacked_points_2d = tf.stack([points_2d_c1,
                                  points_2d_c2],
                                 axis=0)

    return stacked_points_2d

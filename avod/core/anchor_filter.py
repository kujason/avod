import numpy as np

from wavedata.tools.core.integral_image import IntegralImage
from wavedata.tools.core.integral_image_2d import IntegralImage2D

from avod.core import format_checker


def get_empty_anchor_filter(anchors, voxel_grid_3d, density_threshold=1):
    """ Returns a filter for empty boxes from the given 3D anchor list

    Args:
        anchors: list of 3d anchors in the format
            N x [x, y, z, dim_x, dim_y, dim_z]
        voxel_grid_3d: a VoxelGrid object containing a 3D voxel grid of
            pointcloud used to filter the anchors
        density_threshold: minimum number of points in voxel to keep the anchor

    Returns:
        anchor filter: N Boolean mask
    """
    format_checker.check_anchor_format(anchors)

    # Get Integral image of the voxel, add 1 since filled = 0, empty is -1
    integral_image = IntegralImage(voxel_grid_3d.leaf_layout + 1)

    # Make cuboid container
    cuboid_container = np.zeros([len(anchors), 6]).astype(np.uint32)

    top_left_up = np.zeros([len(anchors), 3]).astype(np.float32)
    bot_right_down = np.zeros([len(anchors), 3]).astype(np.float32)

    # Calculate minimum corner
    top_left_up[:, 0] = anchors[:, 0] - (anchors[:, 3] / 2.)
    top_left_up[:, 1] = anchors[:, 1] - (anchors[:, 4])
    top_left_up[:, 2] = anchors[:, 2] - (anchors[:, 5] / 2.)

    # Calculate maximum corner
    bot_right_down[:, 0] = anchors[:, 0] + (anchors[:, 3] / 2.)
    bot_right_down[:, 1] = anchors[:, 1]
    bot_right_down[:, 2] = anchors[:, 2] + (anchors[:, 5] / 2.)

    # map_to_index() expects N x 3 points
    cuboid_container[:, :3] = voxel_grid_3d.map_to_index(
        top_left_up)
    cuboid_container[:, 3:] = voxel_grid_3d.map_to_index(
        bot_right_down)

    # Transpose to pass into query()
    cuboid_container = cuboid_container.T

    # Get point density score for each cuboid
    point_density_score = integral_image.query(cuboid_container)

    # Create the filter
    anchor_filter = point_density_score >= density_threshold

    # Flatten into shape (N,)
    anchor_filter = anchor_filter.flatten()

    return anchor_filter


def get_empty_anchor_filter_2d(anchors, voxel_grid_2d, density_threshold=1):
    """ Returns a filter for empty anchors from the given 2D anchor list

    Args:
        anchors: list of 3d anchors in the format
            N x [x, y, z, dim_x, dim_y, dim_z]
        voxel_grid_2d: a VoxelGrid object containing a 2D voxel grid of
            point cloud used to filter the anchors
        density_threshold: minimum number of points in voxel to keep the anchor

    Returns:
        anchor filter: N Boolean mask
    """
    format_checker.check_anchor_format(anchors)

    # Remove y dimensions from anchors to project into BEV
    anchors_2d = anchors[:, [0, 2, 3, 5]]

    # Get Integral image of the voxel, add 1 since filled = 0, empty is -1
    leaf_layout = voxel_grid_2d.leaf_layout_2d + 1
    leaf_layout = np.squeeze(leaf_layout)
    integral_image = IntegralImage2D(leaf_layout)

    # Make anchor container
    anchor_container = np.zeros([len(anchors_2d), 4]).astype(np.uint32)

    num_anchors = len(anchors_2d)

    # Set up objects containing corners of anchors
    top_left_up = np.zeros([num_anchors, 2]).astype(np.float32)
    bot_right_down = np.zeros([num_anchors, 2]).astype(np.float32)

    # Calculate minimum corner
    top_left_up[:, 0] = anchors_2d[:, 0] - (anchors_2d[:, 2] / 2.)
    top_left_up[:, 1] = anchors_2d[:, 1] - (anchors_2d[:, 3] / 2.)

    # Calculate maximum corner
    bot_right_down[:, 0] = anchors_2d[:, 0] + (anchors_2d[:, 2] / 2.)
    bot_right_down[:, 1] = anchors_2d[:, 1] + (anchors_2d[:, 3] / 2.)

    # map_to_index() expects N x 2 points
    anchor_container[:, :2] = voxel_grid_2d.map_to_index(
        top_left_up)
    anchor_container[:, 2:] = voxel_grid_2d.map_to_index(
        bot_right_down)

    # Transpose to pass into query()
    anchor_container = anchor_container.T

    # Get point density score for each anchor
    point_density_score = integral_image.query(anchor_container)

    # Create the filter
    anchor_filter = point_density_score >= density_threshold

    return anchor_filter


def get_iou_filter(iou_list, iou_range):
    """Returns a boolean filter array that is the output of a given IoU range

    Args:
        iou_list: A numpy array with a list of IoU values
        iou_range: A list of [lower_bound, higher_bound] for IoU range

    Returns:
        iou_filter: A numpy array of booleans that filters for valid range
    """
    # Get bounds
    lower_bound = iou_range[0]
    higher_bound = iou_range[1]

    min_valid_list = lower_bound < iou_list
    max_valid_list = iou_list < higher_bound

    # Get filter for values in between
    iou_filter = np.logical_and(min_valid_list, max_valid_list)

    return iou_filter

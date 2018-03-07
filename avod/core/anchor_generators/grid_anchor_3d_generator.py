"""
Generates 3D anchors, placing them on the ground plane
"""

import numpy as np

from avod.core import anchor_generator


class GridAnchor3dGenerator(anchor_generator.AnchorGenerator):

    def name_scope(self):
        return 'GridAnchor3dGenerator'

    def _generate(self, **params):
        """
        Generates 3D anchors in a grid in the provided 3d area and places
        them on the ground_plane.

        Args:
            **params:
                area_3d: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]

        Returns:
            list of 3D anchors in the form N x [x, y, z, l, w, h, ry]
        """

        area_3d = params.get('area_3d')
        anchor_3d_sizes = params.get('anchor_3d_sizes')
        anchor_stride = params.get('anchor_stride')
        ground_plane = params.get('ground_plane')

        return tile_anchors_3d(area_3d,
                               anchor_3d_sizes,
                               anchor_stride,
                               ground_plane)


def tile_anchors_3d(area_extents,
                    anchor_3d_sizes,
                    anchor_stride,
                    ground_plane):
    """
    Tiles anchors over the area extents by using meshgrids to
    generate combinations of (x, y, z), (l, w, h) and ry.

    Args:
        area_extents: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        anchor_3d_sizes: list of 3d anchor sizes N x (l, w, h)
        anchor_stride: stride lengths (x_stride, z_stride)
        ground_plane: coefficients of the ground plane e.g. [0, -1, 0, 0]

    Returns:
        boxes: list of 3D anchors in box_3d format N x [x, y, z, l, w, h, ry]
    """
    # Convert sizes to ndarray
    anchor_3d_sizes = np.asarray(anchor_3d_sizes)

    anchor_stride_x = anchor_stride[0]
    anchor_stride_z = anchor_stride[1]
    anchor_rotations = np.asarray([0, np.pi / 2.0])

    x_start = area_extents[0][0] + anchor_stride[0] / 2.0
    x_end = area_extents[0][1]
    x_centers = np.array(np.arange(x_start, x_end, step=anchor_stride_x),
                         dtype=np.float32)

    z_start = area_extents[2][1] - anchor_stride[1] / 2.0
    z_end = area_extents[2][0]
    z_centers = np.array(np.arange(z_start, z_end, step=-anchor_stride_z),
                         dtype=np.float32)

    # Use ranges for substitution
    size_indices = np.arange(0, len(anchor_3d_sizes))
    rotation_indices = np.arange(0, len(anchor_rotations))

    # Generate matrix for substitution
    # e.g. for two sizes and two rotations
    # [[x0, z0, 0, 0], [x0, z0, 0, 1], [x0, z0, 1, 0], [x0, z0, 1, 1],
    #  [x1, z0, 0, 0], [x1, z0, 0, 1], [x1, z0, 1, 0], [x1, z0, 1, 1], ...]
    before_sub = np.stack(np.meshgrid(x_centers,
                                      z_centers,
                                      size_indices,
                                      rotation_indices),
                          axis=4).reshape(-1, 4)

    # Place anchors on the ground plane
    a, b, c, d = ground_plane
    all_x = before_sub[:, 0]
    all_z = before_sub[:, 1]
    all_y = -(a * all_x + c * all_z + d) / b

    # Create empty matrix to return
    num_anchors = len(before_sub)
    all_anchor_boxes_3d = np.zeros((num_anchors, 7))

    # Fill in x, y, z
    all_anchor_boxes_3d[:, 0:3] = np.stack((all_x, all_y, all_z), axis=1)

    # Fill in shapes
    sizes = anchor_3d_sizes[np.asarray(before_sub[:, 2], np.int32)]
    all_anchor_boxes_3d[:, 3:6] = sizes

    # Fill in rotations
    rotations = anchor_rotations[np.asarray(before_sub[:, 3], np.int32)]
    all_anchor_boxes_3d[:, 6] = rotations

    return all_anchor_boxes_3d

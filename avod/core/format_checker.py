"""
This module checks for the correct data format and dimensions.

The three different anchor formats as well as corner representations are
used throughout the network, this is just a sanity check before format
conversions.

The ObjectLabel format has the following properties:
- ObjectLabel format is well-documented in the object_utils in wavedata
- It is used more for encoding a generic label class.
- It is a format used in evaluation for Kitti.
- This format is useful for interfacing with obj_utils operations. Note: most
  of the obj_utils operation is not optimized for batch operations and it might
  be slow.

The anchor format is the following [x, y, z, dim_x, dim_y, dim_z] (N x 6):
- [x, y, z] are real number along their respective axis in [metres]
- [dim_x, dim_y, dim_z] are real numbers representing the size of
  the box along their respective axis in [metres]
- This form does not encode rotation, and thus is a natural form to use for
  anchor generation and evaluation.

The box_3d format is the following format [x, y, z, l, w, h, ry] (N x 7):
- [x, y, z] are real number along their respective axis in [metres]
- [l, w, h] are real numbers representing the size of the box
- [ry] is the yaw rotation along the y axis of the camera coordinate.
  It is a value between [-pi ... pi].
- This format is used to simply encode a common 3D box with a single rotation
  in the y axis, which makes it useful for BEV operations.

The box_8c format is the following [[x1,...,x8],[y1...,y8], [z1,...,z8]]
(N x 3 x 8):
- [x1, ..., x8] are the corners in the x-axis
- [y1, ..., y8] are the corners in the y-axis
- [z1, ..., z8] are the corners in the z-axis

The box_8co format is the same as box_8c, except that the corners are ordered,
i.e. the order of corners are preserved throughout the conversion.

The box_4c format is the following
[[x1, x2, x3, x4, z1, z2, z3, z4, h1, h2]] (N x 10):
- [x1, x2, x3, x4, z1, z2, z3, z4] are the corners in the xz plane,
    numbered clockwise starting at the top right
- [h1] is the height above the ground plane to the bottom of the box
- [h2] is the height above the ground plane to the top of the box
"""

import numpy as np
import tensorflow as tf

from wavedata.tools.obj_detection import obj_utils


def check_object_label_format(input_data):
    """Checks for correct ObjectLabel format. If not proper type, raises error.

    Args:
        input_data: input array or tensor to check for valid ObjectLabel format
    """

    # Since the object label is a list, it becomes slow to check
    # Check if it is a ObjectLabel format or a list of Object Label format
    if isinstance(input_data, list):
        if not all(isinstance(x, obj_utils.ObjectLabel) for x in input_data):
            raise TypeError('Given input is not consistent ObjectLabel type.')

        for label in input_data:
            # Check for range of values
            if len(label.t) != 3:
                raise TypeError('Object Label centroid size is wrong.')

    elif isinstance(input_data, obj_utils.ObjectLabel):
        # is a single instance of object label
        if len(input_data.t) != 3:
            raise TypeError('Object Label translation size is wrong.')
    else:
        # not a list of object labels nor a single instance of object label
        raise TypeError('Given input is not an ObjectLabel.')


def check_anchor_format(input_data):
    """Checks for correct anchor format. If not proper type, raises error.

    Args:
        input_data: input numpy array or tensor to check for valid anchor format
    """

    # Check type as either tensor or numpy.ndarray
    if isinstance(input_data, np.ndarray):
        # Check for size for numpy array form (N x 6)
        if input_data.ndim == 2:
            if input_data.shape[1] != 6:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be N x 6 for anchor.')
        elif input_data.ndim == 1:
            if input_data.shape[0] != 6:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be 6 for anchor.')
    elif isinstance(input_data, tf.Tensor):
        # if tensor, check the shape
        if isinstance(input_data, tf.Tensor):
            if input_data.shape[1] != 6:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be N x 6 for box_3d.')
    else:
        raise TypeError('Given input is not of valid types.'
                        '(i.e. np.ndarray or tf.Tensor)')


def check_box_3d_format(input_data):
    """Checks for correct box_3d format. If not proper type, raises error.

    Args:
        input_data: input numpy array or tensor to check for valid box_3d format
    """

    # Check type
    if isinstance(input_data, np.ndarray):
        # Check for size for numpy array form (N x 7)
        if input_data.ndim == 2:
            if input_data.shape[1] != 7:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be N x 7 for box_3d.')
        elif input_data.ndim == 1:
            if input_data.shape[0] != 7:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be 7 for box_3d.')

    elif isinstance(input_data, tf.Tensor):
        # if tensor, check the shape
        if isinstance(input_data, tf.Tensor):
            if input_data.shape[1] != 7:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be N x 7 for box_3d.')
    else:
        raise TypeError('Given input is not of valid types.'
                        '(i.e. np.ndarray or tf.Tensor)')


def check_box_8c_format(input_data):
    """Checks for correct box_8c format. If not proper type, raises error.

    Args:
        input_data: input numpy array or tensor to check for valid box_8c
            format

    Raises:
        ValueError: if input_data with invalid dimensions is given.
    """

    if isinstance(input_data, np.ndarray):
        # Check for size for numpy array form (N x 3 x 8)
        if input_data.ndim == 3:
            if input_data.shape[1:] != (3, 8):
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be N x 3 x 8 for box_8c.')
        elif input_data.ndim == 2:
            if input_data.shape != (3, 8):
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be 3 x 8 for box_8c.')
    elif isinstance(input_data, tf.Tensor):
        # if tensor, check the shape
        if isinstance(input_data, tf.Tensor):
            if input_data.shape[1:] != (3, 8):
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be N x 3 x 8 for box_8c.')
    else:
        raise TypeError('Given input is not of valid types.'
                        '(i.e. np.ndarray or tf.Tensor)')


def check_box_4c_format(input_data):
    """Checks for correct box_4c format. If not proper type, raises error.

    Args:
        input_data: input numpy array or tensor to check for valid box_8c
            format

    Raises:
        ValueError: if input_data with invalid dimensions is given.
    """

    if isinstance(input_data, np.ndarray):
        # Check for size for numpy array
        if input_data.ndim > 2 or input_data.shape[-1] != 10:
            raise TypeError('Given input does not have valid number of '
                            'attributes. Should be N x 10 for box_4c.')
    elif isinstance(input_data, tf.Tensor):
        # if tensor, check the shape
        if isinstance(input_data, tf.Tensor):
            if input_data.shape[1] != 10:
                raise TypeError('Given input does not have valid number of '
                                'attributes. Should be N x 10 for box_4c.')
    else:
        raise TypeError('Given input is not of valid types.'
                        '(i.e. np.ndarray or tf.Tensor)')

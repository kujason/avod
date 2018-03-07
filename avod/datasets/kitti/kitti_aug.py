import copy

import numpy as np

AUG_FLIPPING = 'flipping'
AUG_PCA_JITTER = 'pca_jitter'


def flip_image(image):
    """Flips an image horizontally
    """
    flipped_image = np.fliplr(image)
    return flipped_image


def flip_points(points):
    """Flips a list of points (N, 3)
    """
    flipped_points = np.copy(points)
    flipped_points[:, 0] = -points[:, 0]
    return flipped_points


def flip_point_cloud(point_cloud):
    """Flips a point cloud (3, N)
    """
    flipped_point_cloud = np.copy(point_cloud)
    flipped_point_cloud[0] = -point_cloud[0]
    return flipped_point_cloud


def flip_label_in_3d_only(obj_label):
    """Flips only the 3D position of an object label. The 2D bounding box is
    not flipped to save time since it is not used.

    Args:
        obj_label: ObjectLabel

    Returns:
        A flipped object
    """

    flipped_label = copy.deepcopy(obj_label)

    # Flip the rotation
    if obj_label.ry >= 0:
        flipped_label.ry = np.pi - obj_label.ry
    else:
        flipped_label.ry = -np.pi - obj_label.ry

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_t = (-flipped_label.t[0], flipped_label.t[1], flipped_label.t[2])
    flipped_label.t = flipped_t

    return flipped_label


def flip_boxes_3d(boxes_3d, flip_ry=True):
    """Flips boxes_3d

    Args:
        boxes_3d: List of boxes in box_3d format
        flip_ry bool: (optional) if False, rotation is not flipped to save on
            computation (useful for flipping anchors)

    Returns:
        flipped_boxes_3d: Flipped boxes in box_3d format
    """

    flipped_boxes_3d = np.copy(boxes_3d)

    if flip_ry:
        # Flip the rotation
        above_zero = boxes_3d[:, 6] >= 0
        below_zero = np.logical_not(above_zero)
        flipped_boxes_3d[above_zero, 6] = np.pi - boxes_3d[above_zero, 6]
        flipped_boxes_3d[below_zero, 6] = -np.pi - boxes_3d[below_zero, 6]

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_boxes_3d[:, 0] = -boxes_3d[:, 0]

    return flipped_boxes_3d


def flip_ground_plane(ground_plane):
    """Flips the ground plane by negating the x coefficient
        (ax + by + cz + d = 0)

    Args:
        ground_plane: ground plane coefficients

    Returns:
        Flipped ground plane coefficients
    """
    flipped_ground_plane = np.copy(ground_plane)
    flipped_ground_plane[0] = -ground_plane[0]
    return flipped_ground_plane


def flip_stereo_calib_p2(calib_p2, image_shape):
    """Flips the stereo calibration matrix to correct the projection back to
    image space. Flipping the image can be seen as a movement of both the
    camera plane, and the camera itself. To account for this, the instrinsic
    matrix x0 value is flipped with respect to the image width, and the
    extrinsic matrix t1 value is negated.

    Args:
        calib_p2: 3 x 4 stereo camera calibration matrix
        image_shape: (h, w) image shape

    Returns:
        'Flipped' calibration p2 matrix with shape (3, 4)
    """
    flipped_p2 = np.copy(calib_p2)
    flipped_p2[0, 2] = image_shape[1] - calib_p2[0, 2]
    flipped_p2[0, 3] = -calib_p2[0, 3]

    return flipped_p2


def compute_pca(image_set):
    """Calculates and returns PCA of a set of images

    Args:
        image_set: List of images read with cv2.imread in np.uint8 format

    Returns:
        PCA for the set of images
    """

    # Check for valid input
    assert(image_set[0].dtype == np.uint8)

    # Reshape data into single array
    reshaped_data = np.concatenate([image
                                    for pixels in image_set for image in
                                    pixels])

    # Convert to float and normalize the data between [0, 1]
    reshaped_data = (reshaped_data / 255.0).astype(np.float32)

    # Calculate covariance, eigenvalues, and eigenvectors
    # np.cov calculates covariance around the mean, so no need to shift the
    # data
    covariance = np.cov(reshaped_data.T)
    e_vals, e_vecs = np.linalg.eigh(covariance)

    # svd can also be used instead
    # U, S, V = np.linalg.svd(mean_data)

    pca = np.sqrt(e_vals) * e_vecs

    return pca


def add_pca_jitter(img_data, pca):
    """Adds a multiple of the principle components,
    with magnitude from a Gaussian distribution with mean 0 and stdev 0.1


    Args:
        img_data: Original image in read with cv2.imread in np.uint8 format
        pca: PCA calculated with compute_PCA for the image set

    Returns:
        Image with added noise
    """

    # Check for valid input
    assert (img_data.dtype == np.uint8)

    # Make a copy of the image data
    new_img_data = np.copy(img_data).astype(np.float32) / 255.0

    # Calculate noise by multiplying pca with magnitude,
    # then sum horizontally since eigenvectors are in columns
    magnitude = np.random.randn(3) * 0.1
    noise = (pca * magnitude).sum(axis=1)

    # Add the noise to the image, and clip to valid range [0, 1]
    new_img_data = new_img_data + noise
    np.clip(new_img_data, 0.0, 1.0, out=new_img_data)

    # Change back to np.uint8
    new_img_data = (new_img_data * 255).astype(np.uint8)

    return new_img_data


def apply_pca_jitter(image_in):
    """Applies PCA jitter or random noise to a single image

    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    image_in = np.asarray([image_in], dtype=np.uint8)

    pca = compute_pca(image_in)
    image_out = add_pca_jitter(image_in, pca)

    return image_out

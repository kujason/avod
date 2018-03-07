import copy
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils

from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_projector, box_3d_encoder
from avod.datasets.kitti import kitti_aug


def main():
    """This demo runs through all samples in the trainval set, and checks
    that the 3D box projection of all 'Car', 'Van', 'Pedestrian', and 'Cyclist'
    objects are in the correct flipped 2D location after applying
    modifications to the stereo p2 matrix.
    """

    dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAINVAL,
                                                 use_defaults=True)

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    all_samples = dataset.sample_names

    all_pixel_errors = []
    all_max_pixel_errors = []

    total_flip_time = 0.0

    for sample_idx in range(dataset.num_samples):

        sys.stdout.write('\r{} / {}'.format(sample_idx,
                                            dataset.num_samples - 1))

        sample_name = all_samples[sample_idx]

        img_idx = int(sample_name)

        # Run the main loop to run throughout the images
        frame_calibration_info = calib_utils.read_calibration(
            dataset.calib_dir,
            img_idx)

        # Load labels
        gt_labels = obj_utils.read_labels(dataset.label_dir, img_idx)
        gt_labels = dataset.kitti_utils.filter_labels(
            gt_labels, ['Car', 'Van', 'Pedestrian', 'Cyclist'])

        image = cv2.imread(dataset.get_rgb_image_path(sample_name))
        image_size = [image.shape[1], image.shape[0]]

        # Flip p2 matrix
        calib_p2 = frame_calibration_info.p2
        flipped_p2 = np.copy(calib_p2)
        flipped_p2[0, 2] = image.shape[1] - flipped_p2[0, 2]
        flipped_p2[0, 3] = -flipped_p2[0, 3]

        for obj_idx in range(len(gt_labels)):

            obj = gt_labels[obj_idx]

            # Get original 2D bounding boxes
            orig_box_3d = box_3d_encoder.object_label_to_box_3d(obj)
            orig_bbox_2d = box_3d_projector.project_to_image_space(
                orig_box_3d, calib_p2, truncate=True, image_size=image_size)

            # Skip boxes outside image
            if orig_bbox_2d is None:
                continue

            orig_bbox_2d_flipped = flip_box_2d(orig_bbox_2d, image_size)

            # Do flipping
            start_time = time.time()
            flipped_obj = kitti_aug.flip_label_in_3d_only(obj)
            flip_time = time.time() - start_time
            total_flip_time += flip_time

            box_3d_flipped = box_3d_encoder.object_label_to_box_3d(flipped_obj)
            new_bbox_2d_flipped = box_3d_projector.project_to_image_space(
                box_3d_flipped, flipped_p2, truncate=True,
                image_size=image_size)

            pixel_errors = new_bbox_2d_flipped - orig_bbox_2d_flipped
            max_pixel_error = np.amax(np.abs(pixel_errors))

            all_pixel_errors.append(pixel_errors)
            all_max_pixel_errors.append(max_pixel_error)

            if max_pixel_error > 5:
                print(' Error > 5px', sample_idx, max_pixel_error)
                print(np.round(orig_bbox_2d_flipped, 3),
                      np.round(new_bbox_2d_flipped, 3))

    print('Avg flip time:', total_flip_time / dataset.num_samples)

    # Convert to ndarrays
    all_pixel_errors = np.asarray(all_pixel_errors)
    all_max_pixel_errors = np.asarray(all_max_pixel_errors)

    # Print max values
    print(np.amax(all_max_pixel_errors))

    # Plot pixel errors
    fig, axes = plt.subplots(nrows=3, ncols=1)
    ax0, ax1, ax2 = axes.flatten()

    ax0.hist(all_pixel_errors[:, 0], 50, histtype='bar', facecolor='green')
    ax1.hist(all_pixel_errors[:, 2], 50, histtype='bar', facecolor='green')
    ax2.hist(all_max_pixel_errors, 50, histtype='bar', facecolor='green')

    plt.show()


def flip_box_2d(box_2d, im_size):

    flipped_box_2d = copy.deepcopy(box_2d)

    # Flip in 2D
    x1 = flipped_box_2d[0]
    x2 = flipped_box_2d[2]

    half_width = im_size[0] / 2.0

    diff = x1 - half_width

    # width of bounding box
    width_bb = x2 - x1

    if x1 < half_width:
        new_x2 = half_width + abs(diff)
    else:
        new_x2 = half_width - abs(diff)
    new_x1 = new_x2 - width_bb

    # since we are doing mirror flip,
    # the y's remain unchanged
    flipped_box_2d[0] = int(new_x1)
    flipped_box_2d[2] = int(new_x2)

    return flipped_box_2d


if __name__ == '__main__':
    main()

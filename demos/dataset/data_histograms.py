import sys

import matplotlib.pyplot as plt
import numpy as np

from wavedata.tools.obj_detection import obj_utils

from avod.builders.dataset_builder import DatasetBuilder


def main():
    """Show histograms of ground truth labels
    """

    dataset = DatasetBuilder.build_kitti_dataset(
        # DatasetBuilder.KITTI_TRAIN
        # DatasetBuilder.KITTI_VAL
        DatasetBuilder.KITTI_TRAINVAL
    )

    difficulty = 2

    centroid_bins = 51
    dimension_bins = 21
    orientation_bins = 65

    classes = ['Car']
    # classes = ['Pedestrian']
    # classes = ['Cyclist']
    # classes = ['Pedestrian', 'Cyclist']

    # Dataset values
    num_samples = dataset.num_samples

    all_centroids_x = []
    all_centroids_y = []
    all_centroids_z = []
    all_lengths = []
    all_widths = []
    all_heights = []
    all_orientations = []

    # Counter for total number of valid samples
    num_valid_samples = 0

    for sample_idx in range(num_samples):

        sys.stdout.write('\r{} / {}'.format(sample_idx + 1, num_samples))

        sample_name = dataset.sample_names[sample_idx]
        img_idx = int(sample_name)

        obj_labels = obj_utils.read_labels(dataset.label_dir, img_idx)
        obj_labels = dataset.kitti_utils.filter_labels(obj_labels,
                                                       classes=classes,
                                                       difficulty=difficulty)

        centroids = np.asarray([obj.t for obj in obj_labels])
        lengths = np.asarray([obj.l for obj in obj_labels])
        widths = np.asarray([obj.w for obj in obj_labels])
        heights = np.asarray([obj.h for obj in obj_labels])
        orientations = np.asarray([obj.ry for obj in obj_labels])

        if any(orientations) and np.amax(np.abs(orientations) > np.pi):
            raise ValueError('Invalid orientation')

        if len(centroids) > 0:
            all_centroids_x.extend(centroids[:, 0])
            all_centroids_y.extend(centroids[:, 1])
            all_centroids_z.extend(centroids[:, 2])
            all_lengths.extend(lengths)
            all_widths.extend(widths)
            all_heights.extend(heights)
            all_orientations.extend(orientations)

            num_valid_samples += 1

    print('Finished reading labels, num_valid_samples', num_valid_samples)

    # Get means
    mean_centroid_x = np.mean(all_centroids_x)
    mean_centroid_y = np.mean(all_centroids_y)
    mean_centroid_z = np.mean(all_centroids_z)
    mean_dims = np.mean([all_lengths, all_widths, all_heights])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print('mean_centroid_x {0:0.3f}'.format(mean_centroid_x))
    print('mean_centroid_y {0:0.3f}'.format(mean_centroid_y))
    print('mean_centroid_z {0:0.3f}'.format(mean_centroid_z))
    print('mean_dims {0:0.3f}'.format(mean_dims))

    # Make plots
    f, ax_arr = plt.subplots(3, 3)

    # xyz
    ax_arr[0, 0].hist(all_centroids_x, centroid_bins, facecolor='green')
    ax_arr[0, 1].hist(all_centroids_y, centroid_bins, facecolor='green')
    ax_arr[0, 2].hist(all_centroids_z, centroid_bins, facecolor='green')

    # lwh
    ax_arr[1, 0].hist(all_lengths, dimension_bins, facecolor='green')
    ax_arr[1, 1].hist(all_widths, dimension_bins, facecolor='green')
    ax_arr[1, 2].hist(all_heights, dimension_bins, facecolor='green')

    # orientations
    ax_arr[2, 0].hist(all_orientations, orientation_bins, facecolor='green')

    plt.show(block=True)


if __name__ == '__main__':
    main()

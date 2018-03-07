import numpy as np
from PIL import Image

from avod.builders.dataset_builder import DatasetBuilder


def main():
    """
    Calculates and prints the mean values for the RGB channels in a dataset
    """

    dataset_builder = DatasetBuilder()
    dataset = dataset_builder.build_kitti_dataset(
        dataset_builder.KITTI_TRAIN
        # dataset_builder.KITTI_TRAIN_MINI
    )

    # Options
    debug_print = True
    get_bev_mean = False

    # Dataset values
    dataset_utils = dataset.kitti_utils
    num_samples = dataset.num_samples
    clusters, _ = dataset.get_cluster_info()
    num_bev_maps = len(clusters) + 1  # Height Maps + Density Map

    pixels_sum = np.zeros(3)  # RGB
    bev_sum = np.zeros(num_bev_maps)

    for sample_idx in range(num_samples):
        sample_name = dataset.sample_names[sample_idx]

        image_path = dataset.get_rgb_image_path(sample_name)
        image = np.asarray(Image.open(image_path))

        pixels_r = np.mean(image[:, :, 0])
        pixels_g = np.mean(image[:, :, 1])
        pixels_b = np.mean(image[:, :, 2])

        pixel_means = np.stack((pixels_r, pixels_g, pixels_b))
        pixels_sum += pixel_means

        if get_bev_mean:
            bev_images = dataset_utils.create_bev_maps(sample_name,
                                                       source='lidar')
            height_maps = np.asarray(bev_images['height_maps'])
            density_map = np.asarray(bev_images['density_map'])

            height_means = [np.mean(height_map) for height_map in height_maps]
            density_mean = np.mean(density_map)

            bev_means = np.stack((*height_means, density_mean))
            bev_sum += bev_means

        if debug_print:
            debug_string = '{} / {}, Sample {}, pixel_means {}'.format(
                sample_idx + 1, num_samples, sample_name, pixel_means)
            if get_bev_mean:
                debug_string += ' density_means {}'.format(bev_means)

            print(debug_string)

    print("Dataset: {}, split: {}".format(dataset.name, dataset.data_split))
    print("Image mean: {}".format(pixels_sum / num_samples))

    if get_bev_mean:
        print("BEV mean: {}".format(bev_sum / num_samples))


if __name__ == '__main__':
    main()

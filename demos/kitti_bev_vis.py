import cv2
from google.protobuf import text_format
import numpy as np
import numpy.random as random

from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization import vis_utils

from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_encoder
from avod.core import box_3d_projector


def draw_boxes(image, boxes_norm):
    """Draws green boxes on the bev image

    Args:
        image: bev image
        boxes_norm: box corners normalized to the size of the image
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Returns:
        The image with boxes drawn on it. If boxes_norm is None,
            returns the original image
    """

    # Draw boxes if they exist
    if boxes_norm is not None:
        # Convert image to 3 channel
        image = (image * 255.0).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for box_points in boxes_norm:
            image_shape = np.flip(image.shape[0:2], axis=0)

            for box_point_idx in range(len(box_points)):

                start_point = box_points[box_point_idx] * image_shape
                end_point = box_points[(box_point_idx + 1) % 4] * image_shape

                start_point = start_point.astype(np.int32)
                end_point = end_point.astype(np.int32)

                cv2.line(
                    image, tuple(start_point),
                    tuple(end_point),
                    (0, 255, 0), thickness=1)

    return image


def main():
    """
    Displays the bird's eye view maps for a KITTI sample.
    """

    ##############################
    # Options
    ##############################

    bev_generator = 'slices'

    slices_config = \
        """
        slices {
            height_lo: -0.2
            height_hi: 2.3
            num_slices: 5
        }
        """

    # Use None for a random image
    img_idx = None
    # img_idx = 142
    # img_idx = 191

    show_ground_truth = True  # Whether to overlay ground_truth boxes

    point_cloud_source = 'lidar'
    ##############################
    # End of Options
    ##############################

    dataset_config = DatasetBuilder.copy_config(DatasetBuilder.KITTI_VAL)
    dataset_config = DatasetBuilder.merge_defaults(dataset_config)

    # Overwrite bev_generator
    if bev_generator == 'slices':
        text_format.Merge(slices_config,
                          dataset_config.kitti_utils_config.bev_generator)
    else:
        raise ValueError('Invalid bev_generator')

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    if img_idx is None:
        img_idx = int(random.random() * dataset.num_samples)

    sample_name = "{:06}".format(img_idx)
    print('=== Showing BEV maps for image: {}.png ==='.format(sample_name))

    # Load image
    image = cv2.imread(dataset.get_rgb_image_path(sample_name))
    image_shape = image.shape[0:2]

    kitti_utils = dataset.kitti_utils
    point_cloud = kitti_utils.get_point_cloud(
        point_cloud_source, img_idx, image_shape)
    ground_plane = kitti_utils.get_ground_plane(sample_name)
    bev_images = kitti_utils.create_bev_maps(point_cloud, ground_plane)

    height_maps = np.array(bev_images.get("height_maps"))
    density_map = np.array(bev_images.get("density_map"))

    box_points, box_points_norm = [None, None]
    if show_ground_truth:
        # Get projected boxes
        obj_labels = obj_utils.read_labels(dataset.label_dir, img_idx)

        filtered_objs = obj_labels

        label_boxes = []
        for label in filtered_objs:
            box = box_3d_encoder.object_label_to_box_3d(label)
            label_boxes.append(box)

        label_boxes = np.array(label_boxes)
        box_points, box_points_norm = box_3d_projector.project_to_bev(
            label_boxes, [[-40, 40], [0, 70]])

    rgb_img_size = (np.array((1242, 375)) * 0.75).astype(np.int16)
    img_x_start = 60
    img_y_start = 330

    img_x = img_x_start
    img_y = img_y_start
    img_w = 400
    img_h = 350
    img_titlebar_h = 20

    # Show images
    vis_utils.cv2_show_image("Image", image,
                             size_wh=rgb_img_size, location_xy=(img_x, 0))

    # Height maps
    for map_idx in range(len(height_maps)):
        height_map = height_maps[map_idx]

        height_map = draw_boxes(height_map, box_points_norm)
        vis_utils.cv2_show_image(
            "Height Map {}".format(map_idx), height_map, size_wh=(
                img_w, img_h), location_xy=(
                img_x, img_y))

        img_x += img_w
        # Wrap around
        if (img_x + img_w) > 1920:
            img_x = img_x_start
            img_y += img_h + img_titlebar_h

    # Density map
    density_map = draw_boxes(density_map, box_points_norm)
    vis_utils.cv2_show_image(
        "Density Map", density_map, size_wh=(
            img_w, img_h), location_xy=(
            img_x, img_y))

    cv2.waitKey()


if __name__ == "__main__":
    main()

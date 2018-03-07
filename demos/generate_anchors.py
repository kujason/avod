import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from wavedata.tools.core import calib_utils
from wavedata.tools.visualization import vis_utils

from avod.builders.dataset_builder import DatasetBuilder
from avod.core import anchor_projector
from avod.core import box_3d_encoder
from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.core.label_cluster_utils import LabelClusterUtils


def main():
    """
    Visualization of 3D grid anchor generation, showing 2D projections
        in BEV and image space, and a 3D display of the anchors
    """
    dataset_config = DatasetBuilder.copy_config(
        DatasetBuilder.KITTI_TRAIN)
    dataset_config.num_clusters[0] = 1
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config)

    label_cluster_utils = LabelClusterUtils(dataset)
    clusters, _ = label_cluster_utils.get_clusters()

    # Options
    img_idx = 1
    # fake_clusters = np.array([[5, 4, 3], [6, 5, 4]])
    # fake_clusters = np.array([[3, 3, 3], [4, 4, 4]])

    fake_clusters = np.array([[4, 2, 3]])
    fake_anchor_stride = [5.0, 5.0]
    ground_plane = [0, -1, 0, 1.72]

    anchor_3d_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()

    area_extents = np.array([[-40, 40], [-5, 5], [0, 70]])

    # Generate anchors for cars only
    start_time = time.time()
    anchor_boxes_3d = anchor_3d_generator.generate(
        area_3d=dataset.kitti_utils.area_extents,
        anchor_3d_sizes=fake_clusters,
        anchor_stride=fake_anchor_stride,
        ground_plane=ground_plane)
    all_anchors = box_3d_encoder.box_3d_to_anchor(anchor_boxes_3d)
    end_time = time.time()
    print("Anchors generated in {} s".format(end_time - start_time))

    # Project into bev
    bev_boxes, bev_normalized_boxes = \
        anchor_projector.project_to_bev(all_anchors, area_extents[[0, 2]])

    bev_fig, (bev_axes, bev_normalized_axes) = \
        plt.subplots(1, 2, figsize=(16, 7))
    bev_axes.set_xlim(0, 80)
    bev_axes.set_ylim(70, 0)
    bev_normalized_axes.set_xlim(0, 1.0)
    bev_normalized_axes.set_ylim(1, 0.0)

    plt.show(block=False)

    for box in bev_boxes:
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]

        rect = patches.Rectangle((box[0], box[1]),
                                 box_w, box_h,
                                 linewidth=2,
                                 edgecolor='b',
                                 facecolor='none')

        bev_axes.add_patch(rect)

    for normalized_box in bev_normalized_boxes:
        box_w = normalized_box[2] - normalized_box[0]
        box_h = normalized_box[3] - normalized_box[1]

        rect = patches.Rectangle((normalized_box[0], normalized_box[1]),
                                 box_w, box_h,
                                 linewidth=2,
                                 edgecolor='b',
                                 facecolor='none')

        bev_normalized_axes.add_patch(rect)

    rgb_fig, rgb_2d_axes, rgb_3d_axes = \
        vis_utils.visualization(dataset.rgb_image_dir, img_idx)
    plt.show(block=False)

    image_path = dataset.get_rgb_image_path(dataset.sample_names[img_idx])
    image_shape = np.array(Image.open(image_path)).shape

    stereo_calib_p2 = calib_utils.read_calibration(dataset.calib_dir,
                                                   img_idx).p2

    start_time = time.time()
    rgb_boxes, rgb_normalized_boxes = \
        anchor_projector.project_to_image_space(all_anchors,
                                                stereo_calib_p2,
                                                image_shape)
    end_time = time.time()
    print("Anchors projected in {} s".format(end_time - start_time))

    # Read the stereo calibration matrix for visualization
    stereo_calib = calib_utils.read_calibration(dataset.calib_dir, 0)
    p = stereo_calib.p2

    # Overlay boxes on images

    for anchor_idx in range(len(anchor_boxes_3d)):
        anchor_box_3d = anchor_boxes_3d[anchor_idx]

        obj_label = box_3d_encoder.box_3d_to_object_label(anchor_box_3d)

        # Draw 3D boxes
        vis_utils.draw_box_3d(rgb_3d_axes, obj_label, p)

        # Draw 2D boxes
        rgb_box_2d = rgb_boxes[anchor_idx]

        box_x1 = rgb_box_2d[0]
        box_y1 = rgb_box_2d[1]
        box_w = rgb_box_2d[2] - box_x1
        box_h = rgb_box_2d[3] - box_y1

        rect = patches.Rectangle((box_x1, box_y1),
                                 box_w, box_h,
                                 linewidth=2,
                                 edgecolor='b',
                                 facecolor='none')

        rgb_2d_axes.add_patch(rect)

        if anchor_idx % 32 == 0:
            rgb_fig.canvas.draw()

    plt.show(block=True)


if __name__ == '__main__':
    main()

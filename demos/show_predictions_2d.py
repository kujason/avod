import os
import sys
import time

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.obj_detection import evaluation
from wavedata.tools.visualization import vis_utils

import avod
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_encoder
from avod.core import box_3d_projector
from avod.core import anchor_projector


BOX_COLOUR_SCHEME = {
    'Car': '#00FF00',           # Green
    'Pedestrian': '#00FFFF',    # Teal
    'Cyclist': '#FFFF00'        # Yellow
}


def main():
    """This demo shows RPN proposals and AVOD predictions in 3D
    and 2D in image space. Given certain thresholds for proposals
    and predictions, it selects and draws the bounding boxes on
    the image sample. It goes through the entire proposal and
    prediction samples for the given dataset split.

    The proposals, overlaid, and prediction images can be toggled on or off
    separately in the options section.
    The prediction score and IoU with ground truth can be toggled on or off
    as well, shown as (score, IoU) above the detection.
    """
    dataset_config = DatasetBuilder.copy_config(DatasetBuilder.KITTI_VAL)

    ##############################
    # Options
    ##############################
    dataset_config.data_split = 'val'

    fig_size = (10, 6.1)

    rpn_score_threshold = 0.1
    avod_score_threshold = 0.1

    gt_classes = ['Car']
    # gt_classes = ['Pedestrian', 'Cyclist']

    # Overwrite this to select a specific checkpoint
    global_step = None
    checkpoint_name = 'avod_cars_example'

    # Drawing Toggles
    draw_proposals_separate = False
    draw_overlaid = False
    draw_predictions_separate = True

    # Show orientation for both GT and proposals/predictions
    draw_orientations_on_prop = False
    draw_orientations_on_pred = False

    # Draw 2D bounding boxes
    draw_projected_2d_boxes = True

    # Save images for samples with no detections
    save_empty_images = True

    draw_score = True
    draw_iou = True
    ##############################
    # End of Options
    ##############################

    # Get the dataset
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config)

    # Setup Paths
    predictions_dir = avod.root_dir() + \
        '/data/outputs/' + checkpoint_name + '/predictions'

    proposals_and_scores_dir = predictions_dir + \
        '/proposals_and_scores/' + dataset.data_split

    predictions_and_scores_dir = predictions_dir + \
        '/final_predictions_and_scores/' + dataset.data_split

    # Output images directories
    output_dir_base = predictions_dir + '/images_2d'

    # Get checkpoint step
    steps = os.listdir(proposals_and_scores_dir)
    steps.sort(key=int)
    print('Available steps: {}'.format(steps))

    # Use latest checkpoint if no index provided
    if global_step is None:
        global_step = steps[-1]

    if draw_proposals_separate:
        prop_out_dir = output_dir_base + '/proposals/{}/{}/{}'.format(
            dataset.data_split, global_step, rpn_score_threshold)

        if not os.path.exists(prop_out_dir):
            os.makedirs(prop_out_dir)

        print('Proposal images saved to:', prop_out_dir)

    if draw_overlaid:
        overlaid_out_dir = output_dir_base + '/overlaid/{}/{}/{}'.format(
            dataset.data_split, global_step, avod_score_threshold)

        if not os.path.exists(overlaid_out_dir):
            os.makedirs(overlaid_out_dir)

        print('Overlaid images saved to:', overlaid_out_dir)

    if draw_predictions_separate:
        pred_out_dir = output_dir_base + '/predictions/{}/{}/{}'.format(
            dataset.data_split, global_step,
            avod_score_threshold)

        if not os.path.exists(pred_out_dir):
            os.makedirs(pred_out_dir)

        print('Prediction images saved to:', pred_out_dir)

    # Rolling average array of times for time estimation
    avg_time_arr_length = 10
    last_times = np.repeat(time.time(), avg_time_arr_length) + \
        np.arange(avg_time_arr_length)

    for sample_idx in range(dataset.num_samples):
        # Estimate time remaining with 5 slowest times
        start_time = time.time()
        last_times = np.roll(last_times, -1)
        last_times[-1] = start_time
        avg_time = np.mean(np.sort(np.diff(last_times))[-5:])
        samples_remaining = dataset.num_samples - sample_idx
        est_time_left = avg_time * samples_remaining

        # Print progress and time remaining estimate
        sys.stdout.write('\rSaving {} / {}, Avg Time: {:.3f}s, '
                         'Time Remaining: {:.2f}s'. format(
                             sample_idx + 1,
                             dataset.num_samples,
                             avg_time,
                             est_time_left))
        sys.stdout.flush()

        sample_name = dataset.sample_names[sample_idx]
        img_idx = int(sample_name)

        ##############################
        # Proposals
        ##############################
        if draw_proposals_separate or draw_overlaid:
            # Load proposals from files
            proposals_file_path = proposals_and_scores_dir + \
                "/{}/{}.txt".format(global_step, sample_name)
            if not os.path.exists(proposals_file_path):
                print('Sample {}: No proposals, skipping'.format(sample_name))
                continue
            print('Sample {}: Drawing proposals'.format(sample_name))

            proposals_and_scores = np.loadtxt(proposals_file_path)

            proposal_boxes_3d = proposals_and_scores[:, 0:7]
            proposal_scores = proposals_and_scores[:, 7]

            # Apply score mask to proposals
            score_mask = proposal_scores > rpn_score_threshold
            proposal_boxes_3d = proposal_boxes_3d[score_mask]
            proposal_scores = proposal_scores[score_mask]

            proposal_objs = \
                [box_3d_encoder.box_3d_to_object_label(proposal,
                                                       obj_type='Proposal')
                 for proposal in proposal_boxes_3d]

        ##############################
        # Predictions
        ##############################
        if draw_predictions_separate or draw_overlaid:
            predictions_file_path = predictions_and_scores_dir + \
                "/{}/{}.txt".format(global_step,
                                    sample_name)
            if not os.path.exists(predictions_file_path):
                continue

            # Load predictions from files
            predictions_and_scores = np.loadtxt(
                predictions_and_scores_dir +
                "/{}/{}.txt".format(global_step,
                                    sample_name))

            prediction_boxes_3d = predictions_and_scores[:, 0:7]
            prediction_scores = predictions_and_scores[:, 7]
            prediction_class_indices = predictions_and_scores[:, 8]

            # process predictions only if we have any predictions left after
            # masking
            if len(prediction_boxes_3d) > 0:

                # Apply score mask
                avod_score_mask = prediction_scores >= avod_score_threshold
                prediction_boxes_3d = prediction_boxes_3d[avod_score_mask]
                prediction_scores = prediction_scores[avod_score_mask]
                prediction_class_indices = \
                    prediction_class_indices[avod_score_mask]

                # # Swap l, w for predictions where w > l
                # swapped_indices = \
                #     prediction_boxes_3d[:, 4] > prediction_boxes_3d[:, 3]
                # prediction_boxes_3d = np.copy(prediction_boxes_3d)
                # prediction_boxes_3d[swapped_indices, 3] = \
                #     prediction_boxes_3d[swapped_indices, 4]
                # prediction_boxes_3d[swapped_indices, 4] = \
                #     prediction_boxes_3d[swapped_indices, 3]

        ##############################
        # Ground Truth
        ##############################

        # Get ground truth labels
        if dataset.has_labels:
            gt_objects = obj_utils.read_labels(dataset.label_dir, img_idx)
        else:
            gt_objects = []

        # Filter objects to desired difficulty
        filtered_gt_objs = dataset.kitti_utils.filter_labels(
            gt_objects, classes=gt_classes)

        boxes2d, _, _ = obj_utils.build_bbs_from_objects(
            filtered_gt_objs, class_needed=gt_classes)

        image_path = dataset.get_rgb_image_path(sample_name)
        image = Image.open(image_path)
        image_size = image.size

        # Read the stereo calibration matrix for visualization
        stereo_calib = calib_utils.read_calibration(dataset.calib_dir,
                                                    img_idx)
        calib_p2 = stereo_calib.p2

        ##############################
        # Reformat and prepare to draw
        ##############################
        if draw_proposals_separate or draw_overlaid:
            proposals_as_anchors = box_3d_encoder.box_3d_to_anchor(
                proposal_boxes_3d)

            proposal_boxes, _ = anchor_projector.project_to_image_space(
                    proposals_as_anchors, calib_p2, image_size)

            num_of_proposals = proposal_boxes_3d.shape[0]

            prop_fig, prop_2d_axes, prop_3d_axes = \
                vis_utils.visualization(dataset.rgb_image_dir,
                                        img_idx,
                                        display=False)

            draw_proposals(filtered_gt_objs,
                           calib_p2,
                           num_of_proposals,
                           proposal_objs,
                           proposal_boxes,
                           prop_2d_axes,
                           prop_3d_axes,
                           draw_orientations_on_prop)

            if draw_proposals_separate:
                # Save just the proposals
                filename = prop_out_dir + '/' + sample_name + '.png'
                plt.savefig(filename)

                if not draw_overlaid:
                    plt.close(prop_fig)

        if draw_overlaid or draw_predictions_separate:
            if len(prediction_boxes_3d) > 0:
                # Project the 3D box predictions to image space
                image_filter = []
                final_boxes_2d = []
                for i in range(len(prediction_boxes_3d)):
                    box_3d = prediction_boxes_3d[i, 0:7]
                    img_box = box_3d_projector.project_to_image_space(
                        box_3d, calib_p2,
                        truncate=True, image_size=image_size,
                        discard_before_truncation=False)
                    if img_box is not None:
                        image_filter.append(True)
                        final_boxes_2d.append(img_box)
                    else:
                        image_filter.append(False)
                final_boxes_2d = np.asarray(final_boxes_2d)
                final_prediction_boxes_3d = prediction_boxes_3d[image_filter]
                final_scores = prediction_scores[image_filter]
                final_class_indices = prediction_class_indices[image_filter]

                num_of_predictions = final_boxes_2d.shape[0]

                # Convert to objs
                final_prediction_objs = \
                    [box_3d_encoder.box_3d_to_object_label(
                        prediction, obj_type='Prediction')
                        for prediction in final_prediction_boxes_3d]
                for (obj, score) in zip(final_prediction_objs, final_scores):
                    obj.score = score
            else:
                if save_empty_images:
                    pred_fig, pred_2d_axes, pred_3d_axes = \
                        vis_utils.visualization(dataset.rgb_image_dir,
                                                img_idx,
                                                display=False,
                                                fig_size=fig_size)
                    filename = pred_out_dir + '/' + sample_name + '.png'
                    plt.savefig(filename)
                    plt.close(pred_fig)
                continue

            if draw_overlaid:
                # Overlay prediction boxes on image
                draw_predictions(filtered_gt_objs,
                                 calib_p2,
                                 num_of_predictions,
                                 final_prediction_objs,
                                 final_class_indices,
                                 final_boxes_2d,
                                 prop_2d_axes,
                                 prop_3d_axes,
                                 draw_score,
                                 draw_iou,
                                 gt_classes,
                                 draw_orientations_on_pred)
                filename = overlaid_out_dir + '/' + sample_name + '.png'
                plt.savefig(filename)

                plt.close(prop_fig)

            if draw_predictions_separate:
                # Now only draw prediction boxes on images
                # on a new figure handler
                if draw_projected_2d_boxes:
                    pred_fig, pred_2d_axes, pred_3d_axes = \
                        vis_utils.visualization(dataset.rgb_image_dir,
                                                img_idx,
                                                display=False,
                                                fig_size=fig_size)

                    draw_predictions(filtered_gt_objs,
                                     calib_p2,
                                     num_of_predictions,
                                     final_prediction_objs,
                                     final_class_indices,
                                     final_boxes_2d,
                                     pred_2d_axes,
                                     pred_3d_axes,
                                     draw_score,
                                     draw_iou,
                                     gt_classes,
                                     draw_orientations_on_pred)
                else:
                    pred_fig, pred_3d_axes = \
                        vis_utils.visualize_single_plot(
                            dataset.rgb_image_dir, img_idx, display=False)

                    draw_3d_predictions(filtered_gt_objs,
                                        calib_p2,
                                        num_of_predictions,
                                        final_prediction_objs,
                                        final_class_indices,
                                        final_boxes_2d,
                                        pred_3d_axes,
                                        draw_score,
                                        draw_iou,
                                        gt_classes,
                                        draw_orientations_on_pred)
                filename = pred_out_dir + '/' + sample_name + '.png'
                plt.savefig(filename)
                plt.close(pred_fig)

    print('\nDone')


def draw_proposals(filtered_gt_objs,
                   p_matrix,
                   num_of_proposals,
                   proposal_objs,
                   proposal_boxes,
                   prop_2d_axes,
                   prop_3d_axes,
                   draw_orientations_on_prop):
    # Draw filtered ground truth boxes
    for obj in filtered_gt_objs:
        # Draw 2D boxes
        vis_utils.draw_box_2d(
            prop_2d_axes, obj, test_mode=True, color_tm='r')

        # Draw 3D boxes
        vis_utils.draw_box_3d(prop_3d_axes, obj, p_matrix,
                              show_orientation=draw_orientations_on_prop,
                              color_table=['r', 'y', 'r', 'w'],
                              line_width=2,
                              double_line=False)

    # Overlay proposal boxes on images
    for anchor_idx in range(num_of_proposals):
        obj_label = proposal_objs[anchor_idx]

        # Draw 2D boxes (can't use obj_label since 2D corners are not
        # filled in)
        rgb_box_2d = proposal_boxes[anchor_idx]

        box_x1 = rgb_box_2d[0]
        box_y1 = rgb_box_2d[1]
        box_w = rgb_box_2d[2] - box_x1
        box_h = rgb_box_2d[3] - box_y1

        rect = patches.Rectangle((box_x1, box_y1),
                                 box_w, box_h,
                                 linewidth=2,
                                 edgecolor='cornflowerblue',
                                 facecolor='none')

        prop_2d_axes.add_patch(rect)

        # Draw 3D boxes
        vis_utils.draw_box_3d(prop_3d_axes, obj_label, p_matrix,
                              show_orientation=draw_orientations_on_prop,
                              color_table=[
                                  'cornflowerblue', 'y', 'r', 'w'],
                              line_width=2,
                              double_line=False)


def draw_predictions(filtered_gt_objs,
                     p_matrix,
                     predictions_to_show,
                     prediction_objs,
                     prediction_class,
                     final_boxes,
                     pred_2d_axes,
                     pred_3d_axes,
                     draw_score,
                     draw_iou,
                     gt_classes,
                     draw_orientations_on_pred):
    # Draw filtered ground truth boxes
    gt_boxes = []
    for obj in filtered_gt_objs:
        # Draw 2D boxes
        vis_utils.draw_box_2d(
            pred_2d_axes, obj, test_mode=True, color_tm='r')

        # Draw 3D boxes
        vis_utils.draw_box_3d(pred_3d_axes, obj, p_matrix,
                              show_orientation=draw_orientations_on_pred,
                              color_table=['r', 'y', 'r', 'w'],
                              line_width=2,
                              double_line=False)
        if draw_iou:
            gt_box_2d = [obj.x1, obj.y1, obj.x2, obj.y2]
            gt_boxes.append(gt_box_2d)

    if gt_boxes:
        # the two_2 eval function expects np.array
        gt_boxes = np.asarray(gt_boxes)

    for pred_idx in range(predictions_to_show):
        pred_obj = prediction_objs[pred_idx]
        pred_class_idx = prediction_class[pred_idx]

        rgb_box_2d = final_boxes[pred_idx]

        box_x1 = rgb_box_2d[0]
        box_y1 = rgb_box_2d[1]
        box_w = rgb_box_2d[2] - box_x1
        box_h = rgb_box_2d[3] - box_y1

        box_cls = gt_classes[int(pred_class_idx)]
        rect = patches.Rectangle((box_x1, box_y1),
                                 box_w, box_h,
                                 linewidth=2,
                                 edgecolor=BOX_COLOUR_SCHEME[box_cls],
                                 facecolor='none')

        pred_2d_axes.add_patch(rect)

        # Draw 3D boxes
        vis_utils.draw_box_3d(pred_3d_axes, pred_obj, p_matrix,
                              show_orientation=draw_orientations_on_pred,
                              color_table=['#00FF00', 'y', 'r', 'w'],
                              line_width=2,
                              double_line=False,
                              box_color=BOX_COLOUR_SCHEME[box_cls])

        if draw_score or draw_iou:
            box_x2 = rgb_box_2d[2]
            box_y2 = rgb_box_2d[3]

            pred_box_2d = [box_x1,
                           box_y1,
                           box_x2,
                           box_y2]

            info_text_x = (box_x1 + box_x2) / 2
            info_text_y = box_y1

            draw_prediction_info(pred_2d_axes,
                                 info_text_x,
                                 info_text_y,
                                 pred_obj,
                                 pred_class_idx,
                                 pred_box_2d,
                                 gt_boxes,
                                 draw_score,
                                 draw_iou,
                                 gt_classes)


def draw_3d_predictions(filtered_gt_objs,
                        p_matrix,
                        predictions_to_show,
                        prediction_objs,
                        prediction_class,
                        final_boxes,
                        pred_3d_axes,
                        draw_score,
                        draw_iou,
                        gt_classes,
                        draw_orientations_on_pred):
    # Draw filtered ground truth boxes
    gt_boxes = []
    for obj in filtered_gt_objs:
        # Draw 3D boxes
        vis_utils.draw_box_3d(pred_3d_axes, obj, p_matrix,
                              show_orientation=draw_orientations_on_pred,
                              color_table=['r', 'y', 'r', 'w'],
                              line_width=2,
                              double_line=False)
        if draw_iou:
            gt_box_2d = [obj.x1, obj.y1, obj.x2, obj.y2]
            gt_boxes.append(gt_box_2d)

    if gt_boxes:
        # the two_2 eval function expects np.array
        gt_boxes = np.asarray(gt_boxes)

    for pred_idx in range(predictions_to_show):
        pred_obj = prediction_objs[pred_idx]
        pred_class_idx = prediction_class[pred_idx]

        rgb_box_2d = final_boxes[pred_idx]

        box_x1 = rgb_box_2d[0]
        box_y1 = rgb_box_2d[1]

        # Draw 3D boxes
        box_cls = gt_classes[int(pred_class_idx)]
        vis_utils.draw_box_3d(pred_3d_axes, pred_obj, p_matrix,
                              show_orientation=draw_orientations_on_pred,
                              color_table=['#00FF00', 'y', 'r', 'w'],
                              line_width=2,
                              double_line=False,
                              box_color=BOX_COLOUR_SCHEME[box_cls])

        if draw_score or draw_iou:
            box_x2 = rgb_box_2d[2]
            box_y2 = rgb_box_2d[3]

            pred_box_2d = [box_x1,
                           box_y1,
                           box_x2,
                           box_y2]

            info_text_x = (box_x1 + box_x2) / 2
            info_text_y = box_y1
            draw_prediction_info(pred_3d_axes,
                                 info_text_x,
                                 info_text_y,
                                 pred_obj,
                                 pred_class_idx,
                                 pred_box_2d,
                                 gt_boxes,
                                 draw_score,
                                 draw_iou,
                                 gt_classes)


def draw_prediction_info(ax, x, y,
                         pred_obj,
                         pred_class_idx,
                         pred_box_2d,
                         ground_truth,
                         draw_score,
                         draw_iou,
                         gt_classes):

    label = ""

    if draw_score:
        label += "{:.2f}".format(pred_obj.score)

    if draw_iou and len(ground_truth) > 0:
        if draw_score:
            label += ', '
        iou = evaluation.two_d_iou(pred_box_2d, ground_truth)
        label += "{:.3f}".format(max(iou))

    box_cls = gt_classes[int(pred_class_idx)]

    ax.text(x, y - 4,
            gt_classes[int(pred_class_idx)] + '\n' + label,
            verticalalignment='bottom',
            horizontalalignment='center',
            color=BOX_COLOUR_SCHEME[box_cls],
            fontsize=10,
            fontweight='bold',
            path_effects=[
                patheffects.withStroke(linewidth=2,
                                       foreground='black')])


if __name__ == '__main__':
    main()

import os

import numpy as np
import tensorflow as tf

import avod

from avod.core.mini_batch_preprocessor import MiniBatchPreprocessor
from avod.core.minibatch_samplers import balanced_positive_negative_sampler


class MiniBatchUtils:
    def __init__(self, dataset):

        self._dataset = dataset

        self._mini_batch_sampler = \
            balanced_positive_negative_sampler.BalancedPositiveNegativeSampler()

        ##############################
        # Parse KittiUtils config
        ##############################
        self.kitti_utils_config = dataset.config.kitti_utils_config
        self._area_extents = self.kitti_utils_config.area_extents
        self._anchor_strides = np.reshape(
            self.kitti_utils_config.anchor_strides, (-1, 2))

        ##############################
        # Parse MiniBatchUtils config
        ##############################
        self.config = self.kitti_utils_config.mini_batch_config
        self._density_threshold = self.config.density_threshold

        # RPN mini batches
        rpn_config = self.config.rpn_config

        rpn_iou_type = rpn_config.WhichOneof('iou_type')
        if rpn_iou_type == 'iou_2d_thresholds':
            self.rpn_iou_type = '2d'
            self.rpn_iou_thresholds = rpn_config.iou_2d_thresholds

        elif rpn_iou_type == 'iou_3d_thresholds':
            self.rpn_iou_type = '3d'
            self.rpn_iou_thresholds = rpn_config.iou_3d_thresholds

        self.rpn_neg_iou_range = [self.rpn_iou_thresholds.neg_iou_lo,
                                  self.rpn_iou_thresholds.neg_iou_hi]
        self.rpn_pos_iou_range = [self.rpn_iou_thresholds.pos_iou_lo,
                                  self.rpn_iou_thresholds.pos_iou_hi]

        self.rpn_mini_batch_size = rpn_config.mini_batch_size

        # AVOD mini batches
        avod_config = self.config.avod_config
        self.avod_iou_type = '2d'
        self.avod_iou_thresholds = avod_config.iou_2d_thresholds

        self.avod_neg_iou_range = [self.avod_iou_thresholds.neg_iou_lo,
                                   self.avod_iou_thresholds.neg_iou_hi]
        self.avod_pos_iou_range = [self.avod_iou_thresholds.pos_iou_lo,
                                   self.avod_iou_thresholds.pos_iou_hi]

        self.avod_mini_batch_size = avod_config.mini_batch_size

        # Setup paths
        self.mini_batch_dir = avod.root_dir() + '/data/mini_batches/' + \
            'iou_{}/'.format(self.rpn_iou_type) + \
            dataset.name + '/' + dataset.cluster_split + '/' + \
            dataset.bev_source

        # Array column indices for saving to files
        self.col_length = 9
        self.col_anchor_indices = 0
        self.col_ious = 1
        self.col_offsets_lo = 2
        self.col_offsets_hi = 8
        self.col_class_idx = 8

    def preprocess_rpn_mini_batches(self, indices):
        """Generates rpn mini batch info for the kitti dataset

            Preprocesses data and saves data to files.
            Each file contains information that is used to feed
            to the network for RPN training.
        """

        clusters, _ = self._dataset.get_cluster_info()

        mini_batch_preprocessor = \
            MiniBatchPreprocessor(self._dataset,
                                  self.mini_batch_dir,
                                  self._anchor_strides,
                                  self._density_threshold,
                                  self.rpn_neg_iou_range,
                                  self.rpn_pos_iou_range)

        mini_batch_preprocessor.preprocess(indices)

    def get_file_path(self, classes_name, anchor_strides, sample_name):
        """Gets the full file path to the anchors info

        Args:
            classes_name: name of classes ('Car', 'Pedestrian', 'Cyclist',
                'People')
            anchor_strides: anchor strides
            sample_name: sample name, e.g. '000123'

        Returns:
            The anchors info file path. Returns the folder if
                sample_name is None
        """
        # Round values for nicer folder names
        anchor_strides = np.round(anchor_strides[:, 0], 3)

        anchor_strides_str = \
            ' '.join(str(stride) for stride in anchor_strides)

        if sample_name:
            return self.mini_batch_dir + '/' + classes_name + \
                '[ ' + anchor_strides_str + ']/' + \
                sample_name + ".npy"

        return self.mini_batch_dir + '/' + classes_name + \
            '[ ' + anchor_strides_str + ']'

    def get_anchors_info(self, classes_name, anchor_strides, sample_name):
        """Reads in the file containing the information matrix

        Args:
            classes_name: object type, one of ('Car', 'Pedestrian',
                'Cyclist', 'People')
            anchor_strides: anchor strides
            sample_name: image name to read the corresponding file

        Returns:
            anchor_ious: max iou of the anchor with any ground truth
            anchor_offsets: encoded anchor offsets to the matching ground truth
            anchor_classes: class index of the anchor
                (e.g. 0 or 1, for "Background" or "Car")

            [] if the file contains an empty array
        """
        file_name = self.get_file_path(classes_name, anchor_strides,
                                       sample_name)

        if not os.path.exists(file_name):
            raise FileNotFoundError(
                "{} not found for sample {} in {}, "
                "run the preprocessing script first".format(
                    file_name,
                    sample_name,
                    self.mini_batch_dir))

        anchors_info = np.load(file_name)
        if anchors_info.any():
            return self._parse_anchors_info(anchors_info)
        return []

    def sample_mini_batch(self,
                          max_ious,
                          mini_batch_size,
                          negative_iou_range,
                          positive_iou_range):
        """
        Samples a mini batch based on anchor ious with ground truth

        Args:
            max_ious: a tensor of max ious with ground truth in
                the shape (N,)
            mini_batch_size: size of the mini batch to return
            negative_iou_range: iou range to consider an anchor as negative
            positive_iou_range: iou range to consider an anchor as positive

        Returns:
            mb_sampled: a boolean mask where True indicates anchors sampled
                for the mini batch
            mb_pos_sampled: a boolean mask where True indicates positive anchors
        """

        bkg_and_neg_labels = tf.less(max_ious, negative_iou_range[1])
        pos_labels = tf.greater(max_ious, positive_iou_range[0])
        indicator = tf.logical_or(pos_labels, bkg_and_neg_labels)

        if negative_iou_range[0] > 0.0:
            # If neg_iou_lo is > 0.0, the mini batch may be empty.
            # In that case, use all background and negative labels
            neg_labels = tf.logical_and(
                bkg_and_neg_labels,
                tf.greater_equal(max_ious, negative_iou_range[0]))

            new_indicator = tf.logical_or(pos_labels, neg_labels)

            num_valid = tf.reduce_sum(tf.cast(indicator, tf.int32))
            indicator = tf.cond(
                tf.greater(num_valid, 0),
                true_fn=lambda: tf.identity(new_indicator),
                false_fn=lambda: tf.identity(bkg_and_neg_labels))

        sampler = self._mini_batch_sampler
        mb_sampled, mb_pos_sampled = sampler.subsample(
            indicator, mini_batch_size, pos_labels)

        return mb_sampled, mb_pos_sampled

    def sample_rpn_mini_batch(self, anchor_ious):
        """ Samples a mini batch to train the RPN with preconfigured
            mini batch size and 3D iou ranges

        Args:
            anchor_ious: a tensor of max ious with ground truth in
                the shape (N,)

        Returns:
            mb_sampled: a boolean mask where True indicates anchors sampled
                for the mini batch
            mb_pos_sampled: a boolean mask where True indicates positive anchors
        """
        return self.sample_mini_batch(anchor_ious,
                                      self.rpn_mini_batch_size,
                                      self.rpn_neg_iou_range,
                                      self.rpn_pos_iou_range)

    def sample_avod_mini_batch(self, anchor_ious):
        """ Samples a mini batch to train AVOD with preconfigured
            mini batch size and 2D iou ranges

        Args:
            anchor_ious: a tensor of max ious with ground truth in
                the shape (N,)

        Returns:
            mb_sampled: a boolean mask where True indicates anchors sampled
                for the mini batch
            mb_pos_sampled: a boolean mask where True indicates positive anchors
        """
        return self.sample_mini_batch(anchor_ious,
                                      self.avod_mini_batch_size,
                                      self.avod_neg_iou_range,
                                      self.avod_pos_iou_range)

    def _parse_anchors_info(self, anchors_info):
        """
        Parses anchor indices, offsets, and classes from a matrix

        Args:
            anchors_info: an np.ndarray in the form
                N x [indices, (offsets), class_index]

        Returns:
            anchor_indices: indices of anchors to use after generation
            anchor_ious: max iou of the anchor with any ground truth
            anchor_offsets: encoded anchor offsets to the matching ground truth
            anchor_classes: class index of the anchor
                (e.g. 0 or 1, for "Background" or "Car")
        """
        anchor_indices = np.asarray(
            anchors_info[:, self.col_anchor_indices], dtype=np.int32)

        anchor_ious = np.asarray(
            anchors_info[:, self.col_ious], dtype=np.float32)

        anchor_offsets = np.asarray(
            anchors_info[:, self.col_offsets_lo:self.col_offsets_hi],
            dtype=np.float32)
        anchor_classes = np.asarray(
            anchors_info[:, self.col_class_idx], dtype=np.float32)

        return anchor_indices, anchor_ious, \
            anchor_offsets, anchor_classes

    def mask_class_label_indices(self,
                                 mb_pos_mask,
                                 mb_mask,
                                 max_iou_indices,
                                 class_indices):
        """
        Samples a mini batch based on anchor ious with ground truth

        Args:
            mb_pos_mask: a boolean tensor mask of size [N] of positive anchors
                in the mini-batch
            mb_mask: a boolean tensor mask of size [N] of all anchors in the
                mini-batch.
            max_iou_indices: a tensor of shape [N] indicating the indices
                corresponding to the maximum IoU between predicted anchors and
                the ground truth anchors.
            class_indices: a tensor of shape [num_of_classes] indicating the
                class labels as indices. For instance indices=[0, 1, 2, 3]
                indicating 'background, car, pedestrian, cyclist' etc.

        Returns:
            masked_class_indices: a tensor of boolean mask for class label
                indices. This gives the indices for the positive classes and
                masks negatives or background classes by zero's.
        """

        # mask the indices by the all_mask which is the mini_batch mask
        masked_argmax = tf.boolean_mask(max_iou_indices,
                                        mb_mask)

        # get the corresponding class indices that had high IoUs
        masked_labels = tf.gather(class_indices,
                                  masked_argmax)

        # mask the positives by the total mask again
        # this gives us the 'True' entries
        mask_pos_mask = tf.boolean_mask(mb_pos_mask,
                                        mb_mask)

        # multiply the masked label entries by this positives only
        # this will keep the positive class labels and sets everything else
        # to zero ('Background' class).
        # cast these to int as the class labels are in floats
        mb_class_indices = tf.multiply(tf.cast(masked_labels, tf.int32),
                                       tf.cast(mask_pos_mask, tf.int32))

        return mb_class_indices

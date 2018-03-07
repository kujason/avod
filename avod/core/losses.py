"""Classification and regression loss functions for object detection.

Localization losses:
 * WeightedL2LocalizationLoss
 * WeightedSmoothL1LocalizationLoss

Classification losses:
 * WeightedSoftmaxClassificationLoss
 * WeightedSigmoidClassificationLoss
"""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from avod.core import ops


class Loss(object):
    """Abstract base class for loss functions."""
    __metaclass__ = ABCMeta

    def __call__(self,
                 prediction_tensor,
                 target_tensor,
                 ignore_nan_targets=False,
                 scope=None,
                 **params):
        """Call the loss function.

        Args:
            prediction_tensor: a tensor representing predicted quantities.
            target_tensor: a tensor representing regression or classification
                           targets.
            ignore_nan_targets: whether to ignore nan targets in the loss
                                computation. E.g. can be used if the target
                                tensor is missing groundtruth data that
                                shouldn't be factored into the loss.
            scope: Op scope name. Defaults to 'Loss' if None.
            **params: Additional keyword arguments for specific implementations
                     of the Loss.
        Returns:
            loss: a tensor representing the value of the loss function.
        """
        with tf.name_scope(scope, 'Loss',
                           [prediction_tensor, target_tensor, params]) as scope:
            if ignore_nan_targets:
                target_tensor = tf.where(tf.is_nan(target_tensor),
                                         prediction_tensor,
                                         target_tensor)
            return self._compute_loss(
                prediction_tensor, target_tensor, **params)

    @abstractmethod
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """Method to be overriden by implementations.

        Args:
            prediction_tensor: a tensor representing predicted quantities
            target_tensor: a tensor representing regression or classification
                           targets
            **params: Additional keyword arguments for specific implementations
                      of the Loss.
        Returns:
            loss: a tensor representing the value of the loss function
        """
        pass


class WeightedL2LocalizationLoss(Loss):
    """L2 localization loss function with anchorwise output support.

       Loss[b,a] = .5 * ||weights[b,a] * (prediction[b,a,:] - target[b,a,:])||^2
    """

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.
        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
                             code_size] representing the (encoded) predicted
                             locations of objects.
          target_tensor: A float tensor of shape [batch_size, num_anchors,
                         code_size] representing the regression targets
          weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
          loss: a (scalar) tensor representing the value of the loss function
                or a float tensor of shape [batch_size, num_anchors]
        """
        weighted_diff = (prediction_tensor - target_tensor) * tf.expand_dims(
            weights, 2)
        square_diff = 0.5 * tf.square(weighted_diff)
        return tf.reduce_sum(square_diff)


class WeightedSigmoidClassificationLoss(Loss):
    """Sigmoid cross entropy classification loss function."""

    def _compute_loss(self,
                      prediction_tensor,
                      target_tensor,
                      weights,
                      class_indices=None):
        """Compute loss function.
        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.
        Returns:
            loss: a (scalar) tensor representing the value of the loss function
                or a float tensor of shape [batch_size, num_anchors]
        """
        weights = tf.expand_dims(weights, 2)
        if class_indices is not None:
            weights *= tf.reshape(
                ops.indices_to_dense_vector(class_indices,
                                            tf.shape(prediction_tensor)[2]),
                [1, 1, -1])
        per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor))
        return tf.reduce_sum(per_entry_cross_ent * weights)


class WeightedSmoothL1Loss(Loss):
    """Smooth L1 localization loss function.
    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.
    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def _compute_loss(self, prediction_tensor, target_tensor, weight):
        """Compute loss function.
        Args:
            prediction_tensor: A float tensor of shape [num_anchors,
                code_size] representing the (encoded) predicted
                locations of objects.
            target_tensor: A float tensor of shape [num_anchors,
                code_size] representing the regression targets
        Returns:
          loss: an anchorwise tensor of shape [num_anchors] representing
            the value of the loss function
        """
        diff = prediction_tensor - target_tensor
        abs_diff = tf.abs(diff)
        abs_diff_lt_1 = tf.less(abs_diff, 1)

        anchorwise_smooth_l1norm = tf.reduce_sum(
            tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
            axis=1) * weight
        return anchorwise_smooth_l1norm


class WeightedSoftmaxLoss(Loss):
    """Softmax cross-entropy loss function."""

    def _compute_loss(self, prediction_tensor, target_tensor, weight):
        """Compute loss function.
        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
        Returns:
          loss: a (scalar) tensor representing the value of the loss function
        """
        num_classes = prediction_tensor.get_shape().as_list()[-1]
        per_row_cross_ent = (tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(target_tensor, [-1, num_classes]),
            logits=tf.reshape(prediction_tensor, [-1, num_classes])))

        return tf.reduce_sum(per_row_cross_ent) * weight

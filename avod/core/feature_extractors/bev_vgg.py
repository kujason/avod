"""Contains modified VGG model definition to extract features from
Bird's eye view input.

Usage:
    outputs, end_points = BevVgg(inputs, layers_config)
"""

import tensorflow as tf

from avod.core.feature_extractors import bev_feature_extractor

slim = tf.contrib.slim


class BevVgg(bev_feature_extractor.BevFeatureExtractor):

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def build(self,
              inputs,
              input_pixel_size,
              is_training,
              scope='bev_vgg'):
        """ Modified VGG for BEV feature extraction

        Note: All the fully_connected layers have been transformed to conv2d
              layers and are implemented in the main model.

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False fo validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        vgg_config = self.config

        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, 'bev_vgg', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                    net = slim.repeat(inputs,
                                      vgg_config.vgg_conv1[0],
                                      slim.conv2d,
                                      vgg_config.vgg_conv1[1],
                                      [3, 3],
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params={
                                          'is_training': is_training},
                                      scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net,
                                      vgg_config.vgg_conv2[0],
                                      slim.conv2d,
                                      vgg_config.vgg_conv2[1],
                                      [3, 3],
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params={
                                          'is_training': is_training},
                                      scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net,
                                      vgg_config.vgg_conv3[0],
                                      slim.conv2d,
                                      vgg_config.vgg_conv3[1],
                                      [3, 3],
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params={
                                          'is_training': is_training},
                                      scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net,
                                      vgg_config.vgg_conv4[0],
                                      slim.conv2d,
                                      vgg_config.vgg_conv4[1],
                                      [3, 3],
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params={
                                          'is_training': is_training},
                                      scope='conv4')

                with tf.variable_scope('upsampling'):
                    # This extractor downsamples the input by a factor
                    # of 8 (3 maxpool layers)
                    downsampling_factor = 8
                    downsampled_shape = input_pixel_size / downsampling_factor

                    upsampled_shape = \
                        downsampled_shape * vgg_config.upsampling_multiplier

                    feature_maps_out = tf.image.resize_bilinear(
                        net, upsampled_shape)

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points


class BevVggClassification(bev_feature_extractor.BevFeatureExtractor):
    """This is used in vgg unit tests."""

    def __init__(self):
        super(BevVggClassification, self).__init__(None)

    def build(self,
              inputs,
              num_classes=1000,
              is_training=True,
              dropout_keep_prob=0.5,
              spatial_squeeze=True,
              scope='vgg'):
        """VGG 11-Layers modified version.

        Note: All the fully_connected layers have been transformed to conv2d
              layers.

        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes.
          is_training: whether or not the model is being trained.
          dropout_keep_prob: the probability that activations are kept in the
                             dropout layers during training.
          spatial_squeeze: whether or not should squeeze the spatial dimensions
                           of the outputs. Useful to remove unnecessary
                           dimensions for classification.
          scope: Optional scope for the variables.


        Returns:
          the last op containing the log predictions and end_points dict.
        """
        with tf.variable_scope(scope, 'bev_vgg', [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                outputs_collections=end_points_collection):

                net = slim.repeat(
                    inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(
                    net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(
                    net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(
                    net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(
                    net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                # Change the filter depending on the input dim
                net = slim.conv2d(net, 4096, [9, 9],
                    padding='VALID', scope='fc6')
                net = slim.dropout(net, dropout_keep_prob,
                    is_training=is_training, scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, dropout_keep_prob,
                    is_training=is_training, scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='fc8')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net, end_points

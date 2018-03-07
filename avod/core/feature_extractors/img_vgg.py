"""Contains modified VGG model definition to extract features from
RGB image input.

Usage:
    outputs, end_points = ImgVgg(inputs, layers_config)
"""
import tensorflow as tf

from avod.core.feature_extractors import img_feature_extractor

slim = tf.contrib.slim


class ImgVgg(img_feature_extractor.ImgFeatureExtractor):

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
              scope='img_vgg'):
        """ Modified VGG for image feature extraction.

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
            with tf.variable_scope(scope, 'img_vgg', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d,
                                     slim.fully_connected,
                                     slim.max_pool2d],
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
                    downsampled_shape = \
                        input_pixel_size / downsampling_factor

                    upsampled_shape = \
                        downsampled_shape * vgg_config.upsampling_multiplier

                    feature_maps_out = tf.image.resize_bilinear(
                        net, upsampled_shape)

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points

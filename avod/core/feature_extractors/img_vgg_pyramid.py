import tensorflow as tf

from avod.core.feature_extractors import img_feature_extractor

slim = tf.contrib.slim


class ImgVggPyr(img_feature_extractor.ImgFeatureExtractor):
    """Modified VGG model definition to extract features from
    RGB image input using pyramid features.
    """

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
              scope='img_vgg_pyr'):
        """ Modified VGG for image feature extraction with pyramid features.

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        vgg_config = self.config

        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, 'img_vgg_pyr', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d,
                                     slim.fully_connected,
                                     slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    # Encoder
                    conv1 = slim.repeat(inputs,
                                        vgg_config.vgg_conv1[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv1[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv1')
                    pool1 = slim.max_pool2d(
                        conv1, [2, 2], scope='pool1')

                    conv2 = slim.repeat(pool1,
                                        vgg_config.vgg_conv2[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv2[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv2')
                    pool2 = slim.max_pool2d(
                        conv2, [2, 2], scope='pool2')

                    conv3 = slim.repeat(pool2,
                                        vgg_config.vgg_conv3[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv3[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv3')
                    pool3 = slim.max_pool2d(
                        conv3, [2, 2], scope='pool3')

                    conv4 = slim.repeat(pool3,
                                        vgg_config.vgg_conv4[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv4[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv4')

                    # Decoder (upsample and fuse features)
                    upconv3 = slim.conv2d_transpose(
                        conv4,
                        vgg_config.vgg_conv3[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv3')

                    concat3 = tf.concat(
                        (conv3, upconv3), axis=3, name='concat3')
                    pyramid_fusion3 = slim.conv2d(
                        concat3,
                        vgg_config.vgg_conv2[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion3')

                    upconv2 = slim.conv2d_transpose(
                        pyramid_fusion3,
                        vgg_config.vgg_conv2[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv2')

                    concat2 = tf.concat(
                        (conv2, upconv2), axis=3, name='concat2')
                    pyramid_fusion_2 = slim.conv2d(
                        concat2,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion2')

                    upconv1 = slim.conv2d_transpose(
                        pyramid_fusion_2,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv1')

                    concat1 = tf.concat(
                        (conv1, upconv1), axis=3, name='concat1')
                    pyramid_fusion1 = slim.conv2d(
                        concat1,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion1')

                feature_maps_out = pyramid_fusion1

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points

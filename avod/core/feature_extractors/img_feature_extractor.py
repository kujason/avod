from abc import abstractmethod

import tensorflow as tf


class ImgFeatureExtractor:

    # Kitti image mean per channel
    _R_MEAN = 92.8403
    _G_MEAN = 97.7996
    _B_MEAN = 93.5843

    def __init__(self, extractor_config):
        self.config = extractor_config

    def preprocess_input(self, tensor_in, output_size):
        """Preprocesses the given input.

        Args:
            tensor_in: A `Tensor` of shape=(batch_size, height,
                width, channels) representing an input image.
            output_size: The size of the input (H x W)

        Returns:
            Preprocessed tensor input, resized to the output_size
        """
        image = tf.image.resize_images(tensor_in, output_size)
        image = tf.squeeze(image)
        image = tf.to_float(image)
        image_normalized = self._mean_image_subtraction(image,
                                                        [self._R_MEAN,
                                                         self._G_MEAN,
                                                         self._B_MEAN])
        tensor_out = tf.expand_dims(image_normalized, axis=0)
        return tensor_out

    def _mean_image_subtraction(self, image, means):
        """Subtracts the given means from each image channel.

        For example:
        means = [123.68, 116.779, 103.939]
        image = _mean_image_subtraction(image, means)

        Note that the rank of `image` must be known.

        Args:
        image: a tensor of size [height, width, C].
        means: a C-vector of values to subtract from each channel.

        Returns:
        the centered image.

        Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank
            other than three or if the number of channels in `image` doesn't
            match the number of values in `means`.
        """
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        channels = tf.split(
            axis=2,
            num_or_size_splits=num_channels,
            value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=2, values=channels)

    @abstractmethod
    def build(self, **kwargs):
        pass

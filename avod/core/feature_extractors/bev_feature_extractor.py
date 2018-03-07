from abc import abstractmethod

import tensorflow as tf

class BevFeatureExtractor:

    def __init__(self, extractor_config):
        self.config = extractor_config

    def preprocess_input(self, tensor_in, output_shape):
        """Preprocesses the given input.

        Args:
            tensor_in: A `Tensor` of shape=(batch_size, height,
                width, channel) representing an input image.
            output_shape: The shape of the output (H x W)

        Returns:
            Preprocessed tensor input, resized to the output_size
        """

        # Only reshape if input shape does not match
        if not tensor_in.shape[1:3] == output_shape:
            return tf.image.resize_images(tensor_in, output_shape)

        return tensor_in

    @abstractmethod
    def build(self, **kwargs):
        pass

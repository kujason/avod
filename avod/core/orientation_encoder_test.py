import numpy as np
import tensorflow as tf

from avod.core import orientation_encoder


class OrientationEncoderTest(tf.test.TestCase):
    def test_tf_orientation_to_angle_vector(self):
        # Test conversion for angles between [-pi, pi] with 0.5 degree steps
        np_orientations = np.arange(-np.pi, np.pi, np.pi / 360.0)

        expected_angle_vectors = np.stack([np.cos(np_orientations),
                                           np.sin(np_orientations)], axis=1)

        # Convert to tensors and convert to angle unit vectors
        tf_orientations = tf.convert_to_tensor(np_orientations)
        tf_angle_vectors = orientation_encoder.tf_orientation_to_angle_vector(
            tf_orientations)

        with self.test_session() as sess:
            angle_vectors_out = sess.run(tf_angle_vectors)

            np.testing.assert_allclose(angle_vectors_out,
                                       expected_angle_vectors)

    def test_angle_vectors_to_orientation(self):
        # Test conversion for angles between [-pi, pi] with 0.5 degree steps
        np_angle_vectors = \
            np.asarray([[np.cos(angle), np.sin(angle)]
                        for angle in np.arange(-np.pi, np.pi, np.pi / 360.0)])

        # Check that tf output matches numpy's arctan2 output
        expected_orientations = np.arctan2(np_angle_vectors[:, 1],
                                           np_angle_vectors[:, 0])

        # Convert to tensors and convert to orientation angles
        tf_angle_vectors = tf.convert_to_tensor(np_angle_vectors)
        tf_orientations = orientation_encoder.tf_angle_vector_to_orientation(
            tf_angle_vectors)

        with self.test_session() as sess:
            orientations_out = sess.run(tf_orientations)
            np.testing.assert_allclose(orientations_out,
                                       expected_orientations)

    def test_zeros_angle_vectors_to_orientation(self):
        # Test conversion for angle vectors with zeros in them
        np_angle_vectors = np.asarray(
            [[0, 0],
             [1, 0], [10, 0],
             [0, 1], [0, 10],
             [-1, 0], [-10, 0],
             [0, -1], [0, -10]])

        half_pi = np.pi / 2
        expected_orientations = [0,
                                 0, 0,
                                 half_pi, half_pi,
                                 np.pi, np.pi,
                                 -half_pi, -half_pi]

        # Convert to tensors and convert to orientation angles
        tf_angle_vectors = tf.convert_to_tensor(np_angle_vectors,
                                                dtype=tf.float64)
        tf_orientations = orientation_encoder.tf_angle_vector_to_orientation(
            tf_angle_vectors)

        with self.test_session() as sess:
            orientations_out = sess.run(tf_orientations)
            np.testing.assert_allclose(orientations_out,
                                       expected_orientations)

    def test_two_way_conversion(self):
        # Test conversion for angles between [-pi, pi] with 0.5 degree steps
        np_orientations = np.arange(np.pi, np.pi, np.pi / 360.0)

        tf_angle_vectors = orientation_encoder.tf_orientation_to_angle_vector(
            np_orientations)
        tf_orientations = orientation_encoder.tf_angle_vector_to_orientation(
            tf_angle_vectors)

        # Check that conversion from orientation -> angle vector ->
        # orientation results in the same values
        with self.test_session() as sess:
            orientations_out = sess.run(tf_orientations)
            np.testing.assert_allclose(orientations_out,
                                       np_orientations)

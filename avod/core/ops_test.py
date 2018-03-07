"""Tests for object_detection.utils.ops."""
import numpy as np
import tensorflow as tf

from avod.core import ops


class OpsTestIndicesToDenseVector(tf.test.TestCase):

    def test_indices_to_dense_vector(self):
        size = 10000
        num_indices = np.random.randint(size)
        rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

        expected_output = np.zeros(size, dtype=np.float32)
        expected_output[rand_indices] = 1.

        tf_rand_indices = tf.constant(rand_indices)
        indicator = ops.indices_to_dense_vector(tf_rand_indices, size)

        with self.test_session() as sess:
            output = sess.run(indicator)
            self.assertAllEqual(output, expected_output)
            self.assertEqual(output.dtype, expected_output.dtype)

    def test_indices_to_dense_vector_size_at_inference(self):
        size = 5000
        num_indices = 250
        all_indices = np.arange(size)
        rand_indices = np.random.permutation(all_indices)[0:num_indices]

        expected_output = np.zeros(size, dtype=np.float32)
        expected_output[rand_indices] = 1.

        tf_all_indices = tf.placeholder(tf.int32)
        tf_rand_indices = tf.constant(rand_indices)
        indicator = ops.indices_to_dense_vector(tf_rand_indices,
                                                tf.shape(tf_all_indices)[0])
        feed_dict = {tf_all_indices: all_indices}

        with self.test_session() as sess:
            output = sess.run(indicator, feed_dict=feed_dict)
            self.assertAllEqual(output, expected_output)
            self.assertEqual(output.dtype, expected_output.dtype)

    def test_indices_to_dense_vector_int(self):
        size = 500
        num_indices = 25
        rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

        expected_output = np.zeros(size, dtype=np.int64)
        expected_output[rand_indices] = 1

        tf_rand_indices = tf.constant(rand_indices)
        indicator = ops.indices_to_dense_vector(
            tf_rand_indices, size, 1, dtype=tf.int64)

        with self.test_session() as sess:
            output = sess.run(indicator)
            self.assertAllEqual(output, expected_output)
            self.assertEqual(output.dtype, expected_output.dtype)

    def test_indices_to_dense_vector_custom_values(self):
        size = 100
        num_indices = 10
        rand_indices = np.random.permutation(np.arange(size))[0:num_indices]
        indices_value = np.random.rand(1)
        default_value = np.random.rand(1)

        expected_output = np.float32(np.ones(size) * default_value)
        expected_output[rand_indices] = indices_value

        tf_rand_indices = tf.constant(rand_indices)
        indicator = ops.indices_to_dense_vector(
            tf_rand_indices,
            size,
            indices_value=indices_value,
            default_value=default_value)

        with self.test_session() as sess:
            output = sess.run(indicator)
            self.assertAllClose(output, expected_output)
            self.assertEqual(output.dtype, expected_output.dtype)

    def test_indices_to_dense_vector_all_indices_as_input(self):
        size = 500
        num_indices = 500
        rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

        expected_output = np.ones(size, dtype=np.float32)

        tf_rand_indices = tf.constant(rand_indices)
        indicator = ops.indices_to_dense_vector(tf_rand_indices, size)

        with self.test_session() as sess:
            output = sess.run(indicator)
            self.assertAllEqual(output, expected_output)
            self.assertEqual(output.dtype, expected_output.dtype)

    def test_indices_to_dense_vector_empty_indices_as_input(self):
        size = 500
        rand_indices = []

        expected_output = np.zeros(size, dtype=np.float32)

        tf_rand_indices = tf.constant(rand_indices)
        indicator = ops.indices_to_dense_vector(tf_rand_indices, size)

        with self.test_session() as sess:
            output = sess.run(indicator)
            self.assertAllEqual(output, expected_output)
            self.assertEqual(output.dtype, expected_output.dtype)


if __name__ == '__main__':
    tf.test.main()

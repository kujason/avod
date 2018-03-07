"""Tests for object detection losses module."""

import numpy as np
import tensorflow as tf

from avod.core import losses


class WeightedL2LocalizationLossTest(tf.test.TestCase):

    def testReturnsCorrectLoss(self):
        batch_size = 3
        num_anchors = 10
        code_size = 4
        prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
        target_tensor = tf.zeros([batch_size, num_anchors, code_size])
        weights = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], tf.float32)
        loss_op = losses.WeightedL2LocalizationLoss()
        loss = loss_op(prediction_tensor, target_tensor, weights=weights)

        expected_loss = (3 * 5 * 4) / 2.0
        with self.test_session() as sess:
            loss_output = sess.run(loss)
            self.assertAllClose(loss_output, expected_loss)

    def testReturnsCorrectLossSum(self):
        batch_size = 3
        num_anchors = 16
        code_size = 4
        prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
        target_tensor = tf.zeros([batch_size, num_anchors, code_size])
        weights = tf.ones([batch_size, num_anchors])
        loss_op = losses.WeightedL2LocalizationLoss()
        loss = loss_op(prediction_tensor, target_tensor, weights=weights)

        expected_loss = tf.nn.l2_loss(prediction_tensor - target_tensor)
        with self.test_session() as sess:
            loss_output = sess.run(loss)
            expected_loss_output = sess.run(expected_loss)
            self.assertAllClose(loss_output, expected_loss_output)

    def testReturnsCorrectNanLoss(self):
        batch_size = 3
        num_anchors = 10
        code_size = 4
        prediction_tensor = tf.ones([batch_size, num_anchors, code_size])
        target_tensor = tf.concat([
            tf.zeros([batch_size, num_anchors, code_size / 2]),
            tf.ones([batch_size, num_anchors, code_size / 2]) * np.nan
            ],
            axis=2)
        weights = tf.ones([batch_size, num_anchors])
        loss_op = losses.WeightedL2LocalizationLoss()
        loss = loss_op(prediction_tensor, target_tensor, weights=weights,
                       ignore_nan_targets=True)

        expected_loss = (3 * 5 * 4) / 2.0
        with self.test_session() as sess:
            loss_output = sess.run(loss)
            self.assertAllClose(loss_output, expected_loss)


if __name__ == '__main__':
    tf.test.main()

"""Testing VGG BEV network.
"""
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

import avod.tests as tests
from avod.builders import optimizer_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import constants
from avod.core.feature_extractors import bev_vgg as vgg
from avod.datasets.kitti.kitti_dataset import KittiDataset
from avod.protos import train_pb2

slim = tf.contrib.slim


def fill_feed_dict(dataset: KittiDataset, input_pl, batch_size):
    sample = dataset.next_batch(batch_size)

    bev_input = sample[0].get(constants.KEY_BEV_INPUT)
    bev_input = np.expand_dims(bev_input, axis=0)

    labels = sample[0].get(constants.KEY_LABEL_CLASSES)
    labels = np.expand_dims(labels, axis=1)

    label_pl = tf.placeholder(tf.float32, [None, 1])

    feed_dict = {
        input_pl: bev_input,
        label_pl: labels
    }

    return feed_dict, label_pl


class BevVggTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the Kitti dataset
        test_dir = tests.test_path()

        # Get the unittest-kitti dataset
        dataset_builder = DatasetBuilder()
        cls.dataset = dataset_builder.build_kitti_dataset(
            dataset_builder.KITTI_UNITTEST)

        cls.log_dir = test_dir + '/logs'
        cls.bev_vgg_cls = vgg.BevVggClassification()

    def test_vgg_layers_build(self):
        train_config_text_proto = """
        optimizer {
          gradient_descent {
           learning_rate {
             constant_learning_rate {
               learning_rate: 0.1
              }
            }
          }
        }
        """
        train_config = train_pb2.TrainConfig()
        text_format.Merge(train_config_text_proto, train_config)
        global_summaries = set([])
        batch_size = 1

        with tf.Graph().as_default():
            with tf.name_scope('input'):
                # BEV image placeholder
                image_placeholder = tf.placeholder(
                    tf.float32, (None, 700, 800, 6))
                image_summary = tf.expand_dims(image_placeholder, axis=0)
                tf.summary.image("image", image_summary, max_outputs=5)

            # Check invalid BEV shape
            bev_shape = (300, 300)
            processed_image = self.bev_vgg_cls.preprocess_input(
                image_placeholder, bev_shape)

            predictions, end_points = self.bev_vgg_cls.build(
                processed_image, num_classes=1, is_training=True)

            feed_dict, label_pl = fill_feed_dict(
                self.dataset, image_placeholder, batch_size)

            ###########################
            # Loss Function
            ###########################
            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
                label_pl,
                predictions,
                1.0)
            loss = tf.reduce_mean(cross_entropy)

            ###########################
            # Optimizer
            ###########################
            training_optimizer = optimizer_builder.build(
                train_config.optimizer, global_summaries)

            ###########################
            # Train-op
            ###########################
            train_op = slim.learning.create_train_op(loss, training_optimizer)

            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)

            loss = sess.run(train_op, feed_dict=feed_dict)

            self.assertLess(loss, 1)
            print('Loss ', loss)


if __name__ == '__main__':
    tf.test.main()

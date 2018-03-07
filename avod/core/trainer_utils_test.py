""" Trainer utilities Unit Test."""

import numpy as np
import tensorflow as tf

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.builders import optimizer_builder
from avod.core.models.rpn_model import RpnModel
from avod.core.models.avod_model import AvodModel
from avod.core import trainer
from avod.core import trainer_utils

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.ERROR)


class TrainerUtilsTest(tf.test.TestCase):

    def setUp(self):
        tf.test.TestCase.setUp(self)

        test_pipeline_config_path = avod.root_dir() + \
            '/configs/unittest_pipeline.config'

        self.model_config, self.train_config, _,  dataset_config = \
            config_builder.get_configs_from_pipeline_file(
                test_pipeline_config_path, is_training=True)

        # Generate dataset
        self.dataset = DatasetBuilder.build_kitti_dataset(
            DatasetBuilder.KITTI_UNITTEST, use_defaults=False,
            new_cfg=dataset_config)

    def test_load_model_weights(self):
        # Tests loading weights

        train_val_test = 'train'

        # Overwrite the training iterations
        self.train_config.max_iterations = 1
        self.train_config.overwrite_checkpoints = True

        with tf.Graph().as_default():
            model = RpnModel(self.model_config,
                             train_val_test=train_val_test,
                             dataset=self.dataset)
            trainer.train(model, self.train_config)

            paths_config = self.model_config.paths_config
            rpn_checkpoint_dir = paths_config.checkpoint_dir

            # load the weights back in
            init_op = tf.global_variables_initializer()

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init_op)

                trainer_utils.load_checkpoints(rpn_checkpoint_dir, saver)
                checkpoint_to_restore = saver.last_checkpoints[-1]
                trainer_utils.load_model_weights(sess, checkpoint_to_restore)

                rpn_vars = slim.get_model_variables()
                rpn_weights = sess.run(rpn_vars)
                self.assertGreater(len(rpn_weights), 0,
                                   msg='Loaded RPN weights are empty')

        with tf.Graph().as_default():
            model = AvodModel(self.model_config,
                              train_val_test=train_val_test,
                              dataset=self.dataset)
            model.build()

            # load the weights back in
            init_op = tf.global_variables_initializer()

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init_op)

                trainer_utils.load_checkpoints(rpn_checkpoint_dir, saver)
                checkpoint_to_restore = saver.last_checkpoints[-1]
                trainer_utils.load_model_weights(sess, checkpoint_to_restore)

                avod_vars = slim.get_model_variables()
                avod_weights = sess.run(avod_vars)

                # AVOD weights should include both RPN + AVOD weights
                self.assertGreater(len(avod_weights),
                                   len(rpn_weights),
                                   msg='Expected more weights for AVOD')

                # grab weights corresponding to RPN by index
                # since the model variables are ordered
                rpn_len = len(rpn_weights)
                loaded_rpn_vars = avod_vars[0:rpn_len]
                rpn_weights_reload = sess.run(loaded_rpn_vars)

                # Make sure the reloaded weights match the originally
                # loaded weights
                for i in range(rpn_len):
                    np.testing.assert_array_equal(rpn_weights_reload[i],
                                                  rpn_weights[i])

    def test_path_drop_weights(self):
        # Tests the effect of path-drop on network's feature maps.
        # It sets up a minimal-training process to check the
        # feature before and after running the 'train_op' while
        # path-drop is in effect.

        train_val_test = 'train'
        # overwrite the training iterations
        self.train_config.max_iterations = 2
        self.train_config.overwrite_checkpoints = True

        # Overwrite path drop probabilities
        model_config = config_builder.proto_to_obj(self.model_config)
        model_config.path_drop_probabilities = [0.0, 0.8]

        with tf.Graph().as_default():
            # Set a graph-level seed
            tf.set_random_seed(1245)
            model = RpnModel(model_config,
                             train_val_test=train_val_test,
                             dataset=self.dataset)
            prediction_dict = model.build()
            losses_dict, total_loss = model.loss(prediction_dict)

            global_summaries = set([])
            # Optimizer
            training_optimizer = optimizer_builder.build(
                self.train_config.optimizer,
                global_summaries)
            train_op = slim.learning.create_train_op(
                total_loss,
                training_optimizer)

            init_op = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init_op)
                for step in range(1, self.train_config.max_iterations):
                    feed_dict = model.create_feed_dict()
                    if step == 1:
                        current_feature_maps = sess.run(model.img_feature_maps,
                                                        feed_dict=feed_dict)
                        exp_feature_maps = current_feature_maps
                    train_op_loss = sess.run(train_op, feed_dict=feed_dict)
                    print('Step {}, Total Loss {:0.3f} '.
                          format(step, train_op_loss))

                    updated_feature_maps = sess.run(model.img_feature_maps,
                                                    feed_dict=feed_dict)
            # The feature maps should have remained the same since
            # the image path was dropped
            np.testing.assert_array_almost_equal(
                updated_feature_maps, exp_feature_maps, decimal=4)

    def test_disable_path_drop(self):
        # Test path drop is disabled when the probabilities
        # are set to 1.0.

        train_val_test = 'train'
        # Overwrite path drop probabilities
        model_config = config_builder.proto_to_obj(self.model_config)
        model_config.path_drop_probabilities = [1.0, 1.0]

        with tf.Graph().as_default():
            model = RpnModel(model_config,
                             train_val_test=train_val_test,
                             dataset=self.dataset)
            model.build()
            # These variables are set during path drop only
            # in the case of no path-drop, they should be non-existence
            self.assertFalse(hasattr(model, 'img_path_drop_mask'))
            self.assertFalse(hasattr(model, 'bev_path_drop_mask'))


if __name__ == '__main__':
    tf.test.main()

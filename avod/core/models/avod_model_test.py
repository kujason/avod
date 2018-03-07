"""Tests for avod.core.models.avod_model"""

import numpy as np
import tensorflow as tf

import avod
import avod.builders.config_builder_util as config_build
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import losses
from avod.core.models.avod_model import AvodModel
from avod.protos import pipeline_pb2


class AvodModelTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        pipeline_config = pipeline_pb2.NetworkPipelineConfig()
        dataset_config = pipeline_config.dataset_config
        config_path = avod.root_dir() + '/configs/unittest_model.config'

        cls.model_config = config_build.get_model_config_from_file(config_path)

        dataset_config.MergeFrom(DatasetBuilder.KITTI_UNITTEST)
        cls.dataset = DatasetBuilder.build_kitti_dataset(
            dataset_config)

    def test_avod_loss(self):
        # tests the set up for the model and the loss
        # Use "val" so that the first sample is loaded each time
        avod_model = AvodModel(self.model_config,
                               train_val_test="val",
                               dataset=self.dataset)

        predictions = avod_model.build()
        loss, total_loss = avod_model.loss(predictions)
        feed_dict = avod_model.create_feed_dict()

        with self.test_session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            loss_dict_out = sess.run(loss, feed_dict=feed_dict)
            print('Losses ', loss_dict_out)

    def test_avod_loss_correct_class_mask(self):
        # since its not easy to test the loss function directly
        # this instead tests the logic inside the loss function for
        # masking the negative classifications for the regression loss

        # This unit test tests the following :
        # Given the network classification predictions and the ground-
        # truth(gt) classifications, we find the correctly classified
        # objects by taking the argmax of predictions and gt.
        # This gives us a mask of correct classifications. This mask
        # is then multiplied by the regression predictions, meaning
        # that we only keep the regression losses of correctly identified
        # classes.
        classifications = np.array([[0.945, 0.055],
                                    [0.9, 0.1],
                                    [1.0, 0.0],  # Match
                                    [0.0, 1.0],  # Match
                                    [0.99, 0.01],  # Match
                                    [0.998, 0.0012],
                                    [1.0, 0.0]])

        classifications_gt = np.array([[0., 1.],
                                       [0., 1.],
                                       [1., 0.],  # Match
                                       [0., 1.],  # Match
                                       [1., 0.],  # Match
                                       [0., 1.],
                                       [0., 1.]])

        # Expected correct classifications
        exp_correct_cls_mask = np.array([False,
                                         False,
                                         True,
                                         True,
                                         True,
                                         False,
                                         False])

        # Expected correct classifications with background removed
        exp_positive_cls_mask = np.array([False,
                                          False,
                                          False,
                                          True,
                                          False,
                                          False,
                                          False])

        regressions = np.array([2.91572881, 2.58309603,
                                2.32328176, 2.04413271,
                                2.043607, 1.36410642,
                                0.68909782])

        # Expected regression loss after masking out the background class and
        # incorrect classifications
        exp_final_regression_loss = np.array([0.0, 0.0,
                                              0.0, 2.04413271,
                                              0.0, 0.0,
                                              0.0], dtype=np.float32)
        exp_num_positives = 1

        classifications_tensor = tf.convert_to_tensor(classifications,
                                                      dtype=tf.float32)

        classifications_gt_tensor = tf.convert_to_tensor(classifications_gt,
                                                         dtype=tf.float32)

        regression_tensor = tf.convert_to_tensor(regressions,
                                                 dtype=tf.float32)

        classification_argmax = tf.argmax(classifications_tensor, axis=1)
        class_indices_gt = tf.argmax(classifications_gt_tensor, axis=1)

        correct_classifications_mask = tf.equal(classification_argmax,
                                                class_indices_gt)

        # Mask for which predictions are not background i.e. 0
        not_background_mask = tf.greater(class_indices_gt, 0)

        # Combine the masks and cast to float to apply the mask
        pos_classification_mask = tf.logical_and(
            correct_classifications_mask, not_background_mask)

        pos_classification_floats = tf.cast(pos_classification_mask,
                                            tf.float32)

        pos_regressions = regression_tensor * \
            pos_classification_floats
        num_positives = tf.reduce_sum(pos_classification_floats)

        with self.test_session() as sess:
            # Check correct and positive classification masks
            correct_cls_mask_out, positive_cls_mask_out = \
                sess.run([correct_classifications_mask,
                          pos_classification_mask])
            self.assertAllEqual(correct_cls_mask_out,
                                exp_correct_cls_mask)
            self.assertAllEqual(positive_cls_mask_out,
                                exp_positive_cls_mask)

            # Check output after masks are applied
            regression_loss_out, num_positives_out = \
                sess.run([pos_regressions,
                          num_positives])
            self.assertAllEqual(regression_loss_out,
                                exp_final_regression_loss)
            self.assertAllEqual(num_positives_out,
                                exp_num_positives)

    def test_bool_pos_mask(self):
        # Check that applying a boolean mask gives the same output as
        # multiplying with a float mask

        pos_classification_mask = np.asarray([True, False, True],
                                             dtype=np.bool)

        offsets = np.asarray([[1, 1, 1, 1, 1, 1],
                              [2, 2, 2, 2, 2, 2],
                              [3, 3, 3, 3, 3, 3]], dtype=np.float32)
        offsets_gt = offsets + [[1], [2], [3]]

        angle_vectors = np.asarray([[0.9, 0.0361],
                                    [0.0361, 0.9],
                                    [0.7071, 0.7071]],
                                   dtype=np.float32)
        angle_vectors_gt = np.asarray([[1, 0],
                                       [0.8268, 0.5625],
                                       [0, 1]], dtype=np.float32)

        # Convert to tensors
        tf_pos_classification_floats = tf.cast(pos_classification_mask,
                                               dtype=tf.float32)
        tf_offsets = tf.convert_to_tensor(offsets, dtype=tf.float32)
        tf_offsets_gt = tf.convert_to_tensor(offsets_gt, dtype=tf.float32)
        tf_angle_vectors = tf.convert_to_tensor(angle_vectors,
                                                dtype=tf.float32)
        tf_angle_vectors_gt = tf.convert_to_tensor(angle_vectors_gt,
                                                   dtype=tf.float32)

        reg_loss = losses.WeightedSmoothL1Loss()
        anchorwise_loc_loss = reg_loss(tf_offsets,
                                       tf_offsets_gt,
                                       weight=1.0)
        anchorwise_ang_loss = reg_loss(tf_angle_vectors,
                                       tf_angle_vectors_gt,
                                       weight=1.0)

        # Masking by multiplying with mask floats
        anchorwise_combined_reg_loss = (anchorwise_loc_loss +
                                        anchorwise_ang_loss) * \
            tf_pos_classification_floats

        # Masking with tf.boolean_mask
        pos_localization_loss = tf.reduce_sum(tf.boolean_mask(
            anchorwise_loc_loss, pos_classification_mask))
        pos_orientation_loss = tf.reduce_sum(tf.boolean_mask(
            anchorwise_ang_loss, pos_classification_mask))
        combined_reg_loss = pos_localization_loss + pos_orientation_loss

        with self.test_session() as sess:
            anchorwise_loc_loss_out = sess.run(anchorwise_loc_loss)
            anchorwise_ang_loss = sess.run(anchorwise_ang_loss)

            # Masked with floats mulitplication
            anchorwise_combined_reg_loss_out = sess.run(
                anchorwise_combined_reg_loss)

            # Masked with tf.boolean_mask
            pos_loc_loss_out, pos_ang_loss_out, combined_reg_loss = \
                sess.run([pos_localization_loss,
                          pos_orientation_loss,
                          combined_reg_loss])

            pos_classification_floats_out = sess.run(
                tf_pos_classification_floats)

            expected_pos_loc_loss = np.sum(anchorwise_loc_loss_out *
                                           pos_classification_floats_out)
            expected_pos_ang_loss = np.sum(anchorwise_ang_loss *
                                           pos_classification_floats_out)
            expected_combined_reg_loss = expected_pos_loc_loss + \
                expected_pos_ang_loss

            np.testing.assert_allclose(pos_loc_loss_out, expected_pos_loc_loss)
            np.testing.assert_allclose(pos_ang_loss_out, expected_pos_ang_loss)

            # Check that floats multiplication is the same as tf.boolean_mask
            np.testing.assert_almost_equal(
                np.sum(anchorwise_combined_reg_loss_out),
                combined_reg_loss)

            # Check that combined regression loss is as expected
            np.testing.assert_almost_equal(combined_reg_loss,
                                           expected_combined_reg_loss)


if __name__ == '__main__':
    tf.test.main()

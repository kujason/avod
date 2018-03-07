# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions to build DetectionModel training optimizers."""

import tensorflow as tf

slim = tf.contrib.slim


def build(optimizer_config,
          global_summaries,
          global_step=None):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.
        global_summaries: A set to attach learning rate summary to.
        global_step: (optional) A tensor that contains the global step.
            This is required for applying exponential decay to the learning
            rate.

    Returns:
        An optimizer.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    optimizer_type = optimizer_config.WhichOneof('optimizer')
    optimizer = None

    if optimizer_type == 'rms_prop_optimizer':
        config = optimizer_config.rms_prop_optimizer
        optimizer = tf.train.RMSPropOptimizer(
            _create_learning_rate(config.learning_rate,
                                  global_summaries,
                                  global_step),
            decay=config.decay,
            momentum=config.momentum_optimizer_value,
            epsilon=config.epsilon)

    elif optimizer_type == 'momentum_optimizer':
        config = optimizer_config.momentum_optimizer
        optimizer = tf.train.MomentumOptimizer(
            _create_learning_rate(config.learning_rate,
                                  global_summaries,
                                  global_step),
            momentum=config.momentum_optimizer_value)

    elif optimizer_type == 'adam_optimizer':
        config = optimizer_config.adam_optimizer
        optimizer = tf.train.AdamOptimizer(
            _create_learning_rate(config.learning_rate,
                                  global_summaries,
                                  global_step))

    elif optimizer_type == 'gradient_descent':
        config = optimizer_config.gradient_descent
        optimizer = tf.train.GradientDescentOptimizer(
            _create_learning_rate(config.learning_rate,
                                  global_summaries,
                                  global_step))

    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)

    if optimizer_config.use_moving_average:
        optimizer = tf.contrib.opt.MovingAverageOptimizer(
            optimizer, average_decay=optimizer_config.moving_average_decay)

    return optimizer


def _create_learning_rate(learning_rate_config,
                          global_summaries,
                          global_step):
    """Create optimizer learning rate based on config.

    Args:
        learning_rate_config: A LearningRate proto message.
        global_summaries: A set to attach learning rate summary to.
        global_step: A tensor that contains the global step.

    Returns:
        A learning rate.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    learning_rate = None
    learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
    if learning_rate_type == 'constant_learning_rate':
        config = learning_rate_config.constant_learning_rate
        learning_rate = config.learning_rate

    elif learning_rate_type == 'exponential_decay_learning_rate':
        config = learning_rate_config.exponential_decay_learning_rate
        learning_rate = tf.train.exponential_decay(
            config.initial_learning_rate,
            global_step,
            config.decay_steps,
            config.decay_factor,
            staircase=config.staircase)

    if learning_rate is None:
        raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

    global_summaries.add(tf.summary.scalar('Learning_Rate', learning_rate))
    return learning_rate

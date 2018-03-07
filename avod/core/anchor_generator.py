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

"""Base anchor generator.

The job of the anchor generator is to create (or load) a collection
of bounding boxes to be used as anchors.

Generated anchors are assumed to match some convolutional grid or list of grid
shapes.  For example, we might want to generate anchors matching an 8x8
feature map and a 4x4 feature map.  If we place 3 anchors per grid location
on the first feature map and 6 anchors per grid location on the second feature
map, then 3*8*8 + 6*4*4 = 288 anchors are generated in total.

To support fully convolutional settings, feature map shapes are passed
dynamically at generation time.  The number of anchors to place at each location
is static --- implementations of AnchorGenerator must always be able return
the number of anchors that it uses per location for each feature map.
"""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf


class AnchorGenerator(object):
    """Abstract base class for anchor generators."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def name_scope(self):
        """Name scope.

        Must be defined by implementations.

        Returns:
          a string representing the name scope of the anchor generation operation.
        """
        pass

    def generate(self, **params):
        """Generates a collection of bounding boxes to be used as anchors.
        """
        return self._generate(**params)

    @abstractmethod
    def _generate(self, **params):
        """To be overridden by implementations.

        Args:
          **params: parameters for anchor generation op

        Returns:
          boxes: a BoxList holding a collection of N anchor boxes
        """
        pass

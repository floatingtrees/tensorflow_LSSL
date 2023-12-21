# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Linear State Space Layer"""

import functools

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops.array_ops import shape



class LSSL(Layer):

  def __init__(self, units, use_bias = True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint = None, 
                bias_constraint=None,
                trainable=True,
                name=None,
                conv_op=None,**kwargs):
    super(LSSL, self).__init__(trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.units = units
    self.use_bias = use_bias



  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape) # expected batch_size, sequence_length, features
    self.add_weight(name = "A", 
                    shape = (input_shape[-1], input_shape[-1]),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    trainable=True
      )
    if self.use_bias:
      self.add_weight(name = "B", 
                      shape = (1, input_shape[-1]),
                      initializer=self.kernel_initializer,
                      regularizer=self.kernel_regularizer,
                      constraint=self.kernel_constraint,
                      trainable=True
        )
    self.add_weight(name = "C", 
                    shape = (1, input_shape[-1]),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    trainable=True
      )
    if self.use_bias:
      self.add_weight(name = "D", 
                      shape = (1, 1),
                      initializer=self.kernel_initializer,
                      regularizer=self.kernel_regularizer,
                      constraint=self.kernel_constraint,
                      trainable=True
        )


    def call(self, inputs, training = None, mask = None):
      training = self._get_training_value(training)
      if training is not None:
        if training:
          input_shape = array_ops.shape(inputs)
          kernel_size = input_shape[-2]
          in_units = input_shape[-1]
          kernel = array_ops.zeros((1, kernel_size, in_units, self.units))
          A = 
          for i in range(kernel_size):
            
            
          return nn.conv(inputs, kernel, 1, "VALID")
        else:
          pass





  def get_config(self):
    config = {
        'strides': self.strides,
        'pool_size': self.pool_size,
        'padding': self.padding,
        'data_format': self.data_format,
    }
    base_config = super(LSSL, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))





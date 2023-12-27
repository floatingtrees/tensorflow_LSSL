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

  def __init__(self, units, dt, use_bias = True,
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
    self.dt = dt
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.units = units
    self.use_bias = use_bias
    self.built = False
    self._validate_init()

  def _validate_init(self):
    if self.dt <= 0:
      raise ValueError("Timestep (dt) must be greater than 0")


  def build(self, input_shape):
    self.built = True
    input_shape = tensor_shape.TensorShape(input_shape) # expected batch_size, sequence_length, features
    self.A = self.add_weight(name = "A", 
                    shape = (input_shape[-1], input_shape[-1]),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    trainable=True
      )
    if self.use_bias:
      self.B = self.add_weight(name = "B", 
                      shape = (1, input_shape[-1]),
                      initializer=self.kernel_initializer,
                      regularizer=self.kernel_regularizer,
                      constraint=self.kernel_constraint,
                      trainable=True
        )
    else:
      self.B = None
    self.C = self.add_weight(name = "C", 
                    shape = (1, input_shape[-1]),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    trainable=True
      )
    if self.use_bias:
      self.D = self.add_weight(name = "D", 
                      shape = (1, 1),
                      initializer=self.kernel_initializer,
                      regularizer=self.kernel_regularizer,
                      constraint=self.kernel_constraint,
                      trainable=True
        )
    else:
      self.D = None


    def call(self, inputs):
      training = True
      if training is None:
        training = self._get_training_value(training)
      print("HERE")
      if training is not None:
        input_shape = array_ops.shape(inputs)
        dAdt = self.dt * self.A
        I = array_ops.eye(input_shape[-1])
        A_descretized = (I + dAdt) / (I - dAdt)
        dBdt = self.dt * self.B
        B_descretized = (I + dBdt) / (I - dBdt)
        C_descretized = self.C
        if training: # convolutional output
          input_shape = array_ops.shape(inputs)
          kernel_size = input_shape[-2]
          in_units = input_shape[-1]
          kernel = array_ops.zeros((1, kernel_size, in_units, self.units))
          exp_matrix = array_ops.eye(input_shape[-1]) # GOOGLE SAYS THAT THIS IS THE IDENTITY MATRIX
          for i in range(kernel_size):
            
            kernel[i] = C_descretized * exp_matrix * B_descretized
            if i != kernel_size - 1: # no need to multiply
              exp_matrix = exp_matrix * self.A 
          print("HERE", kernel.shape)
          return nn.conv(inputs, kernel, 1, "VALID")
        else: # output in recurrent state
          
          
          X = array_ops.zeros(array_ops.shape(self.A))
          output_array =  array_ops.zeros(input_shape[0], input_shape[1], self.units)
          for i in range(kernel_size):
            X = A_descretized * X + B_descretized * inputs[:, i, :]
            Y = C_descretized * X
            output_array[:, i, :] = Y
          return output_array



  def get_config(self):
    config = {
        'strides': self.strides,
        'pool_size': self.pool_size,
        'padding': self.padding,
        'data_format': self.data_format,
    }
    base_config = super(LSSL, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))





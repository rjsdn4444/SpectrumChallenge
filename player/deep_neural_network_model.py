from typing import Dict, List
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np


class DnnModel(Model):
    def __init__(self, conv_layers_list: List[Dict], num_action: int):
        super(DnnModel, self).__init__()
        self._num_action = num_action
        self._conv_layers = []
        for conv_layer in conv_layers_list:
            filters = conv_layer['filters']
            kernel_size = conv_layer['kernel_size']
            strides = conv_layer['strides']
            layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                           padding='same', activation=tf.keras.activations.relu,
                           kernel_initializer='glorot_normal', bias_initializer='zeros')
            self._conv_layers.append(layer)
            if conv_layer['max_pool_size'] is not None:
                max_pool_size = conv_layer['max_pool_size']
                layer = MaxPooling2D(pool_size=max_pool_size)
                self._conv_layers.append(layer)
        self._flatten = Flatten()
        self._fully_conn_layer = Dense(units=self._num_action, activation=tf.keras.activations.softmax,
                                       kernel_initializer='glorot_normal', bias_initializer='zeros')

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self._conv_layers:
            x = layer(x)
        x = self._flatten(x)
        outputs = self._fully_conn_layer(x)
        return outputs


# Test Tensorflow model
if __name__ == '__main__':
    layers = [
        {'filters': 4,
         'kernel_size': (2, 2),
         'strides': (2, 2),
         'max_pool_size': None},
        {'filters': 8,
         'kernel_size': (4, 4),
         'strides': (1, 1),
         'max_pool_size': (2, 2)}
    ]
    model = DnnModel(layers, 10)
    learning_rate = 0.001
    model.compile(optimizer=Adam(lr=learning_rate), loss="mse")
    x = np.random.random((3, 1000, 100, 2))
    print(model(x))

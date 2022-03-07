from typing import Dict, List
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np


class Actor(Model):
    def __init__(self, conv_layers_list: List[Dict], num_action: int):
        super(Actor, self).__init__()
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


class Critic(Model):
    def __init__(self, conv_layers_list: List[Dict], num_action: int):
        super(Critic, self).__init__()
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
        self._fully_conn_layer = Dense(units=self._num_action, activation=tf.keras.activations.tanh,
                                       kernel_initializer='glorot_normal', bias_initializer='zeros')

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self._conv_layers:
            x = layer(x)
        x = self._flatten(x)
        outputs = self._fully_conn_layer(x)
        return outputs


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
            if conv_layer['batch_normalize'] is not None:
                layer = BatchNormalization()
                self._conv_layers.append(layer)
            if conv_layer['max_pool_size'] is not None:
                max_pool_size = conv_layer['max_pool_size']
                layer = MaxPooling2D(pool_size=max_pool_size)
                self._conv_layers.append(layer)
        self._flatten = Flatten()
        self._policy = Dense(units=self._num_action, activation=tf.keras.activations.softmax,
                             kernel_initializer='glorot_normal', bias_initializer='zeros')
        self._value = Dense(units=1, activation=tf.keras.activations.tanh,
                            kernel_initializer='glorot_normal', bias_initializer='zeros')

    @tf.function
    def call(self, inputs):
        x = inputs
        for i, layer in enumerate(self._conv_layers):
            x = layer(x)
        x = self._flatten(x)
        policy = self._policy(x)
        value = self._value(x)
        return policy, value


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        return output


def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))
    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))
    return res_block


def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))
    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))
    return res_block


class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params, num_action):
        super(ResNetTypeI, self).__init__()
        self._num_action = num_action
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        self.layer1 = make_basic_block_layer(filter_num=64, blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128, blocks=layer_params[1], stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256, blocks=layer_params[2], stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512, blocks=layer_params[3], stride=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.policy = tf.keras.layers.Dense(units=self._num_action, activation=tf.keras.activations.softmax)
        self.value = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.relu)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value


class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params, num_action):
        super(ResNetTypeII, self).__init__()
        self._num_action = num_action
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        self.layer1 = make_bottleneck_layer(filter_num=64, blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128, blocks=layer_params[1], stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256, blocks=layer_params[2], stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512, blocks=layer_params[3], stride=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.policy = tf.keras.layers.Dense(units=self._num_action, activation=tf.keras.activations.softmax)
        self.value = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.relu)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value


def resnet_18(num_action, writer, step):
    resnet = ResNetTypeI(layer_params=[2, 2, 2, 2], num_action=num_action)
    # with writer.as_default():
    #     tf.summary.histogram('model', resnet, step=step)
    return resnet


def resnet_34(num_action, writer, step):
    resnet = ResNetTypeI(layer_params=[3, 4, 6, 3], num_action=num_action)
    # with writer.as_default():
    #     tf.summary.histogram('model', resnet, step=step)
    return resnet


def resnet_50(num_action, writer, step):
    resnet = ResNetTypeII(layer_params=[3, 4, 6, 3], num_action=num_action)
    # with writer.as_default():
    #     tf.summary.histogram('model', resnet, step=step)
    return resnet


def resnet_101(num_action, writer, step):
    resnet = ResNetTypeII(layer_params=[3, 4, 23, 3], num_action=num_action)
    # with writer.as_default():
    #     tf.summary.histogram('model', resnet, step=step)
    return resnet


def resnet_152(num_action, writer, step):
    resnet = ResNetTypeII(layer_params=[3, 8, 36, 3], num_action=num_action)
    # with writer.as_default():
    #     tf.summary.histogram('model', resnet, step=step)
    return resnet

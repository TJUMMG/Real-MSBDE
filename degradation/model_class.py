import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import sys
from layer import *


class Degradation_Class:
    def __init__(self, LBD_Q, res, is_training, batch_size):
        self.batch_size = batch_size
        # self.label = label
        self.category, self.logit = self.ResNet(LBD_Q, res, is_training, batch_size)
        # self.cost = self.loss(self.logit, self.label)

    def ResNet(self, LBD_Q, res, is_training, batch_size):
        with tf.compat.v1.variable_scope('ResNet', reuse=False):
            height = int(LBD_Q.get_shape()[1])
            width = int(LBD_Q.get_shape()[2])
            with tf.compat.v1.variable_scope('conv_q'):
                x_q = conv_layer(LBD_Q, [3, 3, 3, 64], 1)
            with tf.compat.v1.variable_scope('conv_r'):
                x_r = conv_layer(res, [3, 3, 3, 64], 1)
            with tf.compat.v1.variable_scope('concat'):
                x1 = tf.concat([x_q, x_r], 3)
                x1 = conv_layer(x1, [3, 3, 128, 64], 1)
            with tf.compat.v1.variable_scope('residual1_1'):
                x2 = batch_normalize(x1, is_training)
                x2 = conv_layer(x2, [3, 3, 64, 64], 1)
            with tf.compat.v1.variable_scope('residual1_2'):
                x2 = batch_normalize(x2, is_training)
                x2 = tf.compat.v1.nn.relu(x2)
                x2 = conv_layer(x2, [3, 3, 64, 64], 1)
                x2 += x1
            with tf.compat.v1.variable_scope('residual2_1'):
                x3 = batch_normalize(x2, is_training)
                x3 = conv_layer(x3, [3, 3, 64, 64], 1)
            with tf.compat.v1.variable_scope('residual2_2'):
                x3 = batch_normalize(x3, is_training)
                x3 = tf.compat.v1.nn.relu(x3)
                x3 = conv_layer(x3, [3, 3, 64, 64], 1)
                x3 += x2
            with tf.compat.v1.variable_scope('residual3_1'):
                x4 = batch_normalize(x3, is_training)
                x4 = conv_layer(x4, [3, 3, 64, 64], 1)
            with tf.compat.v1.variable_scope('residual3_2'):
                x4 = batch_normalize(x4, is_training)
                x4 = tf.compat.v1.nn.relu(x4)
                x4 = conv_layer(x4, [3, 3, 64, 64], 1)
                x4 += x3
            with tf.compat.v1.variable_scope('residual4_1'):
                x5 = batch_normalize(x4, is_training)
                x5 = conv_layer(x5, [3, 3, 64, 64], 1)
            with tf.compat.v1.variable_scope('residual4_2'):
                x5 = batch_normalize(x5, is_training)
                x5 = tf.compat.v1.nn.relu(x5)
                x5 = conv_layer(x5, [3, 3, 64, 64], 1)
                x5 += x4
            with tf.variable_scope('conv2'):
                x6 = batch_normalize(x5, is_training)
                x6 = conv_layer(x6, [3, 3, 64, 64], 1)
            with tf.variable_scope('fc'):
                x7 = conv_layer(x6, [3, 3, 64, 9], 1)
            with tf.variable_scope('reshape'):
                x8 = tf.reshape(x7, [batch_size, height, width, 3, 3])
                x9 = tf.compat.v1.nn.softmax(x8)
            prediction = tf.argmax(x9, dimension=4)

        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return prediction, x8


    def loss(self, logit, label):
        # label = tf.one_hot(label, depth=3, axis=3, on_value=1.0)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label), name='loss')
        tf.summary.scalar('cost', cost)
        return cost


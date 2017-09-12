import numpy as np
import tensorflow as tf
import math

class Conv2D_transpose(object):
    def __init__(self, in_channels, output_shape, k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, name_scope="conv2d_transpose"):
        with tf.name_scope(name_scope) as scope:
            # filter : [height, width, output_channels, in_channels]
            self.output_shape = output_shape
            self.s_h = s_h
            self.s_w = s_w
            self.weights = tf.Variable(tf.truncated_normal([k_h, k_w, output_shape[-1], in_channels], stddev=stddev, name="weights"))
            self.biases = tf.Variable(tf.zeros([output_shape[-1]]), name="biases")

    def add(self, prev):
        c = tf.nn.conv2d_transpose(prev, self.weights, output_shape=self.output_shape,strides=[1, self.s_h, self.s_w, 1])
        return tf.nn.bias_add(c, self.biases)

            #
            # if self.activation=="ReLU":
            #     hidden = tf.nn.relu(tf.matmul(prev, self.weights) + self.biases)
            #     return hidden
            # elif self.activation=="Softmax":
            #     # softmax will be done in the loss calc by tf.nn.softmax_cross_entropy_with_logits
            #     hidden = tf.matmul(prev, self.weights) + self.biases
            #     return hidden
            # else:
            #     print("Please specify activation function.")
            #     raise NotImplementedError()

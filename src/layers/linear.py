import numpy as np
import tensorflow as tf
import math

class Linear(object):
    def __init__(self, n_in, n_out, name_scope):
        with tf.name_scope(name_scope) as scope:
            self.weights = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=1.0/math.sqrt(float(n_in)), name="weights"))
            self.biases = tf.Variable(tf.zeros([n_out]), name="biases")

    def add(self, prev):
        return tf.matmul(prev, self.weights) + self.biases

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

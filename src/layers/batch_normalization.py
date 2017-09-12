import numpy as np
import tensorflow as tf
import math

class BN(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.name_scope(name) as scope:
            self.epsilon  = epsilon
            self.momentum = momentum
            self.scope = scope

    def add(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                    decay=self.momentum,
                    updates_collections=None,
                    epsilon=self.epsilon,
                    scale=True,
                    is_training=train,
                    scope=self.scope)

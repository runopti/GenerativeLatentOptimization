import tensorflow as tf

class Sigmoid(object):
    def __init__(self):
        pass

    def add(self, prev):
        return tf.nn.sigmoid(prev)

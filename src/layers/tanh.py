import tensorflow as tf

class Tanh(object):
    def __init__(self):
        pass

    def add(self, prev):
        return tf.nn.tanh(prev)

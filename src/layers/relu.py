import tensorflow as tf

class ReLU(object):
    def __init__(self):
        pass

    def add(self, prev):
        return tf.nn.relu(prev)

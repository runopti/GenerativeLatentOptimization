import numpy as np
import tensorflow as tf
import math

class Identity(object):
    def __init__(self):
        pass
    def add(self, prev):
        return prev

## NOTE:
## The code for the network construction is partially based off of https://github.com/takerum/vat/blob/master/models/fnn_mnist_sup.py
## mlp2.py might be more efficient than mlp.py in terms of graph compilcation.
import tensorflow as tf
import numpy as np
import math
import os
from src import layers as L
from src import costs as cost

class Config(object):
    def __init__(self):
        self.sample_size = 55000
        #self.arch = [512, 1024, 2048, 4800]
        self.arch = [100, 300, 500, 784]
        self.n_input = 100
        self.n_output = 784
        self.batch_size = 100
        self.lr_z = 0.001 # 10.0 # 10.0
        self.lr_theta = 0.00001 #1.0 # 0.001 # 5.0 #10.0
        self.train_clip = False
        self.thresh = 3

        self.n_epoch_z = 5



class MLP(object):
    def __init__(self, config):
        self.n_z = int(config.sample_size / config.batch_size)
        self.input = tf.Variable(tf.truncated_normal([config.sample_size, config.n_input], stddev=1.0/math.sqrt(float(config.n_input)), name="latent_z"))
        # self.input = tf.Variable(tf.truncated_normal([self.n_z, config.batch_size, config.n_input], stddev=1.0/math.sqrt(float(config.n_input)), name="latent_z"))
        #self.begin_id = tf.placeholder(tf.int32, shape=[1])
        #self.end_id = tf.placeholder(tf.int32, shape=[1])
        # self.input_batch = tf.slice(self.input, [0, self.begin_id], [config.n_input, self.end_id-self.begin_id+1])
        self.batch_indices = tf.placeholder(tf.int32, shape=[config.batch_size])
        self.batch_z = tf.nn.embedding_lookup(self.input, self.batch_indices)
        # self.batch_id = 0
        # batch_id = self.batch_id.eval()
        self.target = tf.placeholder(tf.float32, shape=[config.batch_size, config.n_output])
        arch = config.arch

        # Construct a network
        self.lin_layers = []
        self.act_layers = []
        for n, m in zip(arch[:-1], arch[1:]):
            l = L.Linear(n_in=n, n_out=m, name_scope="linear")
            self.lin_layers.append(l)
        for i in range(len(self.lin_layers)-1):
            self.act_layers.append(L.ReLU())
        self.act_layers.append(L.Identity())
        # Feed the data to connect layers in the computational graph
        self.logits = self._inference(self.batch_z)
        # Get loss
        # self.loss = cost.nll(self.logits, self.target)
        self.loss = cost.least_squares(self.logits, self.target)
        # vars_z = [v for v in tf.trainable_variables() if "latent_z" in v.name]
        # print(vars_z[0].get_shape().as_list())
        grads_z = tf.gradients(self.loss, self.input)
        # optimizer_z = tf.train.GradientDescentOptimizer(config.lr_z)
        optimizer_z = tf.train.AdamOptimizer(config.lr_z)
        # For backprop call
        self.train_op_z = optimizer_z.apply_gradients(zip(grads_z, [self.input]))
        # Project onto L2 unit ball
        normalized_vals = tf.nn.l2_normalize(self.input, dim=1)  # row-wise
        self.project_op_z = tf.assign(self.input, normalized_vals)
        # self.input[self.batch_id] = tf.div(self.input[self.batch_id], tf.norm(self.input[self.batch_id], 2))

        vars_theta = [v for v in tf.trainable_variables() if "weight" or "bias" in v.name]
        # print(vars_theta)
        grads_theta = tf.gradients(self.loss, vars_theta)
        # optimizer_theta = tf.train.GradientDescentOptimizer(config.lr_theta)
        optimizer_theta = tf.train.AdamOptimizer(config.lr_theta)
        # For backprop call
        self.train_op_theta = optimizer_theta.apply_gradients(zip(grads_theta, vars_theta))

        # For convenience
        # self.y_hat = tf.nn.softmax(self.logits)
        ## argmax for each row of y_hat (batch_size,n_output) and compare with self.targets
        ## to get accuracy
        # bool_vec = tf.equal(tf.argmax(self.y_hat,axis=1),tf.argmax(self.target,axis=1))
        # self.acc = tf.reduce_mean(tf.cast(bool_vec, tf.float32))

        num_params = 0
        for var in tf.global_variables():
            num_params += np.prod(var.get_shape().as_list())
        print("Number of Model Parameters: {}".format(num_params))
        print("Size of Model : {} MB".format(num_params*4/1e6)) #4 = tf.float32

        # Create a saver
        self.saver = tf.train.Saver()

        # Start Session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def increaseBatchID(self):
        if self.batch_id < self.n_z-1:
            self.batch_id += 1
        else:
            self.batch_id = 0

    def getBatchID(self):
        return self.batch_id

    def visualize(self, batch_indices):
        output = self.sess.run([self.logits], feed_dict={self.batch_indices: batch_indices})
        return output[0]

    def project_z_L2(self):
        self.sess.run(self.project_op_z)

    def _inference(self, input):
        h = input
        for l, act in zip(self.lin_layers, self.act_layers):
            h = l.add(h)
            h = act.add(h)
        return h

    def forward_backprop_z(self, data, batch_indices):
        cost, _ = self.sess.run([self.loss, self.train_op_z], feed_dict={self.target:data, self.batch_indices: batch_indices})
        return cost

    def forward_backprop_theta(self, data, batch_indices):
        cost, _ = self.sess.run([self.loss, self.train_op_theta], feed_dict={self.target:data, self.batch_indices: batch_indices})
        return cost

    def forward(self, data, targets):
        cost = self.sess.run([self.loss], feed_dict={self.target:data})
        return cost[0]

    def get_prediction(self, data):
        y_hat = self.sess.run([self.y_hat], feed_dict={self.input: data})
        return y_hat[0]

    def get_accuracy(self, data, targets):
        acc = self.sess.run([self.acc], feed_dict={self.input:data, self.target: targets})
        return acc[0]

    def load(self, model_path=None):
        if model_path == None:
            raise Exception()
        self.saver.restore(self.sess, model_path)

    def save(self, step, model_dir=None):
        if model_dir == None:
            raise Exception()
        try:
            os.mkdir(model_dir)
        except:
            pass
        model_file = model_dir + "/model"
        self.saver.save(self.sess, model_file, global_step=step)

    def getWeights(self):
        pass

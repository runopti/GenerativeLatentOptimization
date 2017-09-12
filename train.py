'''
 * =====================================================================================
 *
 * 	Author: Yutaro Yamada
 * 	Date:  05/25/2017
 *
 * =====================================================================================
'''


import tensorflow as tf
import os,sys,math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from args import args
import numpy as np
from models.dcgn import DCGN, Config
from utils.progress import Progress
import pickle

def main():
    # load data
    mnist = input_data.read_data_sets(args.dataset_path+'MNIST_data', one_hot=True)
    # we can access to images like this:
    # images = mnist.train.images;  images.shape = []
    # labels = mnist.train.labels; each label is a probability distribution.
    print(mnist.train.num_examples)

    max_epoch = args.max_epoch
    max_loop_z = args.max_loop_z
    with tf.Graph().as_default():
        config = Config()
        # model = MLP(config)
        model = DCGN(config)
        tf.get_default_graph().finalize()

        progress = Progress()

        n_batch_loop = int(mnist.train.num_examples/config.batch_size)
        for epoch in range(max_epoch):
            sum_cost = 0
            progress.start_epoch(epoch, max_epoch)

            for t in range(n_batch_loop):
                # batch_X: batch_size x n_input
                # batch_y: batch_size

                batch_X, batch_y = mnist.train.next_batch(config.batch_size, shuffle=False)
                batch_indices = np.arange(t*config.batch_size, t*config.batch_size+config.batch_size)
                for z_t in range(max_loop_z):
                    cost_z = model.forward_backprop_z(batch_X, batch_indices)
                    model.project_z_L2()
                    #print(cost_z)

                cost_per_sample = model.forward_backprop_theta(batch_X, batch_indices)
                sum_cost += cost_per_sample
                # model.increaseBatchID()

                if t % 10 == 0:
                    progress.show(t, n_batch_loop, {})

            print("cost: {}".format(sum_cost/n_batch_loop))
            model.save(epoch, args.model_dir)


            # Validation
            # val_loss, val_acc = evaluate(model, mnist.validation, config)
            #val_loss = evaluate(model, mnist.validation, config)
            #progress.show(n_batch_loop, n_batch_loop, {
        #        "val_loss" : val_loss,
                # "val_acc" : val_acc,
        #        })

def evaluate(model, dataset, config):
    n_batch_loop = int(dataset.num_examples/config.batch_size)
    sum_cost = 0
    # sum_acc = 0
    for t in range(n_batch_loop):
        batch_X, batch_y = dataset.next_batch(config.batch_size)
        cost_per_sample = model.forward(batch_X, batch_y)
        sum_cost += cost_per_sample
    cost_avg = sum_cost / n_batch_loop
    return cost_avg


def test():
    pass

if __name__=="__main__":
    main()
    #test()

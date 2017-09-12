#from models.mlp import MLP, Config
from models.dcgn import DCGN, Config
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from args import args
import matplotlib.pyplot as plt

def evaluate(model, dataset, config):
    n_batch_loop = int(dataset.num_examples/config.batch_size) 
    sum_cost = 0
    sum_acc = 0 
    for t in range(n_batch_loop): 
        batch_X, batch_y = dataset.next_batch(config.batch_size) 
        cost_per_sample, acc = model.forward(batch_X, batch_y)
        sum_cost += cost_per_sample
        sum_acc += acc
    acc_avg = sum_acc / n_batch_loop 
    cost_avg = sum_cost / n_batch_loop
    return cost_avg, acc_avg 


def main():
    mnist = input_data.read_data_sets(args.dataset_path+'MNIST_data', one_hot=True)
    with tf.Graph().as_default():
        config = Config()
        model = DCGN(config)
        # model = MLP(config)
        model.load("./trained_models_dcgn/model-9")

        t = 0
        batch_indices = np.arange(t*config.batch_size, t*config.batch_size+config.batch_size)
        id = 1 #14
        image = model.visualize(batch_indices)
        print(image.shape)
        plt.imshow(image[id].reshape(28,28), cmap="gray")
        plt.show()

        t += 1
        batch_indices = np.arange(t*config.batch_size, t*config.batch_size+config.batch_size)
        image = model.visualize(batch_indices)
        plt.imshow(image[id].reshape(28,28), cmap="gray")
        plt.show()

        t += 1
        batch_indices = np.arange(t*config.batch_size, t*config.batch_size+config.batch_size)
        image = model.visualize(batch_indices)
        plt.imshow(image[id].reshape(28,28), cmap="gray")
        plt.show()
        
        batch_X, batch_y = mnist.train.next_batch(config.batch_size, shuffle=False)
        plt.imshow(batch_X[id].reshape(28,28),cmap="gray")
        plt.show()

        batch_X, batch_y = mnist.train.next_batch(config.batch_size, shuffle=False)
        plt.imshow(batch_X[id].reshape(28,28),cmap="gray")
        plt.show()

        batch_X, batch_y = mnist.train.next_batch(config.batch_size, shuffle=False)
        plt.imshow(batch_X[id].reshape(28,28),cmap="gray")
        plt.show()
        #test_loss, test_acc = evaluate(model, mnist.test, config)
        #print("loss: {}, acc: {}".format(test_loss, test_acc))
        

if __name__=="__main__":
    main()


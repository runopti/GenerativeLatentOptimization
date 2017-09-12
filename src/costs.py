import tensorflow as tf

def nll(pred, target):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=target))

def least_squares(pred, target):
    return tf.reduce_mean(tf.square(pred-target))

def kl(self, train_clip, thresh):
    # get eval_kl for each layer
    [n.name for n in tf.get_default_graph().as_graph_def().node if 'kl' in n.name]

    tf.get_default_graph().get_tensor_by_name('hidden1/weights:0')
    pass

def ell(pred, target):
    # check if this is right
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=target))

def sgvlb(self, pred, target, config):
    return -(config.n_input*1.0/config.batch_size*ell(pred,target) - config.rw * kl(config.train_clip, config.thresh))

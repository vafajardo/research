import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm, maxout, dropout
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os
import sys
import matplotlib
matplotlib.use('Agg')
from plotting import *

"""
Training a generative adversarial net on MNIST.
"""

# setup logdir
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# Loading MNIST data
mnist = input_data.read_data_sets("/tmp/data/")

# Construction phase
### For simplicity, I set G and D to have the same architecture
n_inputs = 28*28
n_G_hidden1 = 1200
n_G_hidden2 = 1200
n_D_hidden1 = 196
n_D_hidden2 = 196
n_outputs = n_inputs

# Placeholder for minibatch sample from true observations
X = tf.placeholder(tf.float32, shape = [None, n_inputs])
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

# Generator -- have it as same dimension as X for now
Z = tf.random_normal(tf.shape(X), dtype=tf.float32)
G_hidden1 = fully_connected(Z, n_G_hidden1, activation_fn=tf.nn.relu)
G_hidden2 = fully_connected(G_hidden1, n_G_hidden2, activation_fn=tf.nn.relu)
G_logits = fully_connected(G_hidden2, n_outputs, activation_fn=None)
G_outputs = tf.sigmoid(G_logits)

# Discriminator -- uses maxout units as in the original paper
keep_prob = 0.8
def get_discriminative_probability(x, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        D_hidden1 = maxout(x, n_D_hidden1)
        D_hidden2 = maxout(tf.reshape(D_hidden1,(-1,n_D_hidden1)), n_D_hidden2)
        D_hidden2_drop = dropout(D_hidden2, keep_prob, is_training=is_training)

        outputs = fully_connected(tf.reshape(D_hidden2_drop,(-1,n_D_hidden2)),
                            1, activation_fn=tf.nn.sigmoid)
        return outputs

D_outputs_from_G = get_discriminative_probability(G_outputs)
D_outputs_from_X = get_discriminative_probability(X, reuse=True)

# MinMax problem
D_objective = -(tf.reduce_mean(tf.log(1 - D_outputs_from_G))
                        + tf.reduce_mean(tf.log(D_outputs_from_X)))
G_objective = tf.reduce_mean(tf.log(1 - D_outputs_from_G))

# Instantiate the SGD optimizers
learning_rate = 0.000001
D_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
G_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
D_training_op = D_optimizer.minimize(D_objective)
G_training_op = G_optimizer.minimize(G_objective)

cost_summary = tf.summary.scalar('G_obj', G_objective)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Training params
n_epochs = 50000
batch_size = 150
n_iter_D = 1
n_digits = 60

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # update Discriminator
            sess.run(D_training_op, feed_dict={X: X_batch, is_training: True})
            if iteration % n_iter_D == 0:
                # update Generator
                sess.run(G_training_op, feed_dict={X: X_batch, is_training: True})
        D_obj_value, G_obj_value = sess.run([D_objective,G_objective],
                feed_dict={X: X_batch, is_training: True})
        print("\r{}".format(epoch), "D obj value:", D_obj_value,
                 "\tG obj value:", G_obj_value)
        saver.save(sess, "./my_model_all_layers.ckpt")
    # generating digits
    codings_rnd = np.random.normal(size=[n_digits, n_inputs])
    outputs_val = G_outputs.eval(feed_dict={Z: codings_rnd, is_training: False})

n_rows = 6
n_cols = 10
plot_multiple_images(outputs_val.reshape(-1, 28, 28), n_rows, n_cols)
save_fig("gans_generated_digits_plot_k{}".format(n_iter_D))
plt.show()

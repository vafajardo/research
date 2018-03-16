import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm, maxout
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

# Loading MNIST data
mnist = input_data.read_data_sets("/tmp/data/")

# Construction phase
### For simplicity, I set G and D to have the same architecture
n_inputs = 28*28
n_G_hidden1 = 200
n_G_hidden2 = 200
n_D_hidden1 = 150
n_D_hidden2 = 75
n_outputs = n_inputs

# Generator -- have it as same dimension as X for now
Z = tf.placeholder(tf.float32, shape = [None, n_inputs])
G_hidden1 = fully_connected(Z, n_hidden1, activation_fn=tf.nn.sigmoid)
G_hidden2 = fully_connected(G_hidden1, n_hidden2, activation_fn=tf.nn.relu)
G_logits = fully_connected(G_hidden2, n_outputs, activation_fn=None)
G_outputs = tf.sigmoid(logits)

# Discriminator -- uses maxout units as in the original paper
X = tf.placeholder(tf.float32, shape = [None, n_inputs])
x_from_G = tf.placeholder(tf.bool, shape=(), name='x_from_G')
if x_from_G == True:
    D_hidden1 = maxout(G, n_D_hidden1)
    D_hidden2 = maxout(G, n_D_hidden2)
else:
    D_hidden1 = maxout(X, n_D_hidden1)
    D_hidden2 = maxout(X, n_D_hidden2)
D_outputs = fully_connected(D_hidden2, 1, activation_fn=tf.nn.sigmoid)

# MinMax problem
if x_from_G == True:
    D_objective = tf.reduce_mean(tf.log(1 - D_outputs))
    G_objective = tf.reduce_mean(tf.log(1 - D_outputs))
else:
    D_objective = tf.reduce_mean(tf.log(D_outputs))

D_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
G_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
D_training_op = D_optimizer.maximize(D_objective)
G_training_op = G_optimizer.minimize(G_objective)

# Training params
n_epochs = 50
n_iter_D = 10

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # update Discriminator
            sess.run(D_training_op, feed_dict{<FILL-IN>})
            if iteration % n_iter_D == 0:
                # update Generator
                sess.run(G_training_op, feed_dict{<FILL-IN>})


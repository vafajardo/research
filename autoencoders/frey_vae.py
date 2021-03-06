import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os
import sys
from plotting import *
from dataReader import data_reader

# configure matplotlib
plt.rcParams['figure.figsize'] = (13.5, 13.5) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

"""
Training a VAE on frey dataset
"""

def show_examples(data, n=None, n_cols=20, thumbnail_cb=None):
    if n is None:
        n = len(data)
    n_rows = int(np.ceil(n / float(n_cols)))
    figure = np.zeros((img_rows * n_rows, img_cols * n_cols))
    for k, x in enumerate(data[:n]):
        r = k // n_cols
        c = k % n_cols
        figure[r * img_rows: (r + 1) * img_rows,
               c * img_cols: (c + 1) * img_cols] = x
        if thumbnail_cb is not None:
            thumbnail_cb(locals())

    plt.figure(figsize=(12, 10))
    plt.imshow(figure)
    plt.axis("off")
    plt.tight_layout()

# setup logdir
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# Loading Frey dataset
data_path = '/home/andrei/ml/datasets/'
ff = loadmat(data_path + 'frey_rawface.mat', squeeze_me=True, struct_as_record=False)
ff = ff["ff"].T # loadmat loads data as a dict, and we also need to transpose the data


# Construction phase
n_inputs = 28*20
n_hidden1 = 200
n_hidden2 = 200
n_hidden3 = 20 # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs

with tf.contrib.framework.arg_scope(
        [fully_connected],
        activation_fn = tf.nn.elu,
        weights_initializer = tf.contrib.layers.variance_scaling_initializer()):
    X = tf.placeholder(tf.float32, shape = [None, n_inputs], name="X")
    with tf.name_scope("X_encoder"):
        hidden1 = fully_connected(X, n_hidden1)
        hidden2 = fully_connected(hidden1, n_hidden2, activation_fn=tf.nn.tanh)
        hidden2_mean = fully_connected(hidden1, n_hidden2)
        hidden3_mean = fully_connected(hidden2, n_hidden3, activation_fn=None)
        hidden3_gamma = fully_connected(hidden2, n_hidden3, activation_fn=None)
        hidden3_sigma = tf.exp(0.5 * hidden3_gamma)
    noise1 = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)
    hidden3 = hidden3_mean + hidden3_sigma * noise1 # encodings
    with tf.name_scope("X_decoder"):
        hidden4 = fully_connected(hidden3, n_hidden4)
        hidden5 = fully_connected(hidden4, n_hidden5, activation_fn=tf.nn.tanh)
        hidden6_mean = fully_connected(hidden5, n_outputs, activation_fn=None)
        hidden6_gamma = fully_connected(hidden5, n_outputs, activation_fn=None)
        hidden6_sigma = tf.exp(0.5 * hidden6_gamma)
    noise2 = tf.random_normal(tf.shape(hidden6_sigma),stddev=2.0, dtype=tf.float32)
    outputs = hidden6_mean + hidden6_sigma * noise2

eps = 1e-10
with tf.name_scope("ELB"):
    latent_loss = 0.5 * tf.reduce_sum(
        tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)
    reconstruction_loss = 0.5 * tf.reduce_sum(tf.square((X - hidden6_mean) / (eps + tf.exp(hidden6_gamma)))
                                      + tf.log(2*np.pi) + hidden6_gamma) # log Normal
    cost = reconstruction_loss + latent_loss

# Setting up the optimizer
learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(cost)

cost_summary = tf.summary.scalar('ELB', cost)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
init = tf.global_variables_initializer()
# saver = tf.train.Saver()

# Training/Testing params
n_epochs = 10000
batch_size = 100
n_faces = 100
ff_reader = data_reader(ff, batch_size)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = ff_reader.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch = ff_reader.next_batch().astype(np.float32)
            sess.run(training_op, feed_dict={X: X_batch})
        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([cost,
         reconstruction_loss, latent_loss], feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Total loss:", loss_val / X_batch.shape[0],
         "\tReconstruction loss:", reconstruction_loss_val / X_batch.shape[0],
          "\tLatent loss:", latent_loss_val / X_batch.shape[0])

    # generating digits
    codings_rnd = np.random.normal(size=[n_faces, n_hidden3])
    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})

file_writer.close() # not really logging any cost summary, just wanting to get visual of graph

# Plotting the generated digits
img_rows, img_cols = 28, 20
show_examples(outputs_val.reshape((-1, img_rows, img_cols)), n=100, n_cols=25)
save_fig("generated_frey_faces")

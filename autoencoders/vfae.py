import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os
import sys
import matplotlib
matplotlib.use('Agg')
from plotting import *


"""
Implementation of a Variational Fair Autoencoder. With no MMD just yet.
"""

def show_reconstructed_digits(X, outputs, model_path = None, n_test_digits = 2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])

# setup logdir
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# Loading MNIST data
mnist = input_data.read_data_sets("/tmp/data/")

# Construction phase
n_s = 10
n_inputs = 28*28 - n_s
# encoders
n_hidden1 = 500
n_hidden2 = 20 # codings
n_hidden3 = 500
n_hidden4 = 20
# decoders
n_hidden5 = 500
n_hidden6 = 20
n_hidden7 = 500
# n_hidden8 = 20
# final output can take a random sample from the posterior
n_outputs = n_inputs + n_s

alpha = 1
learning_rate = 0.001

with tf.contrib.framework.arg_scope(
        [fully_connected],
        activation_fn = tf.nn.elu,
        weights_initializer = tf.contrib.layers.variance_scaling_initializer()):
    X = tf.placeholder(tf.float32, shape = [None, n_inputs], name="X_wo_s")
    s = tf.placeholder(tf.float32, shape = [None, n_s], name="s")
    X_full = tf.concat([X,s], axis=1)
    y = tf.placeholder(tf.int32, shape = [None, 1], name="y")
    is_unlabelled = tf.placeholder(tf.bool, shape=(), name='is_training')
    with tf.name_scope("X_encoder"):
        hidden1 = fully_connected(tf.concat([X, s], axis=1), n_hidden1)
        hidden2_mean = fully_connected(hidden1, n_hidden2, activation_fn = None)
        hidden2_gamma = fully_connected(hidden1, n_hidden2, activation_fn = None)
        hidden2_sigma = tf.exp(0.5 * hidden2_gamma)
    noise1 = tf.random_normal(tf.shape(hidden2_sigma), dtype=tf.float32)
    hidden2 = hidden2_mean + hidden2_sigma * noise1         # z1
    with tf.name_scope("Z1_encoder"):
        hidden3_ygz1 = fully_connected(hidden2, n_hidden4, activation_fn = tf.nn.tanh)
        hidden4_softmax_mean = fully_connected(hidden3_ygz1, 10, activation_fn = tf.nn.softmax)
        if is_unlabelled == True:
            # impute by sampling from q(y|z1)
            y = tf.assign(y, tf.multinomial(hidden4_softmax_mean, 1,
                                output_type = tf.int32))
        hidden3 = fully_connected(tf.concat([hidden2, tf.cast(y, tf.float32)], axis=1),
                        n_hidden3, activation_fn=tf.nn.tanh)
        hidden4_mean = fully_connected(hidden3, n_hidden4, activation_fn = None)
        hidden4_gamma = fully_connected(hidden3, n_hidden4, activation_fn = None)
        hidden4_sigma = tf.exp(0.5 * hidden4_gamma)
    noise2 = tf.random_normal(tf.shape(hidden4_sigma), dtype=tf.float32)
    hidden4 = hidden4_mean + hidden4_sigma * noise2     # z2
    with tf.name_scope("Z1_decoder"):
        hidden5 = fully_connected(tf.concat([hidden4, tf.cast(y, tf.float32)], axis=1 ),
                    n_hidden5, activation_fn = tf.nn.tanh)
        hidden6_mean = fully_connected(hidden5, n_hidden6, activation_fn = None)
        hidden6_gamma = fully_connected(hidden5, n_hidden6, activation_fn = None)
        hidden6_sigma = tf.exp(0.5 * hidden6_gamma)
    noise3 = tf.random_normal(tf.shape(hidden6_sigma), dtype=tf.float32)
    hidden6 = hidden6_mean + hidden6_sigma * noise3     # z1 (decoded)
    with tf.name_scope("X_decoder"):
        hidden7 = fully_connected(tf.concat([hidden6, s], axis=1), n_hidden7,
                                 activation_fn = tf.nn.tanh)
        hidden8 = fully_connected(hidden7, n_outputs, activation_fn = None)
    outputs = tf.sigmoid(hidden8, name="decoded_X")

# expected lower bound
with tf.name_scope("ELB"):
    kl_z2 = 0.5 * tf.reduce_sum(
                    tf.exp(hidden4_gamma)
                    + tf.square(hidden4_mean)
                    - 1
                    - hidden4_gamma
                    )

    kl_z1 = 0.5 * (tf.reduce_sum(
                    (1 / (1e-10 + tf.exp(hidden6_gamma))) * tf.exp(hidden2_gamma)
                    - 1
                    + hidden6_gamma
                    - hidden2_gamma
                    ) + tf.einsum('ij,ji -> i',
                        (hidden6_mean-hidden2_mean) * (1 / (1e-10 + tf.exp(hidden6_gamma))),
                        tf.transpose((hidden6_mean-hidden2_mean))))

    indices = tf.range(tf.shape(y)[0])
    indices = tf.concat([indices[:, tf.newaxis], y], axis=1)
    eps = 1e-10
    log_q_y_z1 = tf.reduce_sum(tf.log(eps + tf.gather_nd(hidden4_softmax_mean, indices)))

    # Bernoulli log-likelihood
    reconstruction_loss = -(tf.reduce_sum(X_full * tf.log(outputs)
                            + (1 - X_full) * tf.log(1 - outputs)))
    cost = kl_z2 + kl_z1 + reconstruction_loss + alpha * log_q_y_z1

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(cost)

cost_summary = tf.summary.scalar('ELB', cost)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
init = tf.global_variables_initializer()
# saver = tf.train.Saver()

# Training
n_epochs = 50
batch_size = 100
n_digits = 60

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch[:,:-n_s],
                                    s: X_batch[:,-n_s:],
                                    y: y_batch[:,np.newaxis],
                                    is_unlabelled: False})
        kl_z2_val, kl_z1_val, log_q_y_z1_val, reconstruction_loss_val, loss_val = sess.run([
                kl_z2,
                kl_z1,
                log_q_y_z1,
                reconstruction_loss,
                cost],
                feed_dict={X: X_batch[:,:-n_s],
                        s: X_batch[:,-n_s:],
                        y: y_batch[:,np.newaxis]})
        print("\r{}".format(epoch), "Train total loss:", loss_val,
         "\tReconstruction loss:", reconstruction_loss_val,
          "\tKL-z1:", kl_z1_val,
          "\tKL-z2:", kl_z2_val,
          "\tlog_q(y|z1):", log_q_y_z1_val)
        # saver.save(sess, "./my_model_all_layers.ckpt")

    # generating digits
    codings_rnd = np.random.normal(scale=2,size=[n_digits, n_hidden2])
    s_rnd = X_batch[:n_digits, -n_s:]
    # s_rnd = np.random.choice(2, size=[n_digits, n_s], p = [0.98, 0.02])
    # codings_rnd = X_batch[:n_digits, :-n_s]
    outputs_val = outputs.eval(feed_dict={hidden2: codings_rnd,
                                        s: s_rnd,
                                        y: np.zeros((n_digits,1), dtype=np.int32),
                                        is_unlabelled: True})

file_writer.close()

# Plotting the generated digits
n_rows = 6
n_cols = 10
plot_multiple_images(outputs_val.reshape(-1, 28, 28), n_rows, n_cols)
save_fig("vfae_generated_digits_plot_s{}".format(n_s))
plt.show()

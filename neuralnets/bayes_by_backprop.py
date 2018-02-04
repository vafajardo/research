import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")

def neurons_bayes(X, n_neurons, name, mu, rho, noise, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        W = tf.Variable(mu + tf.log(1 + tf.exp(rho))*noise,
                       name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X,W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

# Network architecture
n_inputs = 28*28
n_hidden1 = 400
n_hidden2 = 400
n_outputs = 10

tf.reset_default_graph()

# variational posterior parameters
mu1 = tf.Variable(tf.random_uniform([n_inputs, n_hidden1]), name="mu1")
mu2 = tf.Variable(tf.random_uniform([n_hidden1, n_hidden2]), name="mu2")
mu3 = tf.Variable(tf.random_uniform([n_hidden2, n_outputs]), name="mu3")
rho1 = tf.Variable(tf.random_uniform([n_inputs, n_hidden1], 0, 1.0), name="rho1")
rho2 = tf.Variable(tf.random_uniform([n_hidden1, n_hidden2], 0, 1.0), name="rho1")
rho3 = tf.Variable(tf.random_uniform([n_hidden2, n_outputs], 0, 1.0), name="rho1")

noise1 = tf.random_normal((n_inputs, n_hidden1), dtype=tf.float32)
noise2 = tf.random_normal((n_hidden1, n_hidden2), dtype=tf.float32)
noise3 = tf.random_normal((n_hidden2, n_outputs), dtype=tf.float32)

# defining layers
X = tf.placeholder(tf.float32, shape = [None, n_inputs])
y = tf.placeholder(tf.int64, shape = (None))
hidden1 = neurons_bayes(X, n_hidden1, "hidden1", mu1, rho1, noise1, "relu")
hidden2 = neurons_bayes(hidden1, n_hidden2, "hidden2", mu2, rho2, noise2, "relu")
logits = neurons_bayes(hidden2, n_outputs, "outputs", mu3, rho3, noise3)

# defining loss function
eps = 1e-10
weights = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) \
           if "weights" in v.name]
with tf.name_scope("loss"):
    log_xentropy = tf.reduce_mean(tf.log(eps + tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                 logits=logits)))

    log_prior_weights = [-v**2/(2*0.35**2) - tf.log(eps + tf.sqrt(2*np.pi*0.35**2)) for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) \
           if "weights" in v.name]
    log_prior = (tf.reduce_sum(log_prior_weights[0]) + tf.reduce_sum(log_prior_weights[1])
                + tf.reduce_sum(log_prior_weights[2]))

    sigma1 = tf.log(1+tf.exp(rho1))
    sigma2 = tf.log(1+tf.exp(rho2))
    sigma3 = tf.log(1+tf.exp(rho3))

    log_posterior_1 = -(weights[0] - mu1)**2 /(2*sigma1**2) - tf.log(eps + tf.sqrt(2*np.pi)*sigma1)
    log_posterior_2 = -(weights[1] - mu2)**2 /(2*sigma2**2) - tf.log(eps + tf.sqrt(2*np.pi)*sigma2)
    log_posterior_3 = -(weights[2] - mu3)**2 /(2*sigma3**2) - tf.log(eps + tf.sqrt(2*np.pi)*sigma3)
    log_variational_posterior = (tf.reduce_sum(log_posterior_1) +
                                 tf.reduce_sum(log_posterior_2) +
                                 tf.reduce_sum(log_posterior_3))

    # expected lower bound
    elb = log_variational_posterior - log_xentropy - log_prior

# optimizer
learning_rate = 10e-4
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(elb_mean)

# random sampling to get weights
weights = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) \
           if "weights" in v.name]

noise1 = tf.random_normal((n_inputs, n_hidden1), dtype=tf.float32)
noise2 = tf.random_normal((n_hidden1, n_hidden2), dtype=tf.float32)
noise3 = tf.random_normal((n_hidden2, n_outputs), dtype=tf.float32)

w1_reassign = tf.assign(weights[0], tf.ones((n_inputs, n_hidden1))*mu1 + tf.log(1 + tf.exp(rho1))*noise1)
w2_reassign = tf.assign(weights[1], tf.ones((n_hidden1, n_hidden2))*mu2 + tf.log(1 + tf.exp(rho2))*noise2)
w3_reassign = tf.assign(weights[2], tf.ones((n_hidden2, n_outputs))*mu3 + tf.log(1 + tf.exp(rho3))*noise3)

# evaluation
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

# Training
# training time
n_epochs = 400
batch_size = 128
num_samples = 5

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            for j in num_samples:
                sess.run([w1_reassign, w2_reassign, w3_reassign])
                elb_mean += elb.eval() / num_samples
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                y: mnist.test.labels})
        current_elb = elb.eval(feed_dict={X: X_batch, y:y_batch})
        print(epoch, "Train accuracy:", acc_train, "Test_accuracy:", acc_test, "ELB:", current_elb)

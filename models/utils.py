from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


def get_required_argument(dotmap, key, message, default=None):
    val = dotmap.get(key, default)
    if val is default:
        raise ValueError(message)
    return val


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


def batch_generator(index_array, batch_size):
    index_array = shuffle_rows(index_array)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= index_array.shape[1]:
            batch_count = 0
            index_array = shuffle_rows(index_array)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield index_array[:, start:end]


def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def mmd_ensemble(x_ensemble, y_ensemble, n_ensemble, kernel=gaussian_kernel_matrix):
    mmd_losses = []
    for i in range(n_ensemble):
        mmd_losses.append(maximum_mean_discrepancy(x_ensemble[i], y_ensemble[i], kernel))
    return tf.convert_to_tensor(mmd_losses)


class TensorStandardScaler:
    """Helper class for automatically normalizing inputs into the network.
    """

    def __init__(self, x_dim):
        """Initializes a scaler.

        Arguments:
        x_dim (int): The dimensionality of the inputs into the scaler.

        Returns: None.
        """
        self.fitted = False
        with tf.variable_scope("Scaler"):
            self.mu = tf.get_variable(
                name="scaler_mu", shape=[1, x_dim], initializer=tf.constant_initializer(0.0),
                trainable=False
            )
            self.sigma = tf.get_variable(
                name="scaler_std", shape=[1, x_dim], initializer=tf.constant_initializer(1.0),
                trainable=False
            )

        self.cached_mu, self.cached_sigma = np.zeros([0, x_dim]), np.ones([1, x_dim])

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.mu.load(mu)
        self.sigma.load(sigma)
        self.fitted = True
        self.cache()

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.sigma

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.sigma * data + self.mu

    def get_vars(self):
        """Returns a list of variables managed by this object.

        Returns: (list<tf.Variable>) The list of variables.
        """
        return [self.mu, self.sigma]

    def cache(self):
        """Caches current values of this scaler.

        Returns: None.
        """
        self.cached_mu = self.mu.eval()
        self.cached_sigma = self.sigma.eval()

    def load_cache(self):
        """Loads values from the cache

        Returns: None.
        """
        self.mu.load(self.cached_mu)
        self.sigma.load(self.cached_sigma)

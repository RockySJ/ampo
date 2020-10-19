'''
MMD functions implemented in tensorflow.
'''
from __future__ import division

_eps = 1.0e-5

import tensorflow as tf

mysqrt = lambda x: tf.sqrt(tf.maximum(x + _eps, 0.))


def dot(x, y, name=None):
    "The dot product of two vectors x and y."
    with tf.name_scope(name, "Dot", [x, y]):
        x = tf.convert_to_tensor(x, name='x')
        y = tf.convert_to_tensor(y, name='y')

        x.get_shape().assert_has_rank(1)
        y.get_shape().assert_has_rank(1)

        return tf.squeeze(tf.matmul(tf.expand_dims(x, 0), tf.expand_dims(y, 1)))


def sq_sum(t, name=None):
    "The squared Frobenius-type norm of a tensor, sum(t ** 2)."
    with tf.name_scope(name, "SqSum", [t]):
        t = tf.convert_to_tensor(t, name='t')
        return 2 * tf.nn.l2_loss(t)


def _distance_kernel(X, Y, K_XY_only=False):
    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XY = c(mysqrt(X_sqnorms)) + r(mysqrt(Y_sqnorms)) - mysqrt(-2 * XY + c(X_sqnorms) + r(Y_sqnorms))

    if K_XY_only:
        return K_XY

    K_XX = c(mysqrt(X_sqnorms)) + r(mysqrt(X_sqnorms)) - mysqrt(-2 * XX + c(X_sqnorms) + r(X_sqnorms))
    K_YY = c(mysqrt(Y_sqnorms)) + r(mysqrt(Y_sqnorms)) - mysqrt(-2 * YY + c(Y_sqnorms) + r(Y_sqnorms))

    return K_XX, K_XY, K_YY, False


def _tanh_distance_kernel(X, Y, K_XY_only=False):
    return _distance_kernel(tf.tanh(X), tf.tanh(Y), K_XY_only=K_XY_only)


def _dot_kernel(X, Y, K_XY_only=False):
    K_XY = tf.matmul(X, Y, transpose_b=True)
    if K_XY_only:
        return K_XY

    K_XX = tf.matmul(X, X, transpose_b=True)
    K_YY = tf.matmul(Y, Y, transpose_b=True)

    return K_XX, K_XY, K_YY, False


def _mix_rbf_kernel(X, Y, sigmas=[2.0, 5.0, 10.0, 20.0, 40.0, 80.0], wts=None,
                    K_XY_only=False):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0

    XYsqnorm = -2 * XY + c(X_sqnorms) + r(Y_sqnorms)
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma ** 2)
        K_XY += wt * tf.exp(-gamma * XYsqnorm)

    if K_XY_only:
        return K_XY

    XXsqnorm = -2 * XX + c(X_sqnorms) + r(X_sqnorms)
    YYsqnorm = -2 * YY + c(Y_sqnorms) + r(Y_sqnorms)
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma ** 2)
        K_XX += wt * tf.exp(-gamma * XXsqnorm)
        K_YY += wt * tf.exp(-gamma * YYsqnorm)

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


# def _mix_rq_dot_kernel(X, Y, alphas=[.1, 1., 10.], wts=None, K_XY_only=False):
#     return _mix_rq_kernel(X, Y, alphas=alphas, wts=wts, K_XY_only=K_XY_only, add_dot=.1)


def _mix_rq_1dot_kernel(X, Y, alphas=[0.1, 0.2, 0.5, 1., 2.5, 5., 10], wts=None, K_XY_only=False):
    return _mix_rq_kernel(X, Y, alphas=alphas, wts=wts, K_XY_only=K_XY_only, add_dot=1.)


def _mix_rq_10dot_kernel(X, Y, alphas=[.1, 1., 10.], wts=None, K_XY_only=False):
    return _mix_rq_kernel(X, Y, alphas=alphas, wts=wts, K_XY_only=K_XY_only, add_dot=10.)


def _mix_rq_01dot_kernel(X, Y, alphas=[.1, 1., 10.], wts=None, K_XY_only=False):
    return _mix_rq_kernel(X, Y, alphas=alphas, wts=wts, K_XY_only=K_XY_only, add_dot=.1)


def _mix_rq_001dot_kernel(X, Y, alphas=[.1, 1., 10.], wts=None, K_XY_only=False):
    return _mix_rq_kernel(X, Y, alphas=alphas, wts=wts, K_XY_only=K_XY_only, add_dot=.01)


def _tanh_mix_rq_kernel(X, Y, K_XY_only=False):
    return _mix_rq_kernel(tf.tanh(X), tf.tanh(Y), K_XY_only=K_XY_only)


def _mix_rq_kernel(X, Y, alphas=[.1, 1., 10.], wts=None, K_XY_only=False, add_dot=.0):
    """
    Rational quadratic kernel
    http://www.cs.toronto.edu/~duvenaud/cookbook/index.html
    """
    if wts is None:
        wts = [1.] * len(alphas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0., 0., 0.

    XYsqnorm = tf.maximum(-2. * XY + c(X_sqnorms) + r(Y_sqnorms), 0.)

    for alpha, wt in zip(alphas, wts):
        logXY = tf.log(1. + XYsqnorm / (2. * alpha))
        K_XY += wt * tf.exp(-alpha * logXY)
    if add_dot > 0:
        K_XY += tf.cast(add_dot, tf.float32) * XY

    if K_XY_only:
        return K_XY

    XXsqnorm = tf.maximum(-2. * XX + c(X_sqnorms) + r(X_sqnorms), 0.)
    YYsqnorm = tf.maximum(-2. * YY + c(Y_sqnorms) + r(Y_sqnorms), 0.)

    for alpha, wt in zip(alphas, wts):
        logXX = tf.log(1. + XXsqnorm / (2. * alpha))
        logYY = tf.log(1. + YYsqnorm / (2. * alpha))
        K_XX += wt * tf.exp(-alpha * logXX)
        K_YY += wt * tf.exp(-alpha * logYY)
    if add_dot > 0:
        K_XX += tf.cast(add_dot, tf.float32) * XX
        K_YY += tf.cast(add_dot, tf.float32) * YY

    # wts = tf.reduce_sum(tf.cast(wts, tf.float32))
    wts = tf.reduce_sum(tf.cast(wts, tf.float32))
    return K_XX, K_XY, K_YY, wts


def mmd2(K, biased=False):
    K_XX, K_XY, K_YY, const_diagonal = K
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal, biased)  # numerics checked at _mmd2 return


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    # m = tf.cast(K_XX.get_shape()[0], tf.float32)
    m = tf.cast(256, tf.float32)
    # n = tf.cast(K_YY.get_shape()[0], tf.float32)
    n = tf.cast(256, tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
                + tf.reduce_sum(K_YY) / (n * n)
                - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            const_diagonal = tf.cast(const_diagonal, tf.float32)
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
                + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
                - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2
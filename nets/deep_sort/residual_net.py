# vim: expandtab:ts=4:sw=4
import tensorflow as tf
from tensorflow.contrib import slim


def _batch_norm_fn(x, scope=None):
    if scope is None:
        scope = tf.get_variable_scope().name + "/bn"
    return slim.batch_norm(x, scope=scope)


def create_link(
        incoming, network_builder, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
        regularizer=None, is_first=False, summarize_activations=True):
    if is_first:
        network = incoming
    else:
        network = _batch_norm_fn(incoming, scope=scope + "/bn")
        network = nonlinearity(network)
        if summarize_activations:
            tf.summary.histogram(scope+"/activations", network)

    pre_block_network = incoming
    post_block_network = network_builder(network, scope)

    incoming_dim = pre_block_network.get_shape().as_list()[-1]
    outgoing_dim = post_block_network.get_shape().as_list()[-1]
    if incoming_dim != outgoing_dim:
        assert outgoing_dim == 2 * incoming_dim, \
            "%d != %d" % (outgoing_dim, 2 * incoming)
        projection = slim.conv2d(
            incoming, outgoing_dim, 1, 2, padding="SAME", activation_fn=None,
            scope=scope+"/projection", weights_initializer=weights_initializer,
            biases_initializer=None, weights_regularizer=regularizer)
        network = projection + post_block_network
    else:
        network = incoming + post_block_network
    return network


def create_inner_block(
        incoming, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(1e-3),
        bias_initializer=tf.zeros_initializer(), regularizer=None,
        increase_dim=False, summarize_activations=True):
    n = incoming.get_shape().as_list()[-1]
    stride = 1
    if increase_dim:
        n *= 2
        stride = 2

    incoming = slim.conv2d(
        incoming, n, [3, 3], stride, activation_fn=nonlinearity, padding="SAME",
        normalizer_fn=_batch_norm_fn, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/1")
    if summarize_activations:
        tf.summary.histogram(incoming.name + "/activations", incoming)

    incoming = slim.dropout(incoming, keep_prob=0.6)

    incoming = slim.conv2d(
        incoming, n, [3, 3], 1, activation_fn=None, padding="SAME",
        normalizer_fn=None, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/2")
    return incoming


def residual_block(incoming, scope, nonlinearity=tf.nn.elu,
                   weights_initializer=tf.truncated_normal_initializer(1e3),
                   bias_initializer=tf.zeros_initializer(), regularizer=None,
                   increase_dim=False, is_first=False,
                   summarize_activations=True):

    def network_builder(x, s):
        return create_inner_block(
            x, s, nonlinearity, weights_initializer, bias_initializer,
            regularizer, increase_dim, summarize_activations)

    return create_link(
        incoming, network_builder, scope, nonlinearity, weights_initializer,
        regularizer, is_first, summarize_activations)

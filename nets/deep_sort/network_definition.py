# vim: expandtab:ts=4:sw=4
import tensorflow as tf
import tensorflow.contrib.slim as slim

from . import residual_net


def create_network(images, num_classes=None, add_logits=True, reuse=None,
                   create_summaries=True, weight_decay=1e-8):
    nonlinearity = tf.nn.elu
    conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = slim.l2_regularizer(weight_decay)
    fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    fc_bias_init = tf.zeros_initializer()
    fc_regularizer = slim.l2_regularizer(weight_decay)

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    network = images
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_1",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)
        tf.summary.image("conv1_1/weights", tf.transpose(
            slim.get_variables("conv1_1/weights:0")[0], [3, 0, 1, 2]),
                         max_outputs=128)
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_2",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)

    network = slim.max_pool2d(
        network, [3, 3], [2, 2], scope="pool1", padding="SAME")

    network = residual_net.residual_block(
        network, "conv2_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False, is_first=True,
        summarize_activations=create_summaries)
    network = residual_net.residual_block(
        network, "conv2_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_net.residual_block(
        network, "conv3_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_net.residual_block(
        network, "conv3_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_net.residual_block(
        network, "conv4_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_net.residual_block(
        network, "conv4_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    feature_dim = network.get_shape().as_list()[-1]
    print("feature dimensionality: ", feature_dim)
    network = slim.flatten(network)

    network = slim.dropout(network, keep_prob=0.6)
    network = slim.fully_connected(
        network, feature_dim, activation_fn=nonlinearity,
        normalizer_fn=batch_norm_fn, weights_regularizer=fc_regularizer,
        scope="fc1", weights_initializer=fc_weight_init,
        biases_initializer=fc_bias_init)

    features = network

    # Features in rows, normalize axis 1.
    features = tf.nn.l2_normalize(features, dim=1)

    if add_logits:
        with slim.variable_scope.variable_scope("ball", reuse=reuse):
            weights = slim.model_variable(
                "mean_vectors", (feature_dim, int(num_classes)),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale", (), tf.float32,
                initializer=tf.constant_initializer(0., tf.float32),
                regularizer=slim.l2_regularizer(1e-1))
            if create_summaries:
                tf.summary.scalar("scale", scale)
            scale = tf.nn.softplus(scale)

        # Mean vectors in colums, normalize axis 0.
        weights_normed = tf.nn.l2_normalize(weights, dim=0)
        logits = scale * tf.matmul(features, weights_normed)
    else:
        logits = None
    return features, logits


def create_network_factory(is_training, num_classes, add_logits,
                           weight_decay=1e-8, reuse=None):

    def factory_fn(image):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                     slim.batch_norm, slim.layer_norm],
                                    reuse=reuse):
                    features, logits = create_network(
                        image, num_classes=num_classes, add_logits=add_logits,
                        reuse=reuse, create_summaries=is_training,
                        weight_decay=weight_decay)
                    return features, logits

    return factory_fn


def preprocess(image, is_training=False, input_is_bgr=False):
    if input_is_bgr:
        image = image[:, :, ::-1]  # BGR to RGB
    image = tf.divide(tf.cast(image, tf.float32), 255.0)
    if is_training:
        image = tf.image.random_flip_left_right(image)
    return image

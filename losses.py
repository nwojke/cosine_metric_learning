# vim: expandtab:ts=4:sw=4
import tensorflow as tf


def _pdist(a, b=None):
    sq_sum_a = tf.reduce_sum(tf.square(a), reduction_indices=[1])
    if b is None:
        return -2 * tf.matmul(a, tf.transpose(a)) + \
            tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_a, (1, -1))
    sq_sum_b = tf.reduce_sum(tf.square(b), reduction_indices=[1])
    return -2 * tf.matmul(a, tf.transpose(b)) + \
        tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_b, (1, -1))


def softmargin_triplet_loss(features, labels, create_summaries=True):
    """Softmargin triplet loss.

    See::

        Hermans, Beyer, Leibe: In Defense of the Triplet Loss for Person
        Re-Identification. arXiv, 2017.

    Parameters
    ----------
    features : tf.Tensor
        A matrix of shape NxM that contains the M-dimensional feature vectors
        of N objects (floating type).
    labels : tf.Tensor
        The one-dimensional array of length N that contains for each feature
        the associated class label (integer type).
    create_summaries : Optional[bool]
        If True, creates summaries to monitor training behavior.

    Returns
    -------
    tf.Tensor
        A scalar loss tensor.

    """
    eps = tf.constant(1e-5, tf.float32)
    nil = tf.constant(0., tf.float32)
    almost_inf = tf.constant(1e+10, tf.float32)

    squared_distance_mat = _pdist(features)
    distance_mat = tf.sqrt(tf.maximum(nil, eps + squared_distance_mat))
    label_mat = tf.cast(tf.equal(
        tf.reshape(labels, (-1, 1)), tf.reshape(labels, (1, -1))), tf.float32)

    positive_distance = tf.reduce_max(label_mat * distance_mat, axis=1)
    negative_distance = tf.reduce_min(
        (label_mat * almost_inf) + distance_mat, axis=1)
    loss = tf.nn.softplus(positive_distance - negative_distance)
    if create_summaries:
        fraction_invalid_pdist = tf.reduce_mean(
            tf.cast(tf.less_equal(squared_distance_mat, -eps), tf.float32))
        tf.summary.scalar("fraction_invalid_pdist", fraction_invalid_pdist)

        fraction_active_triplets = tf.reduce_mean(
            tf.cast(tf.greater_equal(loss, 1e-5), tf.float32))
        tf.summary.scalar("fraction_active_triplets", fraction_active_triplets)

        embedding_squared_norm = tf.reduce_mean(
            tf.reduce_sum(tf.square(features), axis=1))
        tf.summary.scalar("mean squared feature norm", embedding_squared_norm)

        mean_distance = tf.reduce_mean(distance_mat)
        tf.summary.scalar("mean feature distance", mean_distance)

        mean_positive_distance = tf.reduce_mean(positive_distance)
        tf.summary.scalar("mean positive distance", mean_positive_distance)

        mean_negative_distance = tf.reduce_mean(negative_distance)
        tf.summary.scalar("mean negative distance", mean_negative_distance)

    return tf.reduce_mean(loss)


def magnet_loss(features, labels, margin=1.0, unique_labels=None):
    """Simple unimodal magnet loss.

    See::

        Rippel, Paluri, Dollar, Bourdev: Metric Learning With Adaptive
        Density Discrimination. ICLR, 2016.

    Parameters
    ----------
    features : tf.Tensor
        A matrix of shape NxM that contains the M-dimensional feature vectors
        of N objects (floating type).
    labels : tf.Tensor
        The one-dimensional array of length N that contains for each feature
        the associated class label (integer type).
    margin : float
        A scalar margin hyperparameter.
    unique_labels : Optional[tf.Tensor]
        Optional tensor of unique values in `labels`. If None given, computed
        from data.

    Returns
    -------
    tf.Tensor
        A scalar loss tensor.

    """
    nil = tf.constant(0., tf.float32)
    one = tf.constant(1., tf.float32)
    minus_two = tf.constant(-2., tf.float32)
    eps = tf.constant(1e-4, tf.float32)
    margin = tf.constant(margin, tf.float32)

    num_per_class = None
    if unique_labels is None:
        unique_labels, sample_to_unique_y, num_per_class = tf.unique_with_counts(labels)
        num_per_class = tf.cast(num_per_class, tf.float32)

    y_mat = tf.cast(tf.equal(
        tf.reshape(labels, (-1, 1)), tf.reshape(unique_labels, (1, -1))),
        dtype=tf.float32)

    # If class_means is None, compute from batch data.
    if num_per_class is None:
        num_per_class = tf.reduce_sum(y_mat, reduction_indices=[0])
    class_means = tf.reduce_sum(
        tf.expand_dims(tf.transpose(y_mat), -1) * tf.expand_dims(features, 0),
        reduction_indices=[1]) / tf.expand_dims(num_per_class, -1)

    squared_distance = _pdist(features, class_means)

    num_samples = tf.cast(tf.shape(labels)[0], tf.float32)
    variance = tf.reduce_sum(
        y_mat * squared_distance) / (num_samples - one)

    const = one / (minus_two * (variance + eps))
    linear = const * squared_distance - y_mat * margin

    maxi = tf.reduce_max(linear, reduction_indices=[1], keepdims=True)
    loss_mat = tf.exp(linear - maxi)

    a = tf.reduce_sum(y_mat * loss_mat, reduction_indices=[1])
    b = tf.reduce_sum((one - y_mat) * loss_mat, reduction_indices=[1])
    loss = tf.maximum(nil, -tf.log(eps + a / (eps + b)))
    return tf.reduce_mean(loss), class_means, variance

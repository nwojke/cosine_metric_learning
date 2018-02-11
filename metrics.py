# vim: expandtab:ts=4:sw=4
import tensorflow as tf
import tensorflow.contrib.slim as slim


def pdist(a, b=None):
    """Compute element-wise squared distance between `a` and `b`.

    Parameters
    ----------
    a : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.
    b : tf.Tensor
        A matrix of shape MxL with M row-vectors of dimensionality L.

    Returns
    -------
    tf.Tensor
        A matrix of shape NxM where element (i, j) contains the squared
        distance between elements `a[i]` and `b[j]`.

    """
    sq_sum_a = tf.reduce_sum(tf.square(a), reduction_indices=[1])
    if b is None:
        return -2 * tf.matmul(a, tf.transpose(a)) + \
            tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_a, (1, -1))
    sq_sum_b = tf.reduce_sum(tf.square(b), reduction_indices=[1])
    return -2 * tf.matmul(a, tf.transpose(b)) + \
        tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_b, (1, -1))


def cosine_distance(a, b=None):
    """Compute element-wise cosine distance between `a` and `b`.

    Parameters
    ----------
    a : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.
    b : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.

    Returns
    -------
    tf.Tensor
        A matrix of shape NxM where element (i, j) contains the cosine distance
        between elements `a[i]` and `b[j]`.

    """
    a_normed = tf.nn.l2_normalize(a, dim=1)
    b_normed = a_normed if b is None else tf.nn.l2_normalize(b, dim=1)
    return (
        tf.constant(1.0, tf.float32) -
        tf.matmul(a_normed, tf.transpose(b_normed)))


def recognition_rate_at_k(probe_x, probe_y, gallery_x, gallery_y, k,
                          measure=pdist):
    """Compute the recognition rate at a given level `k`.

    For a given probe and ranked gallery that is sorted according to a distance
    measure `measure` in descending order, the recognition rate at `k` is::

        recognition_rate_at_k = num_correct / min(k, num_relevant)

    where num_correct refers to the fraction of images in the top k entries of
    the ranked gallery that have the same label as the probe and `num_relevant`
    refers to the total number of elements in the gallery that have the same
    label.

    Parameters
    ----------
    probe_x: tf.Tensor
        A tensor of probe images.
    probe_y: tf.Tensor
        A tensor of probe labels.
    gallery_x: tf.Tensor
        A tensor of gallery images.
    gallery_y: tf.Tensor
        A tensor of gallery labels.
    k: int
        See description above.
    measure: Callable[tf.Tensor, tf.Tensor] -> tf.Tensor
        A callable that computes for two matrices of row-vectors a matrix of
        element-wise distances. See `pdist` for an example.

    Returns
    -------
    tf.Tensor
        Returns a scalar tensor which represents the computed metric.

    """
    # Build a matrix of shape (num_probes, num_gallery_images) where element
    # (i, j) is 1 if probe image i and the gallery image j have the same
    # identity, otherwise 0.
    label_eq_mat = tf.cast(tf.equal(tf.reshape(
        probe_y, (-1, 1)), tf.reshape(gallery_y, (1, -1))),
        tf.float32)

    # For each probe image, compute the number of relevant images in the
    # gallery (same identity). This should always be one for CMC evaluation
    # because we always have exactly one probe and one gallery image for each
    # identity.
    num_relevant = tf.minimum(tf.cast(k, tf.float32), tf.reduce_sum(
        label_eq_mat, reduction_indices=[1]))

    # Rank gallery images by the similarity measure to build a matrix of
    # shape (num_probes, k) where element (i, j) contains the label of the
    # j-th ranked gallery image for probe i.
    predictions = tf.exp(-measure(probe_x, gallery_x))  # Compute similarity.
    _, prediction_indices = tf.nn.top_k(predictions, k=k)
    label_mat = tf.gather(gallery_y, prediction_indices)

    # Just as we have done before, build a matrix where element (i, j) is
    # one if probe i and gallery image j share the same label (same identity).
    # This time, the matrix is ranked by the similarity measure and we only
    # keep the top-k predictions.
    label_eq_mat = tf.cast(tf.equal(
        label_mat, tf.reshape(probe_y, (-1, 1))), tf.float32)

    # Compute the number of true positives in [0, k[, i.e., check if we find
    # the correct gallery image within the top-k ranked results. Then, compute
    # the recognition rate, which in our case is either 0 or 1 since we have
    # only one gallery image that shares the same identity with the probe.
    #
    # This is the final output of our CMC metric.
    true_positives_at_k = tf.reduce_sum(label_eq_mat, reduction_indices=[1])
    return true_positives_at_k / num_relevant


def streaming_mean_cmc_at_k(probe_x, probe_y, gallery_x, gallery_y, k,
                            measure=pdist):
    """Compute cumulated matching characteristics (CMC) at level `k` over
    a stream of data (i.e., multiple batches).

    The function is compatible with TensorFlow-Slim's streaming metrics
    interface, e.g., `slim.metrics.aggregate_metric_map`.

    Parameters
    ----------
    probe_x: tf.Tensor
        A tensor of probe images.
    probe_y: tf.Tensor
        A tensor of probe labels.
    gallery_x: tf.Tensor
        A tensor of gallery images.
    gallery_y: tf.Tensor
        A tensor of gallery labels.
    k: int
        See description above.
    measure: Callable[tf.Tensor, tf.Tensor] -> tf.Tensor
        A callable that computes for two matrices of row-vectors a matrix of
        element-wise distances. See `pdist` for an example.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        The first element in the tuple is the current result. The second element
        is an operation that updates the computed metric based on new data.

    """
    recognition_rate = recognition_rate_at_k(
        probe_x, probe_y, gallery_x, gallery_y, k, measure)
    return slim.metrics.streaming_mean(recognition_rate)


def streaming_mean_averge_precision(probe_x, probe_y, gallery_x, gallery_y,
                                    good_mask, measure=pdist):
    """Compute mean average precision (mAP) over a stream of data.

    Parameters
    ----------
    probe_x: tf.Tensor
        A tensor of N probe images.
    probe_y: tf.Tensor
        A tensor of N probe labels.
    gallery_x: tf.Tensor
        A tensor of M gallery images.
    gallery_y: tf.Tensor
        A tensor of M gallery labels.
    good_mask: Optional[tf.Tensor]
        A matrix of shape NxM where element (i, j) evaluates to 0.0 if the pair
        of i-th probe and j-th gallery image should be excluded from metric
        computation. All other elements should evaluate to 1.0.
    measure: Callable[tf.Tensor, tf.Tensor] -> tf.Tensor
        A callable that computes for two matrices of row-vectors a matrix of
        element-wise distances. See `pdist` for an example.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        The first element in the tuple is the current result. The second element
        is an operation that updates the computed metric based on new data.

    """
    # See Wikipedia:
    # https://en.wikipedia.org/wiki/Information_retrieval#Average_precision
    if good_mask.dtype != tf.float32:
        good_mask = tf.cast(good_mask, tf.float32)

    # Compute similarity measure and mask out diagonal (similarity to self).
    predictions = good_mask * tf.exp(-measure(probe_x, gallery_x))

    # Compute matrix of predicted labels.
    k = tf.shape(gallery_y)[0]
    _, prediction_indices = tf.nn.top_k(predictions, k=k)
    predicted_label_mat = tf.gather(gallery_y, prediction_indices)
    label_eq_mat = tf.cast(tf.equal(
        predicted_label_mat, tf.reshape(probe_y, (-1, 1))), tf.float32)

    # Compute statistics.
    num_relevant = tf.reduce_sum(
        good_mask * label_eq_mat, reduction_indices=[1], keep_dims=True)
    true_positives_at_k = tf.cumsum(label_eq_mat, axis=1)
    retrieved_at_k = tf.cumsum(tf.ones_like(label_eq_mat), axis=1)
    precision_at_k = true_positives_at_k / retrieved_at_k
    relevant_at_k = label_eq_mat
    average_precision = (
        tf.reduce_sum(precision_at_k * relevant_at_k, reduction_indices=[1]) /
        tf.cast(tf.squeeze(num_relevant), tf.float32))

    return slim.metrics.streaming_mean(average_precision)

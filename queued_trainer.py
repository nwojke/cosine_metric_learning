# vim: expandtab:ts=4:sw=4
import string
import os
import threading
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim


def run_in_batches(f, data_dict, out, batch_size):
    """Process data in batches.

    Parameters
    ----------
    f : Callable[Dict[tf.Tensor, np.ndarray] -> np.ndarray
        A function that maps a given input (one or multiple inpu arrays) to a
        single output array.
    data_dict : Dict[tf.Tensor, np.ndarray]
        Maps from symbolic input tensor to numpy data array.
    out : np.ndarray
        The computed function output will be stored in this array; must be have
        compatible shape and length to the output computed by `f`.
    batch_size : int
        The number of samples to compute in each call to `f`. If the length of
        the input array is not divisible by the batch size, the final call to
        `f` contains fewer examples.

    """
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    def pad(x):
        x = np.asarray(x)
        y = np.full((batch_size, ) + x.shape[1:], x[0], dtype=x.dtype)
        y[:x.shape[0]] = x
        return y

    s, e = 0, batch_size
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        remaining_len = len(out) - e
        batch_data_dict = {k: pad(v[e:]) for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)[:remaining_len]


def iterate_forever(batch_size, data, *other_data):
    """Iterate over dataset indefenitely.

    Parameters
    ----------
    batch_size : int
        The batch size.
    data : ndarray
        The first input array.
    other_data
        Additional input arrays; must be of type np.ndarray.

    Returns
    -------
    List[np.ndarray]
        A dataset batch. The length of each entry in the list is `batch_size`.

    """
    data_len = len(data)
    num_batches = int(data_len / batch_size)

    while True:
        data_list = [data] + list(other_data)
        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            batch = [x[s:e] for x in data_list]
            yield batch[0] if len(batch) == 1 else batch
        if e < data_len:
            batch = [x[e:] for x in data_list]
            yield batch[0] if len(batch) == 1 else batch


def random_shuffle_forever(batch_size, data, *other_data):
    """A generator that randomly selects `batch_size` entries from the data.

    Parameters
    ----------
    batch_size : int
        The batch size.
    data : np.ndarray
        The first input array.
    other_data
        Additional input arrays; must be of type np.ndarray

    Returns
    -------
    List[np.ndarray]
        A batch of randomly selected entries. The length of each entry in the
        list is `batch_size`.

    """
    data_list = [data] + list(other_data)
    indices = np.arange(len(data))
    while True:
        batch_indices = np.random.choice(indices, batch_size, replace=False)
        batch = [x[batch_indices] for x in data_list]
        yield batch[0] if len(batch) == 1 else batch


def random_sample_identities_forever(batch_size, num_samples_per_id, data_x,
                                     data_y, num_fa_images=0):
    """A generator that randomly selects a fixed number of entries per label.

    If false alarms are passed into this function, they should have a negative
    label, i.e., `data_y[i] < 0` if the i-th example corresponds to a false
    alarm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_samples_per_id : int
        Number of examples per label in each batch. If the `batch_size` is not
        divisible by `num_samples_per_id` then the batch is filled with false
        alarms. A warning is printed if no false alarms are available to fill
        up the batch.
    data_x : List[string] | np.ndarray
        The data array; either a list of filenames or a tensor of input images.
    data_y : List[int] | np.ndarray
        The label array (either as list of one-dimensional numpy array).
    num_fa_images : Optional[int]
        Number of false alarm images to include in each batch; defaults to zero.

    Returns
    -------
    List[np.ndarray]
        Returns a list of length two where the first entry is the data array
        corresponding to `data_x` and the second entry is the label array
        corresponding to `data_y`. The elements in the list are of length
        `batch_size`.

    """
    assert (batch_size - num_fa_images) % num_samples_per_id == 0
    num_ids_per_batch = int((batch_size - num_fa_images) / num_samples_per_id)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)

    unique_y = np.unique(data_y[data_y >= 0])
    y_to_idx = {y: np.where(data_y == y)[0] for y in unique_y}
    fa_indices = np.where(data_y < 0)[0]

    while True:
        # Draw the desired number of identities.
        indices = np.random.choice(
            len(unique_y), num_ids_per_batch, replace=False)
        batch_unique_y = unique_y[indices]

        batch_x = np.zeros((batch_size, ) + data_x.shape[1:], data_x.dtype)
        batch_y = np.zeros((batch_size, ), data_y.dtype)
        e = 0
        for i, y in enumerate(batch_unique_y):
            num_samples = min(num_samples_per_id, len(y_to_idx[y]))
            indices = np.random.choice(y_to_idx[y], num_samples, replace=False)
            s, e = e, e + num_samples
            batch_x[s:e] = data_x[indices]
            batch_y[s:e] = y

        # Fill up remaining space with false alarms.
        num_samples = len(batch_x) - e
        if num_fa_images > 0:
            num_batch_fa_samples = min(num_samples, len(fa_indices))
            indices = np.random.choice(
                fa_indices, num_batch_fa_samples, replace=False)
            s, e = e, e + num_batch_fa_samples
            batch_x[s:e] = data_x[indices]
            batch_y[s:e] = data_y[indices]

        # If we need to add more data, random sample ids until we have reached
        # the batch size.
        num_samples = len(batch_x) - e
        num_tries = 0
        while num_samples > 0 and num_tries < 100:
            y = np.random.choice(unique_y)
            if y in batch_unique_y:
                # Find a target that we have not yet in this batch.
                num_tries += 1
                continue

            num_samples = min(num_samples, len(y_to_idx[y]))
            indices = np.random.choice(y_to_idx[y], num_samples, replace=False)
            s, e = e, e + num_samples
            batch_x[s:e] = data_x[indices]
            batch_y[s:e] = y
            num_samples = len(batch_x) - e

        if e < batch_size:
            print("ERROR: Failed to sample a full batch. Adding corrupt data.")
        yield [batch_x, batch_y]


def _truncate_dataset_to_batch_size(batch_size, data, *other_data):
    """Truncate given input data to a multiple of the batch size.

    Parameters
    ----------
    batch_size : int
        The batch size. The length of the truncated data is a multiple of this
        value.
    data : np.ndarray
        The first input array.
    *other_data
        Additional input arrays; must be of type np.ndarray.

    Returns
    -------
    List[np.ndarray]
        The truncated data. The length of each entry in the list is a  multiple
        of the batch size.

    """
    num_batches = int(len(data) / batch_size)
    new_len = num_batches * batch_size
    dataset = [data] + list(other_data)
    if new_len < len(data):
        print(
            "WARNING dataset length is not a multiple of batch size. "
            "Truncating from %d to %d." % (len(data), new_len))
        dataset = [x[:new_len] for x in dataset]
    return num_batches, dataset[0] if len(dataset) == 1 else dataset


def _generate_run_id(size=6, chars=None):
    """Generate a random ID of length `size`.

    Parameters
    ----------
    size : int
    chars : Optional[str]
        Optional list of characters to use for generating the ID.

    Returns
    -------
    str
        Returns a random identifier of length `size`.

    """
    if chars is None:
        chars = string.ascii_uppercase + string.digits
    import random
    return ''.join(random.choice(chars) for _ in range(size))


class ThreadSafeIterator(object):
    """
    This class wraps an iterator (or generator) such that only one thread at a
    time is granted access.

    Parameters
    ----------
    iterator_or_generator
        An iterator or generator to be wrapped.

    """

    def __init__(self, iterator_or_generator):
        self._iterator_or_generator = iterator_or_generator
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return next(self._iterator_or_generator)

    def next(self):
        with self._lock:
            return self._iterator_or_generator.next()


class QueuedTrainer(object):
    """
    This class implements code to train and evaluate TensorFlow models based on
    TensorFlow-Slim. Image loading and preprocessing is de-coupled from the
    training steps using a tf.FIFOQueue.

    Parameters
    ----------
    enqueue_vars : List[tf.Tensor]
        A list of tensors to be enqueued; usually the labels and preprocessed
        images.
    input_vars : Optional[List[tf.Tensor]]
        An optional list of input tensors; usually the labels and raw (not
        preprocessed) images or filenames to the images. The list must be of the
        same length as the `enqueue_vars` and there must be a one-to-one
        correspondence, i.e., the i-th element in `enqueue_vars` is i-th
        preprocessed element in `input_vars`. If None, the input_vars are set to
        `enqueue_vars`.
    num_enqueue_threads : Optional[int]
        Number of threads used to preprocess data in parallel.
    queue_capacity : Optional[int]
        Maximum number of elements in the queue; defaults to 512.

    """

    def __init__(self, enqueue_vars, input_vars=None, num_enqueue_threads=4,
                 queue_capacity=512):
        if input_vars is None:
            input_vars = enqueue_vars
        self._input_vars = input_vars
        self._enqueue_vars = enqueue_vars

        shapes = [var.get_shape().as_list()[1:] for var in enqueue_vars]
        dtypes = [var.dtype for var in enqueue_vars]
        self._queue = tf.FIFOQueue(queue_capacity, dtypes, shapes)

        self._num_enqueue_threads = num_enqueue_threads
        self._enqueue_threads = []
        self._enqueue_op = self._queue.enqueue_many(self._enqueue_vars)
        self._stop_op = self._queue.close(cancel_pending_enqueues=True)
        self._coordinator = None

        self._feed_generator = None
        self._batch_size = None
        self._init_fns = []

    def get_input_vars(self, batch_size):
        """Get the top `batch_size` elements from the queue. The tensors
        returned by this functions should be passed on the the TensorFlow model.

        Parameters
        ----------
        batch_size : int
            The batch size.

        Returns
        -------
        List[tf.Tensor]
            Returns the top `batch_size` elements from the queue. There is a
            one-to-one correspondence between the `enqueue_vars` passed in to
            the constructor of this class and the tensors in the list returned
            by this function.

        """
        self._batch_size = batch_size
        return self._queue.dequeue_many(batch_size)

    def run(self, feed_generator, train_op, log_dir="/tmp/slim_trainer/",
            restore_path=None, variables_to_restore=None, run_id=None,
            max_checkpoints_to_keep=0, **kwargs):
        """ Run training.

        Parameters
        ----------
        feed_generator : Iterator[ndarray, ...]
            An iterator or generator that returns batches of training data; must
            return a one-to-one correspondence with the `enqueue_vars` passed
            to the constructor of this class.
        train_op : tf.Tensor
            The training operation created with `slim.learning.create_train_op`.
        log_dir : Optional[str]
            Path to TensorFlow log directory. This value is used in conjunction
            with `run_id` to generate the checkpoint and summary directory;
            defaults to '/tmp/slim_trainer'.
        restore_path : Optional[str]
            An optional checkpoint path. If not None, resumes training from the
            given checkpoint.
        variables_to_restore : Optional[List[str]]
            An optional list of variable scopes. If not None, only restores
            variables under the given scope. This value is ignored if
            `restore_path` is None.
        run_id : Optional[str]
            A string that identifies this training run. The checkpoints and
            TensorFlow summaries are stored in `log_dir/run_id`. If None, a
            random ID will be generated. Point tensorboard to this directory to
            monitor training progress.
        max_checkpoints_to_keep : int
            Keep only the `max_checkpoints_to_keep` newest checkpoints. If 0,
            keep all checkpoints.
        kwargs:
            Additional named arguments passed on to tf.slim.learning.train,
            e.g., `number_of_steps=100` to run 100 iterations of training.

        """
        if restore_path is not None:
            if variables_to_restore is None:
                variables_to_restore = slim.get_variables_to_restore()
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                restore_path, variables_to_restore)
            self._init_fns.append(lambda sess: sess.run(
                init_assign_op, init_feed_dict))
        self._feed_generator = ThreadSafeIterator(feed_generator)
        self._coordinator = tf.train.Coordinator()

        if run_id is None:
            run_id = _generate_run_id(6)
        log_dir = os.path.join(log_dir, run_id)
        print("---------------------------------------")
        print("Run ID: ", run_id)
        print("Log directory: ", log_dir)
        print("---------------------------------------")

        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
        try:
            slim.learning.train(
                train_op, log_dir, self._train_step_fn, saver=saver,
                **kwargs)
        except UnboundLocalError:
            # NOTE(nwojke): Due to a bug in slim, a local variable 'total_loss'
            # is referenced when an exception is raised during training. We
            # catch the exception here because it occurs whenever we close the
            # queue with self._stop_all_threads().
            pass
        self._wait_for_threads()

    def evaluate(self, dataset, checkpoint_dir, log_dir, run_id=None,
                 init_op=None, eval_op=None, final_op=None,
                 summary_op=None, variables_to_restore=None,
                 eval_interval_secs=60, max_num_evaluations=None):
        """Run evaluation. Monitors files in the log directory and computes
        evaluation metrics. This function must be called concurrently to
        training (in a separate process).

        WARNING: The dataset is truncated to the batch size. Thus, the computed
        metrics are only accurate if the dataset length is divisible by the
        batch size.

        Parameters
        ----------
        dataset : List[T]
            The dataset is a list (or tuple) of data arrays. The length of the
            list must be the same as the `input_vars` passed to the constructor
            of this class and there must be a one-to-one correspondence such
            that `dataset[i]` corresponds to the numeric data of its symbolic
            equivalent in `input_vars[i]`.
        checkpoint_dir : str
            The directory where checkpoints are stored. Should be set to
            `log_dir` of the training process.
        log_dir : str
            Path to TensorFlow log directory where evaluation logs will be
            stored. This directory should be different from the `log_dir`
            passed to `run`.
        run_id : Optional[str]
            A string that identifies the training runrun. Should be set to
            `run_id` passed to `run`.
        init_op : Optional[tf.Tensor]
            Optional operation to execute prior to processing the `dataset`.
        eval_op : Optional[tf.Tensor]
            Evaluation operation; will be executed for each batch in the
            `dataset`.
        final_op : Optional[tf.Tensor]
            Optional operation to execute after processing the `dataset`.
        summary_op : Optional[tf.Tensor]
            Summary operation; defaults to `tf.summary.merge_all()`.
        variables_to_restore : Optional[List[tf.Tensor]]
            List of variables to restore; defaults to
            `slim.get_variables_to_restore()`.
        eval_interval_secs : Optional[int]
            Poll the `checkpoint_dir` every `eval_interval_secs` seconds for
            new checkpoints.
        max_num_evaluations : Optional[int]
            Evaluate at most `max_num_evaluations` checkpoints.

        Returns
        -------
        T
            Returns the value of the last call to `final_op` or None.

        """
        if run_id is None:
            print("---------------------------------------")
            print("Checkpoint directory: ", checkpoint_dir)
            print("Log directory: ", log_dir)
            print("---------------------------------------")
        else:
            checkpoint_dir = os.path.join(checkpoint_dir, run_id)
            log_dir = os.path.join(log_dir, run_id)
            print("---------------------------------------")
            print("Run ID: ", run_id)
            print("Checkpoint directory: ", checkpoint_dir)
            print("Log directory: ", log_dir)
            print("---------------------------------------")

        if summary_op is None:
            summary_op = tf.summary.merge_all()

        global_step = tf.train.get_or_create_global_step()

        if variables_to_restore is None:
            variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        summary_writer = tf.summary.FileWriter(log_dir)
        sv = tf.train.Supervisor(
            graph=tf.get_default_graph(), logdir=log_dir, summary_op=None,
            summary_writer=None, global_step=None, saver=saver)

        print("Entering evaluation loop. Waiting for checkpoints.")
        num_batches, dataset = _truncate_dataset_to_batch_size(
            self._batch_size, *dataset)

        final_op_value = None
        num_evaluations = 0
        for checkpoint_path in slim.evaluation.checkpoints_iterator(
                checkpoint_dir, eval_interval_secs):
            with sv.managed_session(start_standard_services=False) as session:
                sv.saver.restore(session, checkpoint_path)
                sv.start_queue_runners(session)

                print("Starting evaluation of '%s'" % checkpoint_path)
                self._feed_generator = iterate_forever(
                    self._batch_size, *dataset)
                self._coordinator = tf.train.Coordinator()
                for fn in self._init_fns:
                    fn(session)
                self._start_enqueue(session, num_threads=1)

                if init_op is not None:
                    session.run(init_op)

                if eval_op is not None:
                    for i in range(num_batches):
                        session.run(eval_op)

                if final_op is not None:
                    final_op_value = session.run(final_op)
                else:
                    final_op_value = None

                summary_str = session.run(summary_op)
                global_step_value = session.run(global_step)
                summary_writer.add_summary(summary_str, global_step_value)
                summary_writer.flush()

                self._stop_all_threads(session)
                print("Finished evaluation of '%s'" % checkpoint_path)

            num_evaluations += 1
            if max_num_evaluations is not None \
                    and num_evaluations >= max_num_evaluations:
                break
        return final_op_value

    def _train_step_fn(self, session, train_op, global_step,
                       train_step_kwargs):
        if len(self._enqueue_threads) == 0:
            for fn in self._init_fns:
                fn(session)
            self._start_enqueue(session)
        total_loss, should_stop = slim.learning.train_step(
            session, train_op, global_step, train_step_kwargs)
        if should_stop or self._coordinator.should_stop():
            self._stop_all_threads(session)
        return total_loss, should_stop

    def _stop_all_threads(self, session):
        self._coordinator.request_stop()
        session.run(self._stop_op)  # Close the queue.

    def _wait_for_threads(self):
        self._coordinator.join(self._enqueue_threads)
        self._enqueue_threads = []

    def _start_enqueue(self, session, num_threads=None):
        if num_threads is None:
            num_threads = self._num_enqueue_threads
        for _ in range(num_threads):
            thread = threading.Thread(
                target=self._run_enqueue_thread, args=(session, ))
            thread.start()
            self._enqueue_threads.append(thread)

    def _run_enqueue_thread(self, session):
        try:
            for data in self._feed_generator:
                if self._coordinator.should_stop():
                    break
                try:
                    feed_dict = {
                        var: value for var, value in
                        zip(self._input_vars, data)}
                    session.run(self._enqueue_op, feed_dict=feed_dict)
                except (tf.errors.CancelledError, tf.errors.AbortedError):
                    # We have been requested to stop enqueuing data.
                    break
        except Exception as e:
            print("EnqueueError:", e)
            self._stop_all_threads(session)

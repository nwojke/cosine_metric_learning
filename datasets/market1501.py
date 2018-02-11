# vim: expandtab:ts=4:sw=4
import os
import numpy as np
import cv2
import scipy.io as sio


# The maximum person ID in the dataset.
MAX_LABEL = 1501

IMAGE_SHAPE = 128, 64, 3


def _parse_filename(filename):
    """Parse meta-information from given filename.

    Parameters
    ----------
    filename : str
        A Market 1501 image filename.

    Returns
    -------
    (int, int, str, str) | NoneType
        Returns a tuple with the following entries:

        * Unique ID of the individual in the image
        * Index of the camera which has observed the individual
        * Filename without extension
        * File extension

        Returns None if the given filename is not a valid filename.

    """
    filename_base, ext = os.path.splitext(filename)
    if '.' in filename_base:
        # Some images have double filename extensions.
        filename_base, ext = os.path.splitext(filename_base)
    if ext != ".jpg":
        return None
    person_id, cam_seq, frame_idx, detection_idx = filename_base.split('_')
    return int(person_id), int(cam_seq[1]), filename_base, ext


def read_train_split_to_str(dataset_dir):
    """Read training data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to the Market 1501 dataset directory.

    Returns
    -------
    (List[str], List[int], List[int])
        Returns a tuple with the following values:

        * List of image filenames (full path to image files).
        * List of unique IDs for the individuals in the images.
        * List of camera indices.

    """
    filenames, ids, camera_indices = [], [], []

    image_dir = os.path.join(dataset_dir, "bounding_box_train")
    for filename in sorted(os.listdir(image_dir)):
        meta_data = _parse_filename(filename)
        if meta_data is None:
            # This is not a valid filename (e.g., Thumbs.db).
            continue

        filenames.append(os.path.join(image_dir, filename))
        ids.append(meta_data[0])
        camera_indices.append(meta_data[1])

    return filenames, ids, camera_indices


def read_train_split_to_image(dataset_dir):
    """Read training images to memory. This consumes a lot of memory.

    Parameters
    ----------
    dataset_dir : str
        Path to the Market 1501 dataset directory.

    Returns
    -------
    (ndarray, ndarray, ndarray)
        Returns a tuple with the following values:

        * Tensor of images in BGR color space of shape 128x64x3.
        * One dimensional array of unique IDs for the individuals in the images.
        * One dimensional array of camera indices.

    """
    filenames, ids, camera_indices = read_train_split_to_str(dataset_dir)

    images = np.zeros((len(filenames), 128, 64, 3), np.uint8)
    for i, filename in enumerate(filenames):
        images[i] = cv2.imread(filename, cv2.IMREAD_COLOR)

    ids = np.asarray(ids, np.int64)
    camera_indices = np.asarray(camera_indices, np.int64)
    return images, ids, camera_indices


def read_test_split_to_str(dataset_dir):
    """Read query and gallery data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to the Market 1501 dataset directory.

    Returns
    -------
    (List[str], List[int], List[str], List[int], ndarray)
        Returns a tuple with the following values:

        * List of N gallery filenames (full path to image files).
        * List of N unique IDs for the individuals in the gallery.
        * List of M query filenames (full path to image files).
        * List of M unique IDs for the individuals in the queries.
        * Matrix of shape MxN such that element (i, j) evaluates to 0 if
          gallery image j should be excluded from metrics computation of
          query i and 1 otherwise.

    """
    # Read gallery.
    gallery_filenames, gallery_ids = [], []

    image_dir = os.path.join(dataset_dir, "bounding_box_test")
    for filename in sorted(os.listdir(image_dir)):
        meta_data = _parse_filename(filename)
        if meta_data is None:
            # This is not a valid filename (e.g., Thumbs.db).
            continue

        gallery_filenames.append(os.path.join(image_dir, filename))
        gallery_ids.append(meta_data[0])

    # Read queries.
    query_filenames, query_ids, query_junk_indices = [], [], []

    image_dir = os.path.join(dataset_dir, "query")
    for filename in sorted(os.listdir(image_dir)):
        meta_data = _parse_filename(filename)
        if meta_data is None:
            # This is not a valid filename (e.g., Thumbs.db).
            continue

        filename_base = meta_data[2]
        junk_matfile = filename_base + "_junk.mat"
        mat = sio.loadmat(os.path.join(dataset_dir, "gt_query", junk_matfile))
        if np.any(mat["junk_index"] < 1):
            indices = []
        else:
            # MATLAB to Python index.
            indices = list(mat["junk_index"].astype(np.int64).ravel() - 1)

        query_junk_indices.append(indices)
        query_filenames.append(os.path.join(image_dir, filename))
        query_ids.append(meta_data[0])

    # The following matrix maps from query (row) to gallery image (column) such
    # that element (i, j) evaluates to 0 if query i and gallery image j should
    # be excluded from computation of the evaluation metrics and 1 otherwise.
    good_mask = np.ones(
        (len(query_filenames), len(gallery_filenames)), np.float32)
    for i, junk_indices in enumerate(query_junk_indices):
        good_mask[i, junk_indices] = 0.

    return gallery_filenames, gallery_ids, query_filenames, query_ids, good_mask


def read_test_split_to_image(dataset_dir):
    """Read query and gallery data to memory. This consumes a lot of memory.

    Parameters
    ----------
    dataset_dir : str
        Path to the Market 1501 dataset directory.

    Returns
    -------
    (ndarray, ndarray, ndarray, ndarray, ndarray)
        Returns a tuple with the following values:

        * Tensor of shape Nx128x64x3 of N gallery images in BGR color space.
        * One dimensional array of N unique gallery IDs.
        * Tensor of shape Mx128x64x3 of M query images in BGR color space.
        * One dimensional array of M unique query IDs.
        * Matrix of shape MxN such that element (i, j) evaluates to 0 if
          gallery image j should be excluded from metrics computation of
          query i and 1 otherwise.

    """
    gallery_filenames, gallery_ids, query_filenames, query_ids, good_mask = (
        read_test_split_to_str(dataset_dir))

    gallery_images = np.zeros((len(gallery_filenames), 128, 64, 3), np.uint8)
    for i, filename in enumerate(gallery_filenames):
        gallery_images[i] = cv2.imread(filename, cv2.IMREAD_COLOR)

    query_images = np.zeros((len(query_filenames), 128, 64, 3), np.uint8)
    for i, filename in enumerate(query_filenames):
        query_images[i] = cv2.imread(filename, cv2.IMREAD_COLOR)

    gallery_ids = np.asarray(gallery_ids, np.int64)
    query_ids = np.asarray(query_ids, np.int64)
    return gallery_images, gallery_ids, query_images, query_ids, good_mask

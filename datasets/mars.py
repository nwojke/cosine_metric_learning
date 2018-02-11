# vim: expandtab:ts=4:sw=4
import os
import numpy as np
import cv2


# The maximum person ID in  the dataset.
MAX_LABEL = 1500

IMAGE_SHAPE = 256, 128, 3


def read_train_test_directory_to_str(directory):
    """Read bbox_train/bbox_test directory.

    Parameters
    ----------
    directory : str
        Path to bbox_train/bbox_test directory.

    Returns
    -------
    (List[str], List[int], List[int], List[int])
        Returns a tuple with the following entries:

        * List of image filenames.
        * List of corresponding unique IDs for the individuals in the images.
        * List of camera indices.
        * List of tracklet indices.

    """

    def to_label(x):
        return int(x) if x.isdigit() else -1

    dirnames = os.listdir(directory)
    image_filenames, ids, camera_indices, tracklet_indices = [], [], [], []
    for dirname in dirnames:
        filenames = os.listdir(os.path.join(directory, dirname))
        filenames = [
            f for f in filenames if os.path.splitext(f)[1] == ".jpg"]
        image_filenames += [
            os.path.join(directory, dirname, f) for f in filenames]
        ids += [to_label(dirname) for _ in filenames]
        camera_indices += [int(f[5]) for f in filenames]
        tracklet_indices += [int(f[7:11]) for f in filenames]

    return image_filenames, ids, camera_indices, tracklet_indices


def read_train_test_directory_to_image(directory, image_shape=(128, 64)):
    """Read images in bbox_train/bbox_test directory.

    Parameters
    ----------
    directory : str
        Path to bbox_train/bbox_test directory.
    image_shape : Tuple[int, int]
        A tuple (height, width) of the desired image size.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        Returns a tuple with the following entries:

        * Tensor of images in BGR color space.
        * One dimensional array of unique IDs for the individuals in the images.
        * One dimensional array of camera indices.
        * One dimensional array of tracklet indices.

    """
    reshape_fn = (
        (lambda x: x) if image_shape == IMAGE_SHAPE[:2]
        else (lambda x: cv2.resize(x, image_shape[::-1])))

    filenames, ids, camera_indices, tracklet_indices = (
        read_train_test_directory_to_str(directory))

    images = np.zeros((len(filenames), ) + image_shape + (3, ), np.uint8)
    for i, filename in enumerate(filenames):
        if i % 1000 == 0:
            print("Reading %s, %d / %d" % (directory, i, len(filenames)))
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        images[i] = reshape_fn(image)
    ids = np.asarray(ids, dtype=np.int64)
    camera_indices = np.asarray(camera_indices, dtype=np.int64)
    tracklet_indices = np.asarray(tracklet_indices, dtype=np.int64)
    return images, ids, camera_indices, tracklet_indices


def read_train_split_to_str(dataset_dir):
    """Read training data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to the MARS dataset directory; ``bbox_train`` should be a
        subdirectory of this folder.

    Returns
    -------
    (List[str], List[int], List[int], List[int])
        Returns a tuple with the following entries:

        * List of image filenames.
        * List of corresponding unique IDs for the individuals in the images.
        * List of camera indices.
        * List of tracklet indices.

    """
    train_dir = os.path.join(dataset_dir, "bbox_train")
    return read_train_test_directory_to_str(train_dir)


def read_train_split_to_image(dataset_dir, image_shape=(128, 64)):
    """Read training images to memory. This consumes a lot of memory.

    Parameters
    ----------
    dataset_dir : str
        Path to the MARS dataset directory; ``bbox_train`` should be a
        subdirectory of this folder.
    image_shape : Tuple[int, int]
        A tuple (height, width) of the desired image size.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        Returns a tuple with the following entries:

        * Tensor of images in BGR color space.
        * One dimensional array of unique IDs for the individuals in the images.
        * One dimensional array of camera indices.
        * One dimensional array of tracklet indices.

    """
    train_dir = os.path.join(dataset_dir, "bbox_train")
    return read_train_test_directory_to_image(train_dir, image_shape)


def read_test_split_to_str(dataset_dir):
    """Read training data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to the MARS dataset directory; ``bbox_test`` should be a
        subdirectory of this folder.

    Returns
    -------
    (List[str], List[int], List[int], List[int])
        Returns a tuple with the following entries:

        * List of image filenames.
        * List of corresponding unique IDs for the individuals in the images.
        * List of camera indices.
        * List of tracklet indices.

    """
    test_dir = os.path.join(dataset_dir, "bbox_test")
    return read_train_test_directory_to_str(test_dir)


def read_test_split_to_image(dataset_dir, image_shape=(128, 64)):
    """Read test images to memory. This consumes a lot of memory.

    Parameters
    ----------
    dataset_dir : str
        Path to the MARS dataset directory; ``bbox_test`` should be a
        subdirectory of this folder.
    image_shape : Tuple[int, int]
        A tuple (height, width) of the desired image size.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        Returns a tuple with the following entries:

        * Tensor of images in BGR color space.
        * One dimensional array of unique IDs for the individuals in the images.
        * One dimensional array of camera indices.
        * One dimensional array of tracklet indices.

    """
    test_dir = os.path.join(dataset_dir, "bbox_test")
    return read_train_test_directory_to_image(test_dir, image_shape)

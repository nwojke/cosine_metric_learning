# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2


def crop_to_shape(images, patch_shape):
    """Crop images to desired shape, respecting the target aspect ratio.

    Parameters
    ----------
    images : List[ndarray]
        A list of images in BGR format (dtype np.uint8)
    patch_shape : (int, int)
        Target image patch shape (height, width).

    Returns
    -------
    ndarray
        A tensor of output images.

    """
    assert len(images) > 0, "Empty image list is not allowed."
    channels = () if len(images[0].shape) == 0 else (images[0].shape[-1], )
    output_images = np.zeros(
        (len(images), ) + patch_shape + channels, dtype=np.uint8)

    target_aspect_ratio = float(patch_shape[1]) / patch_shape[0]
    for i, image in enumerate(images):
        image_aspect_ratio = float(image.shape[1]) / image.shape[0]
        if target_aspect_ratio > image_aspect_ratio:
            # Fix width, modify height.
            crop_height = image.shape[1] / target_aspect_ratio
            crop_width = image.shape[1]
        else:
            # Fix height, modify width.
            crop_width = target_aspect_ratio * image.shape[0]
            crop_height = image.shape[0]

        sx = int((image.shape[1] - crop_width) / 2)
        sy = int((image.shape[0] - crop_height) / 2)
        ex = int(min(sx + crop_width, image.shape[1]))
        ey = int(min(sy + crop_height, image.shape[0]))
        output_images[i, ...] = cv2.resize(
            image[sy:ey, sx:ex], patch_shape[::-1],
            interpolation=cv2.INTER_CUBIC)

    return output_images


def create_validation_split(data_y, num_validation_y, seed=None):
    """Split dataset into training and validation set with disjoint classes.

    Parameters
    ----------
    data_y : ndarray
        A label vector.
    num_validation_y : int | float
        The number of identities to split off for validation. If an integer
        is given, this value should be at least 1 and is interpreted as absolute
        number of validation identities. If a float is given, this value should
        be in [0, 1[ and is interpreted as fraction of validation identities.
    seed : Optional[int]
        A random generator seed used to select the validation idenities.

    Returns
    -------
    (ndarray, ndarray)
        Returns indices of training and validation set.

    """
    unique_y = np.unique(data_y)
    if isinstance(num_validation_y, float):
        num_validation_y = int(num_validation_y * len(unique_y))

    random_generator = np.random.RandomState(seed=seed)
    validation_y = random_generator.choice(
        unique_y, num_validation_y, replace=False)

    validation_mask = np.full((len(data_y), ), False, bool)
    for y in validation_y:
        validation_mask = np.logical_or(validation_mask, data_y == y)
    training_mask = np.logical_not(validation_mask)
    return np.where(training_mask)[0], np.where(validation_mask)[0]


def limit_num_elements_per_identity(data_y, max_num_images_per_id, seed=None):
    """Limit the number of elements per identity to `max_num_images_per_id`.

    Parameters
    ----------
    data_y : ndarray
        A label vector.
    max_num_images_per_id : int
        The maximum number of elements per identity that should remain in
        the data set.
    seed : Optional[int]
        Random generator seed.

    Returns
    -------
    ndarray
        A boolean mask that evaluates to True if the corresponding
        should remain in the data set.

    """
    random_generator = np.random.RandomState(seed=seed)
    valid_mask = np.full((len(data_y), ), False, bool)
    for y in np.unique(data_y):
        indices = np.where(data_y == y)[0]
        num_select = min(len(indices), max_num_images_per_id)
        indices = random_generator.choice(indices, num_select, replace=False)
        valid_mask[indices] = True
    return valid_mask


def create_cmc_probe_and_gallery(data_y, camera_indices=None, seed=None):
    """Create probe and gallery images for evaluation of CMC top-k statistics.

    For every identity, this function selects one image as probe and one image
    for the gallery. Cross-view validation is performed when multiple cameras
    are given.

    Parameters
    ----------
    data_y : ndarray
        Vector of data labels.
    camera_indices : Optional[ndarray]
        Optional array of camera indices. If possible, probe and gallery images
        are selected from different cameras (i.e., cross-view validation).
        If None given, assumes all images are taken from the same camera.
    seed : Optional[int]
        The random seed used to select probe and gallery images.

    Returns
    -------
    (ndarray, ndarray)
        Returns a tuple of indices to probe and gallery images.

    """
    data_y = np.asarray(data_y)
    if camera_indices is None:
        camera_indices = np.zeros_like(data_y, dtype=np.int)
    camera_indices = np.asarray(camera_indices)

    random_generator = np.random.RandomState(seed=seed)
    unique_y = np.unique(data_y)
    probe_indices, gallery_indices = [], []
    for y in unique_y:
        mask_y = data_y == y

        unique_cameras = np.unique(camera_indices[mask_y])
        if len(unique_cameras) == 1:
            # If we have only one camera, take any two images from this device.
            c = unique_cameras[0]
            indices = np.where(np.logical_and(mask_y, camera_indices == c))[0]
            if len(indices) < 2:
                continue  # Cannot generate a pair for this identity.
            i1, i2 = random_generator.choice(indices, 2, replace=False)
        else:
            # If we have multiple cameras, take images of two (randomly chosen)
            # different devices.
            c1, c2 = random_generator.choice(unique_cameras, 2, replace=False)
            indices1 = np.where(np.logical_and(mask_y, camera_indices == c1))[0]
            indices2 = np.where(np.logical_and(mask_y, camera_indices == c2))[0]
            i1 = random_generator.choice(indices1)
            i2 = random_generator.choice(indices2)

        probe_indices.append(i1)
        gallery_indices.append(i2)

    return np.asarray(probe_indices), np.asarray(gallery_indices)

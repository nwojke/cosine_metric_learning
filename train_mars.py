# vim: expandtab:ts=4:sw=4
import functools
import os
import numpy as np
import scipy.io as sio
import train_app
from datasets import mars
from datasets import util
import nets.deep_sort.network_definition as net


IMAGE_SHAPE = 128, 64, 3


class Mars(object):

    def __init__(self, dataset_dir, num_validation_y=0.1, seed=1234):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed

    def read_train(self):
        filenames, ids, camera_indices, _ = mars.read_train_split_to_str(
            self._dataset_dir)
        train_indices, _ = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in train_indices]
        ids = [ids[i] for i in train_indices]
        camera_indices = [camera_indices[i] for i in  train_indices]
        return filenames, ids, camera_indices

    def read_validation(self):
        filenames, ids, camera_indices, _ = mars.read_train_split_to_str(
            self._dataset_dir)
        _, valid_indices = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in valid_indices]
        ids = [ids[i] for i in valid_indices]
        camera_indices = [camera_indices[i] for i in valid_indices]
        return filenames, ids, camera_indices

    def read_test_filenames(self):
        filename = os.path.join(self._dataset_dir, "info", "test_name.txt")
        with open(filename, "r") as file_handle:
            content = file_handle.read()
            lines = content.splitlines()

        image_dir = os.path.join(self._dataset_dir, "bbox_test")
        return [os.path.join(image_dir, f[:4], f) for f in lines]


def main():
    arg_parser = train_app.create_default_argument_parser("mars")
    arg_parser.add_argument(
        "--dataset_dir", help="Path to MARS dataset directory.",
        default="resources/MARS-evaluation-master")
    args = arg_parser.parse_args()
    dataset = Mars(args.dataset_dir, num_validation_y=0.1, seed=1234)

    if args.mode == "train":
        train_x, train_y, _ = dataset.read_train()
        print("Train set size: %d images, %d identities" % (
            len(train_x), len(np.unique(train_y))))

        network_factory = net.create_network_factory(
            is_training=True, num_classes=mars.MAX_LABEL + 1,
            add_logits=args.loss_mode == "cosine-softmax")
        train_kwargs = train_app.to_train_kwargs(args)
        train_app.train_loop(
            net.preprocess, network_factory, train_x, train_y,
            num_images_per_id=4, image_shape=IMAGE_SHAPE, **train_kwargs)
    elif args.mode == "eval":
        valid_x, valid_y, camera_indices = dataset.read_validation()
        print("Validation set size: %d images, %d identities" % (
            len(valid_x), len(np.unique(valid_y))))

        network_factory = net.create_network_factory(
            is_training=False, num_classes=mars.MAX_LABEL + 1,
            add_logits=args.loss_mode == "cosine-softmax")
        eval_kwargs = train_app.to_eval_kwargs(args)
        train_app.eval_loop(
            net.preprocess, network_factory, valid_x, valid_y, camera_indices,
            image_shape=IMAGE_SHAPE, num_galleries=20, **eval_kwargs)
    elif args.mode == "export":
        filenames = dataset.read_test_filenames()

        network_factory = net.create_network_factory(
            is_training=False, num_classes=mars.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        features = train_app.encode(
            net.preprocess, network_factory, args.restore_path,
            filenames, image_shape=IMAGE_SHAPE)
        sio.savemat(
            os.path.join(args.dataset_dir, "feat_test.mat"),
            {"features": features})
    elif args.mode == "finalize":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=mars.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app.finalize(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path, image_shape=IMAGE_SHAPE,
            output_filename="./mars.ckpt")
    elif args.mode == "freeze":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=mars.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app.freeze(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path, image_shape=IMAGE_SHAPE,
            output_filename="./mars.pb")
    else:
        raise ValueError("Invalid mode argument.")


if __name__ == "__main__":
    main()

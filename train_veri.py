# vim: expandtab:ts=4:sw=4
#
# Code to train on VeRi dataset [1].
#
#
# Example call to train a model using the cosine softmax classifier:
#
# ```
# python train_veri.py \
#     --dataset_dir=./VeRi \
#     --loss_mode=cosine-softmax \
#     --log_dir=./output/veri \
#     --run_id=cosine-softmax
# ```
#
# Example call to run evaluation on validation set (parallel to training):
#
# ```
# CUDA_VISIBLE_DEVICES="" python2 train_veri.py \
#     --dataset_dir=./VeRi \
#     --loss_mode=cosine-softmax \
#     --log_dir=./output/veri \
#     --run_id=cosine-softmax \
#     --mode=eval \
#     --eval_log_dir=./eval_output
# ```
#
# Example call to freeze a trained model (here model.ckpt-100000):
#
# ```
# python train_veri.py \
#     --restore_path=./output/veri/cosine-softmax/model.ckpt-100000 \
#     --mode=freeze
# ```
#
# [1] https://github.com/VehicleReId/VeRidataset
#
import functools
import os
import numpy as np
import scipy.io as sio
import train_app
from datasets import veri
from datasets import util
import nets.deep_sort.network_definition as net

class VeRi(object):

    def __init__(self, dataset_dir, num_validation_y=0.1, seed=1234):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed

    def read_train(self):
        filenames, ids, camera_indices = veri.read_train_split_to_str(
            self._dataset_dir)
        train_indices, _ = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in train_indices]
        ids = [ids[i] for i in train_indices]
        camera_indices = [camera_indices[i] for i in train_indices]
        return filenames, ids, camera_indices

    def read_validation(self):
        filenames, ids, camera_indices = veri.read_train_split_to_str(
            self._dataset_dir)
        _, valid_indices = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in valid_indices]
        ids = [ids[i] for i in valid_indices]
        camera_indices = [camera_indices[i] for i in valid_indices]
        return filenames, ids, camera_indices


def main():
    arg_parser = train_app.create_default_argument_parser("veri")
    arg_parser.add_argument(
        "--dataset_dir", help="Path to VeRi dataset directory.",
        default="resources/VeRi")
    args = arg_parser.parse_args()
    dataset = VeRi(args.dataset_dir, num_validation_y=0.1, seed=1234)

    if args.mode == "train":
        train_x, train_y, _ = dataset.read_train()
        print("Train set size: %d images, %d identites" % (
            len(train_x), len(np.unique(train_y))))

        network_factory = net.create_network_factory(
            is_training=True, num_classes=veri.MAX_LABEL +1,
            add_logits=args.loss_mode == "cosine-softmax")
        train_kwargs = train_app.to_train_kwargs(args)
        train_app.train_loop(
            net.preprocess, network_factory, train_x, train_y,
            num_images_per_id=4, image_shape=veri.IMAGE_SHAPE,
            **train_kwargs)
    elif args.mode == "eval":
        valid_x, valid_y, camera_indices = dataset.read_validation()
        print("Validation set size: %d images, %d identites" % (
            len(valid_x), len(np.unique(valid_y))))

        network_factory = net.create_network_factory(
            is_training=False, num_classes=veri.MAX_LABEL + 1,
            add_logits=args.loss_mode == "cosine-softmax")
        eval_kwargs = train_app.to_eval_kwargs(args)
        train_app.eval_loop(
            net.preprocess, network_factory, valid_x, valid_y, camera_indices,
            image_shape=veri.IMAGE_SHAPE, **eval_kwargs)
    elif args.mode == "export":
        raise NotImplementedError()
    elif args.mode == "finalize":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=veri.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app.finalize(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path, image_shape=veri.IMAGE_SHAPE,
            output_filename="./veri.ckpt")
    elif args.mode == "freeze":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=veri.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app.freeze(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path, image_shape=veri.IMAGE_SHAPE,
            output_filename="./veri.pb")
    else:
        raise ValueError("Invalid mode argument.")


if __name__ == "__main__":
    main()

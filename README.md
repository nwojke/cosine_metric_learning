# cosine_metric_learning

## Introduction

This repository contains code for training a metric feature representation to be
used with the [deep_sort tracker](https://github.com/nwojke/deep_sort). The
approach is described in

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }

Pre-trained models used in the paper can be found
[here](https://owncloud.uni-koblenz-landau.de/owncloud/s/leZNP94NUN58M73). A
preprint of the paper is available [here](http://elib.dlr.de/116408/).
The repository comes with code to train a model on the
[Market1501](http://www.liangzheng.org/Project/project_reid.html)
and [MARS](http://www.liangzheng.com.cn/Project/project_mars.html) datasets.

## Training on Market1501

The following description assumes you have downloaded the Market1501 dataset to
``./Market-1501-v15.09.15``. The following command starts training
using the cosine-softmax classifier described in the above paper:
```
python train_market1501.py \
    --dataset_dir=./Market-1501-v15.09.15/ \
    --loss_mode=cosine-softmax \
    --log_dir=./output/market1501/ \
    --run_id=cosine-softmax
```
This will create a directory `./output/market1501/cosine-softmax` where
TensorFlow checkpoints are stored and which can be monitored using
``tensorboard``:
```
tensorboard --logdir ./output/market1501/cosine-softmax --port 6006
```
The code splits off 10% of the training data for validation.
Concurrently to training, run the following command to run CMC evaluation
metrics on the validation set:
```
CUDA_VISIBLE_DEVICES="" python train_market1501.py \
    --mode=eval \
    --dataset_dir=./Market-1501-v15.09.15/ \
    --loss_mode=cosine-softmax \
    --log_dir=./output/market1501/ \
    --run_id=cosine-softmax \
    --eval_log_dir=./eval_output/market1501
```
The command will block indefinitely to monitor the training directory for saved
checkpoints and each stored checkpoint in the training directory is evaluated on
the validation set. The results of this evaluation are stored in
``./eval_output/market1501/cosine-softmax`` to be monitored using
``tensorboard``:
```
tensorboard --logdir ./eval_output/market1501/cosine-softmax --port 6007
```

## Training on MARS

To train on MARS, download the
[evaluation software](https://github.com/liangzheng06/MARS-evaluation) and
extract ``bbox_train.zip`` and ``bbox_test.zip`` from the
[dataset website](http://www.liangzheng.com.cn/Project/project_mars.html)
into the evaluation software directory. The following description assumes they
are stored in ``./MARS-evaluation-master/bbox_train`` and
``./MARS-evaluation-master/bbox_test``. Training can be started with the following
command:
```
python train_mars.py \
    --dataset_dir=./MARS-evaluation-master \
    --loss_mode=cosine-softmax \
    --log_dir=./output/mars/ \
    --run_id=cosine-softmax
```
Again, this will create a directory `./output/mars/cosine-softmax` where
TensorFlow checkpoints are stored and which can be monitored using
``tensorboard``:
```
tensorboard --logdir ./output/mars/cosine-softmax --port 7006
```
As for Market1501, 10% of the training data are split off for validation.
Concurrently to training, run the following command to run CMC evaluation
metrics on the validation set:
```
CUDA_VISIBLE_DEVICES="" python train_mars.py \
    --mode=eval \
    --dataset_dir=./MARS-evaluation-master/ \
    --loss_mode=cosine-softmax \
    --log_dir=./output/mars/ \
    --run_id=cosine-softmax \
    --eval_log_dir=./eval_output/mars
```
Evaluation metrics on the validation set can be monitored with ``tensorboard``
```
tensorboard --logdir ./eval_output/mars/cosine-softmax
``` 

## Testing

Final model testing has been carried out using evaluation software provided by
the dataset authors. The training scripts can be used to write features of the
test split. The following command exports MARS test features to
``./MARS-evaluation-master/feat_test.mat``
```
python train_mars.py \
    --mode=export \
    --dataset_dir=./MARS-evaluation-master \
    --loss_mode=cosine-softmax .\
    --restore_path=PATH_TO_CHECKPOINT
``` 
where ``PATH_TO_CHECKPOINT`` the checkpoint file to evaluate. Note that the
evaluation script needs minor adjustments to apply the cosine similarity metric.
More precisely, change the feature computation in
``utils/process_box_features.m`` to average pooling (line 8) and apply
a re-normalization at the end of the file. The modified file should look like
this:
```
function video_feat = process_box_feat(box_feat, video_info)

nVideo = size(video_info, 1);
video_feat = zeros(size(box_feat, 1), nVideo);
for n = 1:nVideo
    feature_set = box_feat(:, video_info(n, 1):video_info(n, 2));
%    video_feat(:, n) = max(feature_set, [], 2); % max pooling 
     video_feat(:, n) = mean(feature_set, 2); % avg pooling
end

%%% normalize train and test features
sum_val = sqrt(sum(video_feat.^2));
for n = 1:size(video_feat, 1)
    video_feat(n, :) = video_feat(n, :)./sum_val;
end
```
The Market1501 script contains a similar export functionality which can be
applied in the same way as described for MARS:
```
python train_market1501.py \
    --mode=export \
    --dataset_dir=./Market-1501-v15.09.15/
    --sdk_dir=./Market-1501_baseline-v16.01.14/
    --loss_mode=cosine-softmax \
    --restore_path=PATH_TO_CHECKPOINT
```
This command creates ``./Market-1501_baseline-v16.01.14/feat_query.mat`` and
``./Market-1501_baseline-v16.01.14/feat_test.mat`` to be used with the
Market1501 evaluation code. 

## Model export

To export your trained model for use with the
[deep_sort tracker](https://github.com/nwojke/deep_sort), run the following
command:
```
python train_mars.py --mode=freeze --restore_path=PATH_TO_CHECKPOINT
```
This will create a ``mars.pb`` file which can be supplied to Deep SORT. Again,
the Market1501 script contains a similar function.

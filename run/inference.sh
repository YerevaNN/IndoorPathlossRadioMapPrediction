#!/usr/bin/env bash

python run.py --config-name=inference \
  gpu=0 \
  split=test \
  network.image_size=518 \
  network.min_mlp_tokens=0 \
  network.num_channels=4 \
  datamodule.idx_to_freq=null \
  datamodule.max_freq=null \
  datamodule.mixup_ratio=0.0 \
  datamodule.crop_ratio=0.0 \
  datamodule.channels_to_use=[0,1,2,5] \
  datamodule.num_workers=12

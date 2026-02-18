#!/usr/bin/env bash

python run.py --config-name=train \
  datamodule.batch_size=10 \
  datamodule.num_workers=6 \
  datamodule.channels_to_use=[0,1,2,5] \
  datamodule.idx_to_freq=null \
  datamodule.max_freq=null \
  network.num_channels=4 \
  network.image_size=518 \
  network.min_mlp_tokens=0 \
  trainer.max_epochs=500 \
  trainer.devices=[0]

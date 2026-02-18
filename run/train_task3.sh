#!/usr/bin/env bash

python run.py --config-name=train \
  datamodule=icassp_task3 \
  datamodule.batch_size=4 \
  datamodule.num_workers=6 \
  datamodule.channels_to_use=[0,1,2,3,5] \
  network.num_channels=6 \
  network.image_size=518 \
  network.min_mlp_tokens=0 \
  network.neck_input_dim=512 \
  network.neck_size=[32,32,32,64,64,64,128,128,128,256,256,256,512,512] \
  trainer.max_epochs=500 \
  trainer.devices=[0]

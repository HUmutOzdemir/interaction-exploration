# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

python -m affordance_seg.train_unet \
	   --data_dir affordance_seg/data/rgb/ \
	   --cv_dir affordance_seg/cv/rgb_unet \
	   --gpus 1 \
	   --weight_decay 1e-5 \
	   --train
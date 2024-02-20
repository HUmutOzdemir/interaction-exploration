# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

for GPU in 0; do
    python -m affordance_seg.collect_dset \
         --out-dir affordance_seg/data/rgb/ \
         NUM_PROCESSES 8 \
         LOAD interaction_exploration/cv/rgb/run_comparison/ckpt.24.pth \
         EVAL.DATASET affordance_seg/data/episode_splits/episodes_K_256_split_$GPU.json \
         ENV.NUM_STEPS 256 \
         TORCH_GPU_ID 0 \
         X_DISPLAY :0 \
         ENV.ENV_NAME ThorBeaconsFixedScaleComparison-v0 \
         MODE eval
done


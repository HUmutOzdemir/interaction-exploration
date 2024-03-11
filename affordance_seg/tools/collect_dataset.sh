# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

for GPU in 0; do
    for SPLIT in {0..24}; do
        timeout 45m python -m affordance_seg.collect_dset \
            --out-dir affordance_seg/data/rgb/ \
            NUM_PROCESSES 1 \
            LOAD interaction_exploration/cv/rgb/run_comparison/ckpt.48.pth \
            EVAL.DATASET affordance_seg/data/episode_splits/episode_split_${GPU}_splitted/episode_split_${SPLIT}.json \
            ENV.NUM_STEPS 128 \
            TORCH_GPU_ID 0 \
            X_DISPLAY :0 \
            ENV.ENV_NAME ThorBeaconsFixedScaleComparison-v0 \
            MODE eval
        pkill -f uoezdemir
        pkill -f ai2thor
        pkill -f python
    done
done


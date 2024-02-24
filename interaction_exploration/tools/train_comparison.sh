python -m interaction_exploration.run \
    --config interaction_exploration/config/intexp.yaml \
    --mode train \
    SEED 0 \
    TORCH_GPU_ID 0 \
    X_DISPLAY :0 \
    ENV.NUM_STEPS 256 \
    NUM_PROCESSES 16 \
    CHECKPOINT_FOLDER interaction_exploration/cv/intexp/run_comparison/ \
    MODEL.BEACON_MODEL affordance_seg/cv/rgb_unet/epoch=07-val_loss=0.5053.ckpt
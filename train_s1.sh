export CUDA_VISIBLE_DEVICES=0

python train_s1.py \
    --config config/train_s1.yaml \
    --output_dir output/stage1 \
    --no_wandb \
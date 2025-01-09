CUDA_VISIBEL_DEVICES=0,1 torchrun --nproc_per_node=2 dpo.py \
    --per_device_train_batch_size 8 \
    --ddp_find_unused_parameters true \

    
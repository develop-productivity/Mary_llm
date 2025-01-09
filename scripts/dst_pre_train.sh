
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 pre_train.py \
    --per_device_train_batch_size 64 \
    --ddp_find_unused_parameters true \
    --num_train_epochs 5 \
    --data_ratio 0.8 \
    --run_name llm_pretrain_data_0.8 \

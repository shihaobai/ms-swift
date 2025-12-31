export NPROC_PER_NODE=2
export CUDA_VISIBLE_DEVICES=6,7
swift sft \
    --model /mtc/baishihao/niren \
    --train_type full \
    --dataset '/mtc/baishihao/ms-swift/data_messages_3w_messages.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --lr_scheduler_type linear \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir ./niren_output4 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 20 \
    --model_author shihaobai \
    --use_logits_to_keep false \
    --save_total_limit 2 \
    --model_name niren --model_type mistral_with_mtp --template mistral_niren
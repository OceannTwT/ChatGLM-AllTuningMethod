
LR=1e-4

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --num_gpus=8 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file gbl_50000.json \
    --prompt_column Query \
    --response_column Label \
    --overwrite_cache \
    --model_name_or_path ./chatglm-6b \
    --output_dir /mnt/task_wrapper/user_output/artifacts/lr_50k_8gpu_baseline/  \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 6250 \
    --logging_steps 10 \
    --save_steps 1000000 \
    --learning_rate $LR \
    --use_lora_train True \
    --remove_unused_columns False \
    --source_prefix "You are a siri labeler, label this text to word, text: " \
    --fp16
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="train_dpo"


MODEL_NAME_OR_PATH="/share/shared_models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"

torchrun --nproc_per_node=4 \
--master_port 22002 \
train_dpo.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_type="llama-lora" \
--train_data_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.e_win_lose.jsonl" \
--per_device_train_batch_size=10 \
--use_data_percent=1.0 \
--max_length=3072 \
--max_prompt_length=2048 \
--max_ctx_num=3 \
--use_data_percent=1.0 \
--learning_rate=1e-6 \
--num_train_epochs=1 \
--warmup_ratio=0.05 \
--logging_steps=1 \
--save_strategy='epoch' \
--save_total_limit=2 \
--log_level="info" \
--report_to="none" \
--run_name="llama2-chat" \
--output_dir="./checkpoints/G_0" \
--force_emptying_dir \
--bf16=true \
--gradient_checkpointing \
--deepspeed="ds_config.2.json" \



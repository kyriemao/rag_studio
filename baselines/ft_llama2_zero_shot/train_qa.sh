export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="train_qa"


MODEL_NAME_OR_PATH="/share/shared_models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"

torchrun --nproc_per_node=4 \
--master_port 22005 \
train_qa.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_type="llama-lora" \
--train_data_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/train.wiki.jsonl" \
--per_device_train_batch_size=48 \
--use_data_percent=1.0 \
--max_seq_len=256 \
--learning_rate=2e-5 \
--num_train_epochs=3 \
--response_template="[/INST]" \
--logging_steps=1 \
--save_strategy='epoch' \
--save_total_limit=3 \
--log_level="info" \
--report_to="wandb" \
--run_name="ft_llama2_zero_shot" \
--output_dir="./checkpoints" \
--force_emptying_dir \
--bf16=true \
--gradient_checkpointing \
--deepspeed="../../ds_config.2.json" \


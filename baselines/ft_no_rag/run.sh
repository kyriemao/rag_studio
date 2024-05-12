export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="sft"

# train
MODEL_NAME_OR_PATH=/share/shared_models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33
# MODEL_NAME_OR_PATH=/share/shared_models/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9

torchrun --nproc_per_node=8 \
--master_port 22005 \
../sft.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_type="llama-lora" \
--train_data_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/train.wiki.jsonl" \
--per_device_train_batch_size=32 \
--use_data_percent=1.0 \
--max_seq_len=300 \
--max_ctx_num=0 \
--warmup_ratio=0.02 \
--learning_rate=1e-4 \
--num_train_epochs=2 \
--response_template="[/INST]" \
--logging_steps=1 \
--save_strategy='epoch' \
--save_total_limit=3 \
--log_level="info" \
--report_to="wandb" \
--run_name="ft_no_rag" \
--output_dir="./checkpoints" \
--force_emptying_dir \
--bf16=true \
--gradient_checkpointing \
--deepspeed="../../ds_config.2.json" \

# inference
accelerate launch --num_processes=8 \
../rag_inference.py \
--model_name_or_path="./checkpoints" \
--model_type="llama-lora" \
--batch_size=4 \
--max_ctx_num=0 \
--max_new_tokens=32 \
--input_data_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \
--output_path="./result.jsonl" \

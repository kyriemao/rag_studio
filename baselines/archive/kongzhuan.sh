export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="kongzhuan"


# 1. raft
MODEL_NAME_OR_PATH=/share/shared_models/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9

torchrun --nproc_per_node=8 \
--master_port 21005 \
sft.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_type="llama-lora" \
--train_data_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/train.wiki.jsonl" \
--per_device_train_batch_size=64 \
--use_data_percent=1.0 \
--max_seq_len=256 \
--max_ctx_num=0 \
--learning_rate=1e-4 \
--num_train_epochs=99999 \
--response_template="[/INST]" \
--logging_steps=1 \
--save_strategy='epoch' \
--save_total_limit=1 \
--log_level="info" \
--report_to="none" \
--run_name="ft_no_rag" \
--output_dir="./kongzhuan_checkpoints" \
--force_emptying_dir \
--bf16=true \
--gradient_checkpointing \
--deepspeed="../ds_config.2.json" \
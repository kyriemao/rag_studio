export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="train_retriever"

MODEL_NAME_OR_PATH="BAAI/bge-base-en-v1.5"

torchrun --nproc_per_node=4 \
--master_port 22004 \
train_retriever.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_type="bge" \
--train_data_path="/share/kelong/rag_studio/benchmarks/triviaqa/hard_neg.jsonl" \
--normalize_emb \
--per_device_train_batch_size=64 \
--ddp_broadcast_buffers=false \
--ddp_find_unused_parameters=true \
--temperature=0.01 \
--use_data_percent=1.0 \
--gradient_accumulation_steps=1 \
--max_q_len=64 \
--max_p_len=512 \
--neg_num=3 \
--warmup_steps=50 \
--learning_rate=1e-4 \
--num_train_epochs=2 \
--logging_steps=1 \
--save_strategy='steps' \
--save_steps=100 \
--save_total_limit=2 \
--log_level="info" \
--report_to="none" \
--run_name="bge" \
--output_dir="./checkpoints/R_0" \
--force_emptying_dir \
--bf16=true \
--deepspeed="ds_config.1.json" \


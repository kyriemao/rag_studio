export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="sft"

if [ ! -f "train.ctx.jsonl" ]; then
    # 1. get training data
    # index the corpus
    MODEL_PATH="BAAI/bge-base-en-v1.5"

    torchrun --nproc_per_node=4 \
    --master_port 28120 \
    indexing.py \
    --model_name_or_path=$MODEL_PATH \
    --model_type="bge" \
    --normalize_emb \
    --max_p_len=512 \
    --corpus_path="/share/kelong/rag_studio/benchmarks/triviaqa/corpus.jsonl" \
    --per_device_eval_batch_size=1024 \
    --num_psg_per_block=1000000 \
    --data_output_dir="/share/kelong/rag_studio/benchmarks/triviaqa/corpus_index" \

    # perform retrieval
    python ../../retrieval.py \
    --model_name_or_path=$MODEL_PATH \
    --model_type="bge" \
    --embedding_size=768 \
    --query_data_path_list="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/train.wiki.jsonl" \
    --max_q_len=128 \
    --per_device_eval_batch_size=64 \
    --index_dir="/share/kelong/rag_studio/benchmarks/triviaqa/corpus_index" \
    --data_output_dir="./train_retrieved_data" \

    # incoporate top ctx passages
    python ../../data_prepare.py \
    --stage="add_ctx" \
    --n_context_psgs=3 \
    --corpus_path="/share/kelong/rag_studio/benchmarks/triviaqa/corpus.jsonl" \
    --xy_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/train.wiki.jsonl" \
    --retrieved_data_path="./train_retrieved_data" \
    --xy_ctx_path="./train.ctx.jsonl" \

else
    # 如果不存在，输出提示信息
    echo "train.ctx.jsonl exists in the current directory."
fi



# 2. train
# MODEL_NAME_OR_PATH=/share/shared_models/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
MODEL_NAME_OR_PATH=/share/shared_models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33

torchrun --nproc_per_node=8 \
--master_port 22007 \
../sft.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_type="llama-lora" \
--train_data_path="./train.ctx.jsonl" \
--per_device_train_batch_size=32 \
--use_data_percent=1.0 \
--max_seq_len=768 \
--max_ctx_num=3 \
--warmup_ratio=0.02 \
--learning_rate=1e-4 \
--num_train_epochs=2 \
--response_template="[/INST]" \
--logging_steps=3 \
--save_strategy='epoch' \
--save_total_limit=1 \
--log_level="info" \
--report_to="wandb" \
--run_name="ft_rag" \
--output_dir="./checkpoints" \
--force_emptying_dir \
--bf16=true \
--gradient_checkpointing \
--deepspeed="../../ds_config.2.json" \

# 3. inference
accelerate launch --num_processes=8 \
../rag_inference.py \
--model_name_or_path="./checkpoints" \
--model_type="llama-lora" \
--batch_size=4 \
--max_ctx_num=3 \
--max_new_tokens=32 \
--input_data_path="../rag/dev.wiki.ctx.jsonl" \
--output_path="./result.jsonl" \

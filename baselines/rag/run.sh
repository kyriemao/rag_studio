export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

if [ ! -f "dev.wiki.ctx.jsonl" ]; then
    # 1. get ctx data
    # index the corpus
    MODEL_PATH="BAAI/bge-base-en-v1.5"

    torchrun --nproc_per_node=4 \
    --master_port 28120 \
    ../../indexing.py \
    --model_name_or_path=$MODEL_PATH \
    --model_type="bge" \
    --normalize_emb \
    --max_p_len=150 \
    --corpus_path="/share/kelong/rag_studio/benchmarks/triviaqa/corpus.jsonl" \
    --per_device_eval_batch_size=2048 \ 
    --num_psg_per_block=1000000 \
    --data_output_dir="/share/kelong/rag_studio/benchmarks/triviaqa/corpus_index" \


    # perform retrieval
    python ../../retrieval.py \
    --model_name_or_path=$MODEL_PATH \
    --model_type="bge" \
    --embedding_size=768 \
    --query_data_path_list="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \
    --max_q_len=128 \
    --per_device_eval_batch_size=64 \
    --index_dir="/share/kelong/rag_studio/benchmarks/triviaqa/corpus_index" \
    --data_output_dir="./retrieved_data" \


    # incoporate top ctx passages
    python ../../data_prepare.py \
    --stage="add_ctx" \
    --n_context_psgs=3 \
    --corpus_path="/share/kelong/rag_studio/benchmarks/triviaqa/corpus.jsonl" \
    --xy_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \
    --retrieved_data_path="./retrieved_data" \
    --xy_ctx_path="./dev.wiki.ctx.jsonl" \

else
    echo "dev.wiki.ctx.jsonl exists in the current directory."
fi


# 2. inference
MODEL_NAME_OR_PATH=/share/shared_models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33
accelerate launch --num_processes=8 \
../rag_inference.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_type="llama" \
--batch_size=4 \
--max_ctx_num=3 \
--max_new_tokens=32 \
--input_data_path="./dev.wiki.ctx.jsonl" \
--output_path="./result.jsonl" \

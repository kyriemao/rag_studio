export CUDA_VISIBLE_DEVICES=0,1,2,3

# First retrieve three context passages
# 1. index the corpus
MODEL_PATH="BAAI/bge-base-en-v1.5"

torchrun --nproc_per_node=4 \
--master_port 28120 \
../../indexing.py \
--model_name_or_path=$MODEL_PATH \
--model_type="bge" \
--normalize_emb \
--max_p_len=512 \
--corpus_path="/share/kelong/rag_studio/benchmarks/triviaqa/corpus.jsonl" \
--per_device_eval_batch_size=1024 \
--num_psg_per_block=1000000 \
--data_output_dir="./corpus_index" \


# 2. perform retrieval
python ../../retrieval.py \
--model_name_or_path=$MODEL_PATH \
--model_type="bge" \
--embedding_size=768 \
--query_data_path_list="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \
--max_q_len=128 \
--per_device_eval_batch_size=64 \
--index_dir="./corpus_index" \
--data_output_dir="./retrieved_data" \


# 3. incoporate top ctx passages
python ../../data_prepare.py \
--stage="add_ctx" \
--n_context_psgs=3 \
--corpus_path="/share/kelong/rag_studio/benchmarks/triviaqa/corpus.jsonl" \
--xy_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \
--retrieved_data_path="./retrieved_data" \
--xy_ctx_path="./dev.wiki.ctx.jsonl" \

# 4. perform rag
python llama2_n_shot_rag.py \
--openai_api_base="http://localhost:8000/v1/chat/completions" \
--model_name_or_path="llama2" \
--model_type="llama" \
--batch_size=64 \
--max_ctx_num=3 \
--max_new_tokens=128 \
--input_data_path="./dev.wiki.ctx.jsonl" \
--output_path="./bge_llama2_three_shot_rag.jsonl" \


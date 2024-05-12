export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_NAME_OR_PATH=/share/shared_models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33

accelerate launch --num_processes=8 \
../rag_inference.py \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_type="llama" \
--batch_size=4 \
--max_ctx_num=0 \
--max_new_tokens=32 \
--input_data_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \
--output_path="./result.jsonl" \
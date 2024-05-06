python llama2_n_shot_rag.py \
--openai_api_base="http://localhost:8000/v1/chat/completions" \
--model_name_or_path="llama2" \
--model_type="llama" \
--batch_size=64 \
--max_ctx_num=0 \
--max_new_tokens=128 \
--input_data_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \
--output_path="./llama2_zero_shot.jsonl" \


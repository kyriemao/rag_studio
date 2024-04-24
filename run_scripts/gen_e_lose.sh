python gen_e_lose.py \
--openai_api_base="http://localhost:8000/v1/chat/completions" \
--model_name_or_path="llama2" \
--batch_size=64 \
--max_ctx_num=3 \
--max_new_tokens=1024 \
--input_data_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.ctx.jsonl" \
--output_path="/share/kelong/rag_studio/benchmarks/triviaqa/e_lose.jsonl" \

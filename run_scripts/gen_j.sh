python gen_j.py \
--openai_api_base="http://localhost:8000/v1/chat/completions" \
--model_name_or_path="llama2" \
--batch_size=64 \
--max_new_tokens=128 \
--xy_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.ctx.jsonl" \
--e_lose_path="/share/kelong/rag_studio/benchmarks/triviaqa/e_lose.jsonl" \
--output_path="/share/kelong/rag_studio/benchmarks/triviaqa/judgment.jsonl" \

python gen_e_win.py \
--openai_api_base="http://localhost:8000/v1/chat/completions" \
--model_name_or_path="llama2" \
--model_type="llama" \
--batch_size=64 \
--max_new_tokens=2048 \
--is_seed \
--input_data_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.ctx.jsonl" \
--output_path="/share/kelong/rag_studio/benchmarks/triviaqa/e_win.seed.jsonl" \


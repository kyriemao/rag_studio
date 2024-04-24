python gen_e_lose.py \
--openai_api_base="http://localhost:8000/v1/chat/completions" \
--model_name_or_path="llama2" \
--model_type="llama" \
--batch_size=128 \
--max_ctx_num=3 \
--max_new_tokens=1024 \
--input_data_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.ctx.jsonl" \
--output_path="/share/kelong/rag_studio/benchmarks/triviaqa/e_lose.jsonl" \

# combine e_win, e_lose
python data_prepare.py \
--stage="add_e_win_lose" \
--xy_ctx_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.ctx.jsonl" \
--e_win_path="/share/kelong/rag_studio/benchmarks/triviaqa/e_win.seed.jsonl" \
--e_lose_path="/share/kelong/rag_studio/benchmarks/triviaqa/e_lose.jsonl" \
--xy_e_win_lose_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy_e_win_lose.jsonl" \
--is_seed \
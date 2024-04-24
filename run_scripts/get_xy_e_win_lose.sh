# combine e_win, e_lose to get the training data of DPO

python data_prepare.py \
--stage="add_e_win_lose" \
--xy_ctx_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.ctx.jsonl" \
--e_win_path="/share/kelong/rag_studio/benchmarks/triviaqa/e_win.seed.jsonl" \
--e_lose_path="/share/kelong/rag_studio/benchmarks/triviaqa/e_lose.jsonl" \
--xy_e_win_lose_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.e_win_lose.jsonl" \
--is_seed \
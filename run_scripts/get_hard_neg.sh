python data_prepare.py \
--stage="get_hard_neg" \
--n_context_psgs=3 \
--xy_ctx_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.ctx.jsonl" \
--e_lose_path="/share/kelong/rag_studio/benchmarks/triviaqa/e_lose.jsonl" \
--judgment_path="/share/kelong/rag_studio/benchmarks/triviaqa/judgment.jsonl" \
--hard_neg_path="/share/kelong/rag_studio/benchmarks/triviaqa/hard_neg.jsonl" \

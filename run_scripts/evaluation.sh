
# python evaluation.py \
# --pred_path="/share/kelong/rag_studio/baselines/ft_no_rag/result.jsonl" \
# --oracle_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \

# python evaluation.py \
# --pred_path="/share/kelong/rag_studio/baselines/ft_rag/result2.ctx3.jsonl" \
# --oracle_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \

# python evaluation.py \
# --pred_path="/share/kelong/rag_studio/baselines/ft_rag/result2.ctx0.jsonl" \
# --oracle_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \


python evaluation.py \
--pred_path="/share/kelong/rag_studio/baselines/no_rag/result.jsonl" \
--oracle_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \

python evaluation.py \
--pred_path="/share/kelong/rag_studio/baselines/rag/result.jsonl" \
--oracle_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/dev.wiki.jsonl" \
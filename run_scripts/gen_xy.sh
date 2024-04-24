python gen_xy.py \
--openai_api_base="http://localhost:8000/v1/chat/completions" \
--model_name_or_path="llama2" \
--model_type="llama" \
--batch_size=64 \
--max_new_tokens=2048 \
--num_passage=500 \
--num_question=2 \
--corpus_path="/share/kelong/rag_studio/benchmarks/triviaqa/corpus.jsonl" \
--output_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.raw.jsonl" \


python data_prepare.py \
--stage="gen_xy" \
--corpus_path="/share/kelong/rag_studio/benchmarks/triviaqa/corpus.jsonl" \
--xy_raw_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.raw.jsonl" \
--xy_path="/share/kelong/rag_studio/benchmarks/triviaqa/xy.jsonl" \

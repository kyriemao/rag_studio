python chunking.py \
--tokenizer="Qwen/Qwen1.5-7B-Chat" \
--corpus_path="/share/kelong/rag_studio/benchmarks/triviaqa/original/corpus.wiki.jsonl" \
--chunk_strategy="first" \
--chunk_size=512 \
--output_path="/share/kelong/rag_studio/benchmarks/triviaqa/corpus.jsonl" \
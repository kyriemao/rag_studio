python chunking.py \
--tokenizer="/share/shared_models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33" \
--corpus_path="/share/kelong/rag_studio/benchmarks/triviaqa/original_data/corpus.wiki.jsonl" \
--chunk_strategy="hard" \
--chunk_size=150 \
--output_path="/share/kelong/rag_studio/benchmarks/triviaqa/corpus.jsonl" \
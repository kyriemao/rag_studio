import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

from IPython import embed



def chunk_text(text: str, chunk_size: int, chunk_strategy: str, tokenizer: AutoTokenizer) -> list[str]:
    """
    Breaks down the `text` into chunks of size `chunk_size` using the specified `chunk_strategy`.
    """
    if chunk_strategy == "hard":
        tokens = tokenizer.encode(text)
        chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
        chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
        return chunks
    elif chunk_strategy == "first":
        tokens = tokenizer.encode(text)
        chunks = [tokenizer.decode(tokens[:chunk_size], skip_special_tokens=True)]
        return chunks

    elif chunk_strategy == "semantic":
        # TODO
        raise NotImplementedError("Semantic chunking is not yet implemented.")
    else:
        raise ValueError("Invalid chunking strategy. Choose between 'hard' and 'semantic'.")
    
    
def get_chunks(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    with open(args.output_path, 'w') as fw:
        with open(args.corpus_path, 'r') as f:
            for line in tqdm(f, desc="Chunking documents..."):
                line = json.loads(line)
                text = line["text"]
                chunks = chunk_text(text, args.chunk_size, args.chunk_strategy, tokenizer)
                for i, chunk in enumerate(chunks):
                    if args.chunk_strategy == 'first':
                        _id = line['_id']
                    else:
                        _id = line["_id"] + f"_{i}"
                    d = {}
                    d["_id"] = _id
                    d["text"] = chunk
                    fw.write(json.dumps(d) + '\n')
                    fw.flush()
                    
    print(f"Chunked documents written to {args.output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk the input document corpus into smaller chunks for processing")
    parser.add_argument("--tokenizer", type=str, help="The tokenizer to use for chunking the text.")
    parser.add_argument("--corpus_path", type=str, help="Path to the input document corpus file. Has been chunked")
    parser.add_argument("--chunk_strategy", type=str, choices=['hard', 'semantic', 'first'], default='hard', help="The chunking strategy to use")
    parser.add_argument("--chunk_size", type=int, help="The token size per chunk")
    parser.add_argument("--output_path", type=str, required=True, help="Default output path.")
    args = parser.parse_args()
    
    get_chunks(args)
    
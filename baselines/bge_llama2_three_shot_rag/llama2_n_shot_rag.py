import os
import re
import json
import asyncio
import aiohttp
import argparse
import random
from IPython import embed
from tqdm import tqdm
from collections import defaultdict

from torch.utils.data import DataLoader, Dataset


PROMPT = """
Answer the following question. Only return the answer without any other words.

{question}
"""

RAG_PROMPT = """
contextual information: {context}

question: {question}

Given the context information, answer the above question. You may disregard the context if it's not relevant.  Only return the answer without any other words.
""".strip("\n")

def parse_fn(result):
    text = result['choices'][0]['message']['content']
    text = text.strip('\n').strip(" ")
    is_succeed = True
    answer = text
    return is_succeed, answer        


class RagInferenceDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 max_ctx_num,
                 prompt_template):
        self.max_ctx_num = max_ctx_num
        self.prompt_template = prompt_template
        self.test_data = self._load_inference_data(data_path)
    
    def _load_inference_data(self, data_path):
        test_data = []
        with open(data_path, "r") as f:
            for line in tqdm(f, desc='loading inference query data...'):
                line = json.loads(line)
                _id = line['_id']
                question = line['question']
                if self.max_ctx_num == 0:
                    prompt = self.prompt_template.format(question=question)
                else:
                    # form context
                    top_psgs = [psg_item[1] for psg_item in line['top']]
                    context = []
                    for i in range(len(top_psgs)):
                        context.append("[{}] {}".format(i+1, top_psgs[i]))
                    context = "\n".join(context)
                    prompt = self.prompt_template.format(context=context, question=question)
                
                test_data.append({"_id": _id, "prompt": prompt})
               
        return test_data
    
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, item):
        return self.test_data[item]

class RagInferenceCollator:
    def __call__(self, batch):
        ids = []
        prompts = []
        for sample in batch:
            ids.append(sample['_id'])
            prompts.append(sample['prompt'])
        return {'sample_ids': ids, 'prompts': prompts}

async def generate_text(args,
                        session, 
                        config, 
                        prompt, 
                        parse_fn, 
                        sample_idx, 
                        output_result):
    """
    Asynchronously generates text based on the provided prompt and configuration.

    Args:
        args: Arguments passed to the function.
        session: Session object used for asynchronous operations.
        config (dict): Configuration settings provided as a dictionary.
        prompt (str): Initial text prompt to start generating text from.
        parse_fn: Function used for parsing text.
        sample_idx: Sample idx of this task.
        output_result: Variable to store the output.
    """

    response = await session.post(
        args.openai_api_base,
        headers={"Content-Type": "application/json"},
        json={
            "model": args.model_name_or_path,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            **config,
        },
    )
    response_json = await response.json()
    success, res = parse_fn(response_json)
    output_result[sample_idx]['_id'] = sample_idx
    output_result[sample_idx]['success'] = success
    output_result[sample_idx]['output'] = res
        
    return output_result


async def main(args):
    GEN_CONFIG = {"do_sample":True, "temperature": 0.8, "top_p": 0.95, "max_tokens": args.max_new_tokens}
    
    # 1. load inference dataset
    prompt_template = PROMPT if args.max_ctx_num == 0 else RAG_PROMPT
    dataset = RagInferenceDataset(args.input_data_path, args.max_ctx_num, prompt_template)
    collator = RagInferenceCollator()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)
    output_result =  defaultdict(dict)
    
    with open(args.output_path, "w") as fw:
        async with aiohttp.ClientSession() as session:        
            tasks = []
            
            # collect batch tasks
            for batch in tqdm(dataloader, desc="Performing inference..."):
                for i in range(len(batch['prompts'])):
                    task = asyncio.create_task(generate_text(args, session, GEN_CONFIG, batch['prompts'][i], parse_fn, batch['sample_ids'][i], output_result))
                    tasks.append(task)

                # wait for the batch tasks to complete
                await asyncio.gather(*tasks)
                
                # write batch output
                for key in output_result:     
                    fw.write(json.dumps(output_result[key]) + '\n')
                    # fw.flush()

                output_result.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given the query and retrieved passages, perform RAG to generate the answer.")
    parser.add_argument("--openai_api_base", type=str, default="http://localhost:8000/v1/chat/completions", help="OpenAI API base URL.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--model_type", type=str, required=True, help="model type")
    parser.add_argument("--max_ctx_num", type=int, default=1, help="The number of passages to incorporate.")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max generated tokens for LLM.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Default output path.")
    args = parser.parse_args()
    
    asyncio.run(main(args))
    
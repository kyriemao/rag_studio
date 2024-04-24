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


GEN_LOSE_RATIONALE_PROMPT = """
Context: {context}

Question: {question}

Given the above context and the question, generate a response in two parts.
1. Rationale: Provide a concise rationale that explains how specific parts of the contextual information contributed to generating the correct answer. You should start by listing the helpful sentences from the contextual information that were instrumental in answering the question. Follow this by a brief explanation of how they enabled you to derive the correct answer. If none of the contextual information is helpful, output "None of the context is helpful to answer the question" in explanation.
2. Answer: Provide a concise and direct answer to the question, focusing solely on the conclusion derived from the rationale.

The output format should be:
Rationale: 
Helpful Sentence 1: $sentence_1
Helpful Sentence 2: $sentence_2
...
Explanation: $explanation or "None of the context is helpful to answer the question"

Answer: $Answer
""".strip('\n')


def parse_fn(result):
    text = result['choices'][0]['message']['content']
    text = text.strip('\n').strip(" ")
    helpful_pattern = r"Helpful Sentence \d+: \"(.*?)\""
    explanation_pattern = r"Explanation: (.*)"
    answer_pattern = r"Answer: (.*)"
         
    helpful_sentences = re.findall(helpful_pattern, text)
        
    explanation_match = re.search(explanation_pattern, text)
    explanation = explanation_match.group(1) if explanation_match else None

    answer_match = re.search(answer_pattern, text)
    answer = answer_match.group(1) if answer_match else None
        
    is_succeed = explanation is not None and answer is not None
    return is_succeed, {"helpful_sentences": helpful_sentences, "explanation": explanation, "answer": answer}



class GenLoseRationaleDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 max_ctx_num,
                 prompt_template,
                 ignore_ids):
        self.max_ctx_num = max_ctx_num
        self.prompt_template = prompt_template
        self.ignore_ids = ignore_ids
        self.test_data = self._load_data(data_path)
        

    def _load_data(self, data_path):
        test_data = []
        with open(data_path, "r") as f:
            for line in tqdm(f):
                line = json.loads(line)
                _id = line['_id']
                question = line['question']
                answer = line['answer']
                
                top_psgs = [psg_item[1] for psg_item in line['top_no_pos']][:self.max_ctx_num]
                no_pos_ctx_text = "\n".join(["[{}] {}".format(i+1, top_psgs[i]) for i in range(len(top_psgs))])
                no_pos_text = self.prompt_template.format(context=no_pos_ctx_text, question=question)

                pos_psgs = [psg_item[1] for psg_item in line['pos']]
                top_psgs = pos_psgs + top_psgs
                top_psgs = top_psgs[:self.max_ctx_num]
                random.shuffle(top_psgs)
                has_pos_ctx_text = "\n".join(["[{}] {}".format(i+1, top_psgs[i]) for i in range(len(top_psgs))])
                has_pos_text = self.prompt_template.format(context=has_pos_ctx_text, question=question)    

                has_pos_id = "{}_{}".format(_id, "has_pos")
                no_pos_id = "{}_{}".format(_id, "no_pos")

                if has_pos_id not in self.ignore_ids:
                    test_data.append({"_id": has_pos_id, "prompt": has_pos_text})         
                if no_pos_id not in self.ignore_ids:
                    test_data.append({"_id": no_pos_id, "prompt": no_pos_text})       
                
        return test_data
    
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, item):
        return self.test_data[item]

class GenLoseRationaleCollator:
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
    output_result[sample_idx]['e_lose'] = res
        
    return output_result


async def main(args):
    GEN_CONFIG = {"do_sample":True, "temperature": 0.8, "top_p": 0.95, "max_tokens": args.max_new_tokens}
    
    # 1. load inference dataset
    ignore_ids = set()
    if os.path.exists(args.output_path):
        keeped_lines = []
        with open(args.output_path, "r") as f:
            for line in tqdm(f):
                line = json.loads(line)
                if line['success']:
                    ignore_ids.add(line['_id'])
                    keeped_lines.append(line)
    
        with open(args.output_path, "w") as f:
            for line in keeped_lines:
                f.write(json.dumps(line) + '\n')

    print("Ignoring {} samples that have been generated.".format(len(ignore_ids)))
    dataset = GenLoseRationaleDataset(args.input_data_path, args.max_ctx_num, GEN_LOSE_RATIONALE_PROMPT, ignore_ids)
    collator = GenLoseRationaleCollator()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)
    output_result =  defaultdict(dict)
    
    with open(args.output_path, "a+") as fw:
        async with aiohttp.ClientSession() as session:        
            tasks = []
            
            # collect batch tasks
            for batch in tqdm(dataloader, desc="Generating lose rationale..."):
                for i in range(len(batch['prompts'])):
                    task = asyncio.create_task(generate_text(args, session, GEN_CONFIG, batch['prompts'][i], parse_fn, batch['sample_ids'][i], output_result))
                    tasks.append(task)

                # wait for the batch tasks to complete
                await asyncio.gather(*tasks)
                
                # write batch output
                for key in output_result:     
                    fw.write(json.dumps(output_result[key]) + '\n')
                    fw.flush()

                output_result.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the rationales and answers for questions.")
    parser.add_argument("--openai_api_base", type=str, default="http://localhost:8000/v1/chat/completions", help="OpenAI API base URL.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--max_ctx_num", type=int, default=2, help="The number of passages to incorporate.")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max generated tokens for LLM.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Default output path.")
    args = parser.parse_args()
    
    asyncio.run(main(args))
    
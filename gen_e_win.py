from IPython import embed
import os
import re
import json
import asyncio
import aiohttp
import argparse
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset


GEN_WIN_RATIONALE_PROMPT = """
Context:
{context}

Question:
{question}

Answer:
{answer}

Task:
Generate a concise rationale explaining how specific parts (sentences) of the provided context contribute to deriving the answer. If the context does not inform the answer, output: "None of the context is helpful to answer the question." 

Output Format:
Helpful Sentence 1: $sentence_1
Helpful Sentence 2: $sentence_2
...
Explanation:
$explanation or "None of the context is helpful to answer the question"
""".strip('\n')

class GenWinRationaleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class GenWinRationaleCollator:
    def __init__(self, prompt_template, is_seed, max_ctx_num):
        self.prompt_template = prompt_template
        self.is_seed = is_seed
        self.max_ctx_num = max_ctx_num
    
    def __call__(self, batch):
        ids = []
        prompts = []
        for sample in batch:
            ids.append(sample['_id'])
            question = sample['question']
            answer = sample['answer']
            pos_idx, pos_psg = sample['pos'][0]
            if self.is_seed:
                has_pos_id = "{}_{}".format(sample['_id'], "has_pos")
                has_pos_ctx_text = pos_psg
                prompt = self.prompt_template.format(context=has_pos_ctx_text,
                                                     question=question,
                                                     answer=answer)
                prompts.append(prompt)
                ids.append(has_pos_id)
            else:
                top_psgs = [ psg_item[1] for psg_item in sample['top_no_pos'] ][:self.max_ctx_num]
                no_pos_ctx = ["[{}]: {}".format(i+1, context[i]) for i in range(len(top_psgs))]
                no_pos_ctx_text = "\n\n".join(has_pos_ctx)
                prompt = self.prompt_template.format(context=no_pos_ctx_text,
                                                     question=question,
                                                     answer=answer)
                no_pos_id = "{}_{}".format(sample['_id'], "no_pos")
                ids.append(no_pos_id)
                
                context = [pos_psg] + top_psgs
                context = context[:self.max_ctx_num]
                random.shuffle(context)
                has_pos_ctx = ["[{}]: {}".format(i+1, context[i]) for i in range(len(context))]
                has_pos_ctx_text = "\n\n".join(has_pos_ctx)
                prompt = self.prompt_template.format(context=has_pos_ctx_text,
                                                    question=question,
                                                    answer=answer)
                has_pos_id = "{}_{}".format(sample['_id'], "has_pos")
                ids.append(has_pos_id)
                prompts.append(prompt)

        return {'_ids': ids, 'prompts': prompts}

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
    output_result[sample_idx]['e_win'] = res
    return output_result

    
def parse_fn(result):
    def extract_rationales(text):
        # Define patterns to match Helpful Sentence and Explanation sections
        helpful_pattern = r"Helpful Sentence \d+: \"(.*?)\""
        explanation_pattern = r"Explanation: (.*)"
        
        # Find all matches for helpful sentences
        helpful_sentences = re.findall(helpful_pattern, text)
        
        # Find the explanation match
        explanation_match = re.search(explanation_pattern, text)
        explanation = explanation_match.group(1) if explanation_match else None
        
        return helpful_sentences, explanation

    res = result['choices'][0]['message']['content']
    success = True
    try:
        helpful_sentences, explanation = extract_rationales(res)
        generated_rationales = {"helpful_sentences": helpful_sentences, "explanation": explanation}
        success = explanation is not None
        return success, generated_rationales
    except Exception as e:
        print(e)
        success = False
        return success, None
    

async def main(args):
    GEN_CONFIG = {"do_sample": True, "temperature": 0.8, "top_p": 0.95, "max_tokens": args.max_new_tokens}
    
    # 1. load exist psg_ids
    exist_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            for line in f:
                line = json.loads(line)
                if line['success']:
                    exist_ids.add(line['_id'])
                    
    data = []
    with open(args.input_data_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            if line['_id'] not in exist_ids:
                data.append(line)

    dataset = GenWinRationaleDataset(data)
    collator = GenWinRationaleCollator(prompt_template=GEN_WIN_RATIONALE_PROMPT, is_seed=args.is_seed, max_ctx_num=args.max_ctx_num)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)
    output_result =  defaultdict(dict)
                
    with open(args.output_path, "a+") as fw:
        async with aiohttp.ClientSession() as session:        
            tasks = []
            
            # collect batch tasks
            for batch in tqdm(dataloader, desc="Generating win rationale ..."):
                for i in range(len(batch['prompts'])):
                    task = asyncio.create_task(generate_text(args, session, GEN_CONFIG, batch['prompts'][i], parse_fn, batch['_ids'][i], output_result))
                    tasks.append(task)

                # wait for the batch tasks to complete
                await asyncio.gather(*tasks)
                
                # write batch output
                for key in output_result:     
                    fw.write(json.dumps(output_result[key]) + '\n')
                    fw.flush()

                output_result.clear()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate win rationale (e).")
    parser.add_argument("--openai_api_base", type=str, default="http://localhost:8000/v1/chat/completions", help="OpenAI API base URL.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--model_type", type=str, required=True, help="model type")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max generated tokens for LLM.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    
    parser.add_argument("--max_ctx_num", type=int, default=2, help="Max number of context passages.")
    parser.add_argument("--is_seed", action='store_true', help="Indicates if this is the seed round.")
    parser.add_argument("--input_data_path", type=str, help="Path to the input xy data for generating win rationale.")
    parser.add_argument("--output_path", type=str, required=True, help="Default output path.")
    args = parser.parse_args()
    
    asyncio.run(main(args))

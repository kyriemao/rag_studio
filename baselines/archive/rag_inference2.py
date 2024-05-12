import os
import re
import torch
import json
import asyncio
import aiohttp
import argparse
import random
from accelerate import Accelerator
from IPython import embed
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from rag_studio.models import load_model


RAG_PROMPT = """
{context}

Question: {question}
Answer:
""".strip("\n")


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

                if self.max_ctx_num > 0:
                    # form context
                    top_psgs = [psg_item[1] for psg_item in line['top']]
                    context = []
                    for i in range(len(top_psgs)):
                        context.append("[{}] {}".format(i+1, top_psgs[i]))
                    ctx_text = "\n\n".join(context)
                else:
                    ctx_text = ""
                prompt = self.prompt_template.format(context=ctx_text, question=question).lstrip(' ').strip('\n')
                # prompt = self.prompt_template.format(context=ctx_text, question=question)
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


def main(args):
    GEN_CONFIG = {"do_sample":True, "temperature": 0.001, "top_p": 0.95, "max_new_tokens": args.max_new_tokens}
    
    accelerator = Accelerator()
    
    with accelerator.main_process_first():
        # load model and inference data
        model, tokenizer = load_model(args, for_eval=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        prompt_template = RAG_PROMPT
        dataset = RagInferenceDataset(args.input_data_path, args.max_ctx_num, prompt_template)
        collator = RagInferenceCollator()
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    model, dataloader = accelerator.prepare(model, dataloader)
    accelerator.wait_for_everyone()   
    
    all_results = []
    all_ids = []
    for batch in tqdm(dataloader, desc="Inferencing..."):
        unwrapped_model = accelerator.unwrap_model(model)
        with torch.inference_mode():
            _ids = batch["sample_ids"]
            prompts = batch["prompts"]
            
            encodes = tokenizer(prompts, padding="longest", add_special_tokens=True, return_tensors="pt")
            input_ids = encodes.input_ids.to(unwrapped_model.device)
            attention_mask = encodes.attention_mask.to(unwrapped_model.device)
            output = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **GEN_CONFIG
            )
            output_texts = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            batch_results = []
            for i in range(len(output_texts)):
                batch_results.append(output_texts[i][len(prompts[i]):].strip(' '))

            all_batch_results = accelerator.gather_for_metrics(batch_results)
            all_batch_ids = accelerator.gather_for_metrics(_ids)
            all_results.extend(all_batch_results)
            all_ids.extend(all_batch_ids)

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        with open(args.output_path, 'w') as fw:
            assert len(all_ids) == len(all_results)
            for i in range(len(all_ids)):
                obj = {"_id": all_ids[i], "success":True, "output": all_results[i]}
                fw.write(json.dumps(obj) + "\n")
                fw.flush()
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given the query and retrieved passages, perform RAG to generate the answer.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--model_type", type=str, required=True, help="Model type.")
    parser.add_argument("--model_dtype", type=str, default='auto', help="Model dtype.")
    parser.add_argument("--max_ctx_num", type=int, default=1, help="The number of passages to incorporate.")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max generated tokens for LLM.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Default output path.")
    args = parser.parse_args()
    
    main(args)
    
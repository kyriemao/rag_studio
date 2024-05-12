from IPython import embed
from dataclasses import dataclass, field
from typing import Optional, List, Literal

import json
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import HfArgumentParser, TrainingArguments, Trainer

import sys
from rag_studio.utils import write_running_args
from rag_studio.models import load_model

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        metadata={"help": "Type of the pretrained model or model identifier from huggingface.co/models"}
    )
    model_dtype: Literal['bf16', 'fp16', 'fp32', 'auto'] = field(default="auto", metadata={"help": "the data type of the model"})
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA."})
    
@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "The path to the training data."})
    max_seq_len: int = field(default=2048, metadata={"help": "The maximum total input sequence length for SFT."})
    max_ctx_num: int = field(default=1, metadata={"help": "The maximum number of context examples for SFT."})
    use_data_percent: float = field(default=1.0, metadata={"help": "The percent of training data to use."})
    force_emptying_dir: bool = field(default=False, metadata={"help": "Whether to force empty the output directory."})
    

class SftDataset(Dataset):
    def __init__(self, 
                 train_data_path,
                 max_ctx_num,
                 use_data_percent: float=1.0):
        
        self.train_data_path = train_data_path
        self.use_data_percent = use_data_percent
        self.max_ctx_num = max_ctx_num
        self.train_data = self._load_training_data()
        
    def _load_training_data(self):
        train_data = []
        with open(self.train_data_path) as f:
            for line in tqdm(f):
                line = json.loads(line)
                question = line['question']
                answer = line['gold_answers'][0]
                
                if self.max_ctx_num > 0:
                    # build raft context -- for triviaqa, we only consider the retrieved passages
                    ctx_psgs = [psg_item[1] for psg_item in line['top']][:self.max_ctx_num]
                    ctx = ["[{}]: {}".format(i+1, ctx_psgs[i]) for i in range(len(ctx_psgs))]
                    ctx_text = "\n\n".join(ctx)
                else:
                    ctx_text = ""    
                train_data.append([ctx_text, question, answer])  
            
        if self.use_data_percent < 1.0:
            random.seed(7)
            return random.sample(train_data, int(len(train_data) * self.use_data_percent))
        else:
            return train_data    
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        return self.train_data[item]
    
        
class SftCollator:
    def __init__(self, data_args, tokenizer):
        self.tokenizer = tokenizer
        self.max_seq_len = data_args.max_seq_len
        self.template = "{context}\n\nQuestion: {question}\nAnswer: {answer} </s>"

    def __call__(self, batch):
        texts = [self.template.format(context=sample[0], question=sample[1], answer=sample[2]) for sample in batch]
        completion = [sample[2] + " " + self.tokenizer.eos_token for sample in batch]
        data = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_seq_len, add_special_tokens=True)
        data_completion = self.tokenizer(completion, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_seq_len, add_special_tokens=False)
        data_mask_reverse = 1 - data_completion["attention_mask"]
        data_mask = data_mask_reverse * -100
        data["labels"] = data["input_ids"].clone()
        data["labels"] *= data_completion["attention_mask"]
        data["labels"] += data_mask
        data = {k: v.cuda() for k, v in data.items()}
        
        return data
            

        

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    # 1. load model
    model, tokenizer = load_model(model_args, for_eval=False)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # 2. load data

    train_dataset = SftDataset(train_data_path=data_args.train_data_path,
                                max_ctx_num=data_args.max_ctx_num,
                                use_data_percent=data_args.use_data_percent)
    train_collator = SftCollator(data_args, tokenizer)
        
    # 3. train
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=lambda x: train_collator(x),
    )

    trainer.train()

    # 4. save model and training args
    trainer.save_model(training_args.output_dir)
        
    if dist.get_rank() == 0:
        write_running_args(training_args.output_dir, [model_args, data_args, training_args])
    
if __name__ == '__main__':
    main()
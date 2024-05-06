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

from trl import SFTTrainer
from datasets import Dataset
from transformers import HfArgumentParser, TrainingArguments

import sys
from rag_studio.utils import write_running_args
from rag_studio.models import load_model

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LLAMA2_TRAIN_QA_PROMPT = """
[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Answer the following question. Only return the answer without any other words.

{question} [/INST] {answer} </s>
""".strip("\n")

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
    response_template: str = field(default=None, metadata={"help": "The response template for SFT."})
    use_data_percent: float = field(default=1.0, metadata={"help": "The percent of training data to use."})
    force_emptying_dir: bool = field(default=False, metadata={"help": "Whether to force empty the output directory."})
    

class SFTDataset(Dataset):
    def __init__(self, 
                 train_data_path,
                 prompt_template,
                 use_data_percent: float=1.0):
        
        self.train_data_path = train_data_path
        self.prompt_template = prompt_template
        self.use_data_percent = use_data_percent
        self.train_data = self._load_training_data()
        
    def _load_training_data(self):
        train_data = []
        with open(self.train_data_path) as f:
            for line in tqdm(f):
                line = json.loads(line)
                question = line['question']
                answer = line['gold_answers'][0]
                text = self.prompt_template.format(question=question, answer=answer)      
                train_data.append(text)
            
        if self.use_data_percent < 1.0:
            random.seed(7)
            return random.sample(train_data, int(len(train_data) * self.use_data_percent))
        else:
            return train_data    
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        return self.train_data[item]
    
        
class SFTCollator:
    def __init__(self, data_args, tokenizer):
        self.tokenizer = tokenizer
        self.response_template = data_args.response_template
        self.max_seq_len = data_args.max_seq_len
        
    def __call__(self, examples):
        n_sample = len(examples)
        input_ids = [torch.tensor(example['input_ids']) for example in examples]
        attention_mask = [torch.tensor(example['attention_mask']) for example in examples]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        labels = input_ids.detach().clone()
        ignore_idx = -100
        labels[labels == self.tokenizer.pad_token_id] = ignore_idx
        
        # mask the prompt part
        response_template_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)[1:]
        for i in range(n_sample):
            response_token_ids_start_idx = None
            for idx in np.where(labels[i] == response_template_ids[0])[0]:
                if (
                    response_template_ids
                    == labels[i][idx: idx + len(response_template_ids)].tolist()
                ):
                    response_token_ids_start_idx = idx

            if response_token_ids_start_idx is None:
                logger.warning(
                    f"Could not find response key `{response_template_ids}` in the "
                    f'following instance: {labels[i]} '
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                labels[i, :] = ignore_idx
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + len(response_template_ids)
                # Make pytorch loss function ignore all tokens up through the end of the response key
                labels[i, :response_token_ids_end_idx] = ignore_idx
   
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}

    

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    # 1. load model
    model, tokenizer = load_model(model_args, for_eval=False)

    # 2. load data
    train_dataset = SFTDataset(train_data_path=data_args.train_data_path,
                                    prompt_template=LLAMA2_TRAIN_QA_PROMPT,
                                    use_data_percent=data_args.use_data_percent)
    train_hf_dataset = {"text": [train_dataset[i] for i in range(len(train_dataset))]}
    train_hf_dataset = Dataset.from_dict(train_hf_dataset)
    train_collator = SFTCollator(data_args, tokenizer)
        
    # 3. train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_len,
        train_dataset=train_hf_dataset,
        data_collator=lambda x: train_collator(x),
        dataset_text_field="text",
        dataset_num_proc=64,
    )

    trainer.train()

    # 4. save model and training args
    trainer.save_model(training_args.output_dir)
        
    if dist.get_rank() == 0:
        write_running_args(training_args.output_dir, [model_args, data_args, training_args])
    
if __name__ == '__main__':
    main()
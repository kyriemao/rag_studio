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

from trl import DPOTrainer
from datasets import Dataset
from transformers import HfArgumentParser, TrainingArguments

from utils import write_running_args
from models import load_model

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DPO_LLAMA2_INPUT_PROMPT = """
[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Context: {context}

Question: {question}

Given the above context and the question, generate a response in two parts.
1. Rationale: Provide a concise rationale that explains how specific parts of the contextual information contributed to generating the correct answer. You should start by listing the helpful sentences from the contextual information that were instrumental in answering the question. Follow this by a brief explanation of how they enabled you to derive the correct answer. If none of the contextual information is helpful, output "None of the contextual information is helpful to answer the question".

2. Answer: Provide a concise and direct answer to the question, focusing solely on the conclusion derived from the rationale.

Output Format:
Rationale: 
Helpful Sentence 1: $sentence_1
Helpful Sentence 2: $sentence_2 
...
Explanation: $explanation or "None of the contextual information is helpful to answer the question"

Answer: $Answer [/INST]
""".strip('\n') 



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
    train_data_path: str = field(default=None, metadata={"help": "the training data path"})
    max_seq_len: int = field(default=2048, metadata={"help": "The maximum total input sequence length for SFT."})
    max_ctx_num: int = field(default=5, metadata={"help": "The maximum number of context passages."})
    use_data_percent: float = field(default=1.0, metadata={"help": "The percent of training data to use."})
    force_emptying_dir: bool = field(default=False, metadata={"help": "Whether to force empty the output directory."})


class DPOTrainDataset(Dataset):
    def __init__(self, 
                 train_data_path,
                 dpo_input_prompt,
                 max_ctx_num,
                 use_data_percent: float=1.0):
        
        self.train_data_path = train_data_path
        self.dpo_input_prompt = dpo_input_prompt
        self.max_ctx_num = max_ctx_num
        self.use_data_percent = use_data_percent
        self.train_data = self._load_training_data()
        
    """
    train data format:
    {
        "_id":
        "query": 
        "answer": 
        "pos":
        "top_no_pos":
        "e_win": {"has_pos", "no_pos"}
        "e_lose": {"has_pos", "no_pos"}
    }
    """
    def _load_training_data(self):
        train_data = []
        
        with open(self.train_data_path) as f:
            for line in tqdm(f):
                line = json.loads(line)
                question = line['question']
                answer = line['answer']
                
                # form [x, y_win, y_lose], has_pos and no_pos
                # has_pos
                if 'has_pos' in line['e_win'] and 'has_pos' in line['e_lose']:
                    y_win_has_pos_helpful_sentences = line['e_win']['has_pos']['helpful_sentences']
                    y_win_has_pos_explanation = line['e_win']['has_pos']['explanation']
                    y_lose_has_pos_helpful_sentences = line['e_lose']['has_pos']['helpful_sentences']
                    y_lose_has_pos_explanation = line['e_lose']['has_pos']['explanation']
                    y_lose_has_pos_answer = line['e_lose']['has_pos']['answer']
                    
                    y_win_has_pos = self._form_rationale(y_win_has_pos_helpful_sentences, y_win_has_pos_explanation, answer)
                    y_lose_has_pos = self._form_rationale(y_lose_has_pos_helpful_sentences, y_lose_has_pos_explanation, y_lose_has_pos_answer)

                    # x_has_pos
                    top_psgs = [ psg_item[1] for psg_item in line['top_no_pos'] ][:self.max_ctx_num]
                    pos_psgs = [ psg_item[1] for psg_item in line['pos'] ]
                    ctx_psgs = pos_psgs + top_psgs
                    ctx_psgs = ctx_psgs[:self.max_ctx_num]
                    has_pos_ctx = ["[{}]: {}".format(i+1, ctx_psgs[i]) for i in range(len(ctx_psgs))]
                    has_pos_ctx_text = "\n\n".join(has_pos_ctx)
                    x_has_pos = self.dpo_input_prompt.format(context=has_pos_ctx_text,
                                                             question=question)
                    
                    train_data.append([x_has_pos, y_win_has_pos, y_lose_has_pos])
                
                # no_pos
                if 'no_pos' in line['e_win'] and 'no_pos' in line['e_lose']:
                    y_win_no_pos_helpful_sentences = line['e_win']['no_pos']['helpful_sentences']
                    y_win_no_pos_explanation = line['e_win']['no_pos']['explanation']
                    y_lose_no_pos_helpful_sentences = line['e_lose']['no_pos']['helpful_sentences']
                    y_lose_no_pos_explanation = line['e_lose']['no_pos']['explanation']
                    y_lose_has_pos_answer = line['e_lose']['no_pos']['answer']

                    y_win_no_pos = self._form_rationale(y_win_no_pos_helpful_sentences, y_win_no_pos_explanation, answer)
                    y_lose_no_pos = self._form_rationale(y_lose_no_pos_helpful_sentences, y_lose_no_pos_explanation, y_lose_has_pos_answer)

                    # x_no_pos
                    ctx_psgs = [ psg_item[1] for psg_item in line['top_no_pos'] ][:self.max_ctx_num]
                    no_pos_ctx = ["[{}]: {}".format(i+1, ctx_psgs[i]) for i in range(len(ctx_psgs))]
                    no_pos_ctx_text = "\n\n".join(no_pos_ctx)
                    x_no_pos = self.dpo_input_prompt.format(context=no_pos_ctx_text,
                                                            question=question)

                    train_data.append([x_no_pos, y_win_no_pos, y_lose_no_pos])
                
                

        if self.use_data_percent < 1.0:
            random.seed(7)
            return random.sample(train_data, int(len(train_data) * self.use_data_percent))
        else:
            return train_data    
    
    def _form_rationale(self, helpful_sentences, explanation, answer):
        rationale = []
        if len(helpful_sentences) == 0:
            rationale.append("Helpful Sentence: None")
        else:
            rationale.extend(["Helpful Sentence {}: {}".format(i+1, helpful_sentences[i]) for i in range(len(helpful_sentences))])
        rationale.append("Explanation: {}".format(explanation))
        rationale = "Rationale:\n{}\n\nAnswer:{}".format("\n".join(rationale), answer)
        return rationale


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        return self.train_data[item]
    

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    # 1. load model
    model, tokenizer = load_model(model_args, for_eval=False)
    model_ref, _ = load_model(model_args, for_eval=False)
    
    # 2. load data
    if model_args.model_type in ['llama', 'llama-lora']:
        dpo_input_prompt = DPO_LLAMA2_INPUT_PROMPT
    else:
        raise NotImplementedError("Only support llama2 model types.")
    
    train_dataset = DPOTrainDataset(train_data_path=data_args.train_data_path,
                                    dpo_input_prompt=dpo_input_prompt,
                                    max_ctx_num=data_args.max_ctx_num,
                                    use_data_percent=data_args.use_data_percent)
    
    # print(train_dataset[0][0])
    # embed()
    # input()
    
    
    train_hf_dataset = {"prompt": [train_dataset[i][0] for i in range(len(train_dataset))],
                        "chosen": [train_dataset[i][1] for i in range(len(train_dataset))],
                        "rejected": [train_dataset[i][2] for i in range(len(train_dataset))]}
    train_hf_dataset = Dataset.from_dict(train_hf_dataset)
        
    # 3. train
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_len,
        max_prompt_length=2304,
        train_dataset=train_hf_dataset
    )

    dpo_trainer.train()

    # 4. save model and training args
    dpo_trainer.save_model(training_args.output_dir)
        
    if dist.get_rank() == 0:
        write_running_args(training_args.output_dir, [model_args, data_args, training_args])
    
if __name__ == '__main__':
    main()
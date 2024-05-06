from IPython import embed
from dataclasses import dataclass, field
from typing import Optional, List, Literal

import math
import json
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset

from transformers import Trainer
from transformers import HfArgumentParser, TrainingArguments

from utils import write_running_args
from models import load_model, retriever_forward

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
    normalize_emb: bool = field(default=False)


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data"})
    neg_type: Literal['hard', 'in_batch'] = field(default='in_batch', metadata={"help": "the type of negative samples"})
    neg_num: int = field(default=0, metadata={"help": "the number of negative samples"})
    
    max_q_len: int = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_p_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    
    use_data_percent: float = field(default=1.0, metadata={"help": "The percent of training data to use."})
    force_emptying_dir: bool = field(default=False, metadata={"help": "force to emptying the dir"})


@dataclass
class MatchingTrainingArguments:
    min_lr: float = field(default=0.0, metadata={"help": "The minimum learning rate in the cosine annealing scheduler."})
    temperature: float = field(default=0.01, metadata={"help": "The temperature of the ranking loss."})


class RetrieverTrainDataset(Dataset):
    def __init__(self, 
                 train_data_path,
                 neg_num,
                 use_data_percent: float=1.0):
        
        self.train_data_path = train_data_path
        self.neg_num = neg_num
        self.use_data_percent = use_data_percent
        self.data = self._load_training_data()
        
    def _load_training_data(self):
        train_data = []
        with open(self.train_data_path) as f:
            for line in tqdm(f):
                line = json.loads(line)
                if 'hard_neg' not in line:
                    neg_psgs = []
                else:
                    neg_psg_ids, neg_psgs = [], []
                    for psg_item in line['hard_neg']:
                        neg_psg_ids.append(psg_item[0])
                        neg_psgs.append(psg_item[1])
                neg_psgs = neg_psgs[:self.neg_num]
                for (pos_psg_id, pos_psg) in line['pos']:
                    train_data.append({'question': line['question'], 'pos_psg': pos_psg, 'neg_psgs': neg_psgs})
        
        if self.use_data_percent < 1.0:
            random.seed(7)
            return random.sample(train_data, int(len(train_data) * self.use_data_percent))
        else:
            return train_data    
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return [self.data[item]['question'], self.data[item]['pos_psg'], self.data[item]['neg_psgs']]
    
        
class RetrieverTrainCollator:
    def __init__(self, data_args, tokenizer):
        self.tokenizer = tokenizer
        self.max_p_len = data_args.max_p_len
        self.max_q_len = data_args.max_q_len
        
    def __call__(self, batch):
        query_inputs = []
        pos_psg_inputs = []
        neg_psg_inputs = []
        
        for i in range(len(batch)):
            query_inputs.append(batch[i][0])
            pos_psg_inputs.append(batch[i][1])
            neg_psg_inputs.extend(batch[i][2])
        query_input_encodings = self.tokenizer(query_inputs, padding="longest", max_length=self.max_q_len, truncation=True, return_tensors='pt')  
        pos_psg_input_encodings = self.tokenizer(pos_psg_inputs, padding="longest", max_length=self.max_p_len, truncation=True, return_tensors='pt')      
        if len(neg_psg_inputs) > 0:
            neg_psg_input_encodings = self.tokenizer(neg_psg_inputs, padding="longest", max_length=self.max_p_len, truncation=True, return_tensors='pt')
        else:
            neg_psg_input_encodings = None
        return  {"query_input_encodings": query_input_encodings,
                 "pos_psg_input_encodings": pos_psg_input_encodings,
                 "neg_psg_input_encodings": neg_psg_input_encodings}


class RankingLoss:
    def __init__(self, temperature):
        self.temperature = temperature
        self.loss_func = nn.CrossEntropyLoss()
    
    def __call__(self, query_embs, pos_psg_embs, neg_psg_embs):
        '''
        query_embs: B * dim (B is actuall n_query of this gpu)
        pos_psg_embs: (B + x) * dim (x is the total number of postive psgs of other gpus)
        neg_psg_embs: x' * dim, x' is the total number of negative psgs of other gpus, Optional
        '''
        n_query = len(query_embs) # n_query = per_gpu_batch_size
        doc_embs = pos_psg_embs
        if neg_psg_embs is not None:
            doc_embs = torch.cat([pos_psg_embs, neg_psg_embs], dim=0)
        score_mat = query_embs.mm(doc_embs.T) # n_query * n_psgs
        score_mat /= self.temperature
        label_mat = torch.arange(n_query).to(query_embs.device)  # only the first n_query docs are positive
        loss = self.loss_func(score_mat, label_mat)

        return loss
    
class RetrieverTrainer(Trainer):
    def __init__(self, 
                 model_args,
                 match_training_args, 
                 **kwargs):
        super(RetrieverTrainer, self).__init__(**kwargs)
        self.model_type = model_args.model_type
        self.normalize_emb = model_args.normalize_emb
        
        self.ranking_loss_func = RankingLoss(match_training_args.temperature)
        
        if match_training_args.min_lr > 0:
            assert self.args.lr_scheduler_type == 'cosine'
            assert match_training_args.min_lr < self.args.learning_rate
            num_cycles = self.get_num_cycles_for_cosine_lr_scheduler(self.args.learning_rate, match_training_args.min_lr)
            self.args.lr_scheduler_kwargs['num_cycles'] = num_cycles

    
    def _dist_gather_tensor(self, t: Optional[torch.Tensor], emb_dim: int, dtype):
        '''
        Support gathering different sizes of tensors (even 0) from the other gpus through padding
        refer to https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
        '''
        if t is not None:
            t = t.contiguous()
        
        cuda_device = f'cuda:{torch.distributed.get_rank()}'
        world_size = dist.get_world_size()
        local_size = torch.tensor(t.shape[0] if t is not None else 0, device=cuda_device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        size_diff = max(all_sizes).item() - local_size.item()
        # if all gpus have no data, return None
        if max(all_sizes).item() == 0:
            return None
        if size_diff > 0:
            padding = torch.zeros((size_diff, emb_dim), device=cuda_device, dtype=dtype)
            t = torch.cat((t, padding)) if t is not None else padding
            
        all_tensors_padded = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(all_tensors_padded, t)
        # cut the padding
        all_tensors = []
        for iter_t, size in zip(all_tensors_padded, all_sizes):
            all_tensors.append(iter_t[:size])
        # always put tensors of the current rank at the first place
        all_tensors[dist.get_rank()] = t
        all_tensors.pop(dist.get_rank())
        all_tensors = [t] + all_tensors
    
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
    
    
    def get_num_cycles_for_cosine_lr_scheduler(self, init_lr, min_lr):
        y = 2 * (min_lr / init_lr) - 1
        num_cycles = math.acos(y) / math.pi * 0.5
        return num_cycles
    
    
    def compute_loss(self, model, inputs):
        query_input_encodings = inputs.pop('query_input_encodings')
        pos_psg_input_encodings = inputs.pop('pos_psg_input_encodings')
        neg_psg_input_encodings = inputs.pop('neg_psg_input_encodings')
        
        query_embs = retriever_forward(model, self.model_type, query_input_encodings, self.normalize_emb)
        pos_psg_embs = retriever_forward(model, self.model_type, pos_psg_input_encodings, self.normalize_emb)
        neg_psg_embs = None
        if neg_psg_input_encodings:
            neg_psg_embs = retriever_forward(model, self.model_type, neg_psg_input_encodings, self.normalize_emb)

        emb_dim = query_embs.shape[1] # for cross gpu broadcasting
        dtype = query_embs.dtype # for cross gpu broadcasting
        pos_psg_embs = self._dist_gather_tensor(pos_psg_embs, emb_dim, dtype)
        neg_psg_embs = self._dist_gather_tensor(neg_psg_embs, emb_dim, dtype)
        ranking_loss = self.ranking_loss_func(query_embs, pos_psg_embs, neg_psg_embs)

        return ranking_loss    
    


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, MatchingTrainingArguments, TrainingArguments))
    model_args, data_args, match_training_args, training_args = parser.parse_args_into_dataclasses()
        
    # 1. load model
    model, tokenizer = load_model(model_args, for_eval=False)

    # 2. load data
    train_dataset = RetrieverTrainDataset(train_data_path=data_args.train_data_path,
                                          neg_num=data_args.neg_num,
                                          use_data_percent=data_args.use_data_percent)
    train_collator = RetrieverTrainCollator(data_args, tokenizer)

    # 3. train
    trainer = RetrieverTrainer(model_args=model_args,
                               match_training_args=match_training_args,
                               model=model,
                               tokenizer=tokenizer,
                               train_dataset=train_dataset,
                               data_collator=train_collator,
                               args=training_args)
    trainer.train()

    # 4. save model and training args
    trainer.save_model(training_args.output_dir)
        
    if dist.get_rank() == 0:
        write_running_args(training_args.output_dir, [model_args, data_args, match_training_args, training_args])
    
    
if __name__ == '__main__':
    main()
from IPython import embed
from dataclasses import dataclass, field
from typing import Literal

import gc
import os
import json
from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import HfArgumentParser

from models import load_model, retriever_forward
from utils import pstore, pload, write_running_args, mkdirs

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_blocks(data_output_dir, block_id):
    n_gpus = torch.cuda.device_count()
    all_embs = []
    all_embids = []
    for rank in range(n_gpus):
        emb_output_path = os.path.join(data_output_dir, "emb_block.rank_{}.{}.pb".format(rank, block_id))
        embid_output_path = os.path.join(data_output_dir, "embid_block.rank_{}.{}.pb".format(rank, block_id)) 
        emb = pload(emb_output_path)
        embid = pload(embid_output_path)
        
        all_embs.append(emb)
        all_embids.append(embid)
    all_embs = np.concatenate(all_embs, axis=0)
    all_embids = np.concatenate(all_embids, axis=0)
    
    emb_output_path = os.path.join(data_output_dir, "emb_block.{}.pb".format(block_id))
    embid_output_path = os.path.join(data_output_dir, "embid_block.{}.pb".format(block_id))
    pstore(all_embs, emb_output_path, high_protocol=True)
    pstore(all_embids, embid_output_path, high_protocol=True)
    
    # remove the previous sub-block files
    for rank in range(n_gpus):
        emb_output_path = os.path.join(data_output_dir, "emb_block.rank_{}.{}.pb".format(rank, block_id))
        embid_output_path = os.path.join(data_output_dir, "embid_block.rank_{}.{}.pb".format(rank, block_id)) 
        cmd = "rm {}".format(emb_output_path)
        os.system(cmd)
        cmd = "rm {}".format(embid_output_path)
        os.system(cmd)
        print("Removed {} and {}.".format(emb_output_path, embid_output_path))
        
        
def psg_generator(corpus_path: str, num_psg_per_block: int):
    batch_psgs = []
    with open(corpus_path, 'r') as f:
        for line in tqdm(f, desc='loading colletion {}...'.format(corpus_path)):
            obj = json.loads(line)
            psg_id = obj['_id']
            psg = obj['text']
            if 'title' in obj:
                title = obj['title']
            else:
                title = ""
            if len(title) > 0:
                psg = title + ". " + psg
            batch_psgs.append([psg_id, psg])
            if len(batch_psgs) == num_psg_per_block:
                yield batch_psgs
                batch_psgs = []
                gc.collect()
        yield batch_psgs



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
    corpus_path: str = field(
        metadata={"help": "the path to the corpus file"}
    )
    num_psg_per_block: int = field(
        default=1000000,
        metadata={
            "help": "the number of passages to store in each block"
        },
    )
    max_p_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for pasasge. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_output_dir: str = field(default=None, metadata={"help": "the dir to store outputs, which can be used for various tasks"})
    force_emptying_dir: bool = field(default=False, metadata={"help": "force to emptying the dir"})


@dataclass
class EvalArguments:
    per_device_eval_batch_size: int = field(default=64, metadata={"help": "eval batch size per device for inference"})
    
class IndexingCollator:
    def __init__(self, max_p_len, tokenizer):
        self.max_p_len = max_p_len
        self.tokenizer = tokenizer
         
    def __call__(self, batch: list):    
        sample_ids, psgs = zip(*batch)
        psgs = list(psgs)
        inputs = self.tokenizer(psgs, padding="longest", max_length=self.max_p_len, truncation=True, return_tensors='pt')
        inputs['sample_ids'] = sample_ids
        
        return inputs
    
    
@torch.no_grad()
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        if data_args.data_output_dir:
            mkdirs([data_args.data_output_dir], force_emptying_dir=data_args.force_emptying_dir)
        write_running_args(data_args.data_output_dir, [model_args, data_args, eval_args])
        
    
    # 1.load model
    model, tokenizer = load_model(model_args, for_eval=True)
    model.to("cuda")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 2. load data
    dataset_generator = psg_generator(data_args.corpus_path, data_args.num_psg_per_block)
    index_collator = IndexingCollator(data_args.max_p_len, tokenizer)
    
    for cur_block_id, psgs in tqdm(enumerate(dataset_generator), desc="Dense indexing..."):
        psg_ids = []
        psg_embs = []
        distributed_sampler = DistributedSampler(psgs)
        dataloader =  DataLoader(psgs, 
                                 sampler=distributed_sampler,
                                 batch_size=eval_args.per_device_eval_batch_size, 
                                 collate_fn=index_collator,
                                 shuffle=False)

        for batch in tqdm(dataloader, desc="Generating block #{}".format(cur_block_id), position=0, leave=True):
            inputs = {k: v.to("cuda") for k, v in batch.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            embs = retriever_forward(model, model_args.model_type, inputs, model_args.normalize_emb)

            if embs.dtype != torch.float32:
                embs = embs.float()
            embs = embs.detach().cpu().numpy()
            psg_ids.extend(batch['sample_ids'])
            psg_embs.append(embs)
            
        psg_embs = np.concatenate(psg_embs, axis=0)
        psg_ids = np.array(psg_ids)
        embid_output_path = os.path.join(data_args.data_output_dir, "embid_block.rank_{}.{}.pb".format(dist.get_rank(), cur_block_id))
        emb_output_path = os.path.join(data_args.data_output_dir, "emb_block.rank_{}.{}.pb".format(dist.get_rank(), cur_block_id))
        
        # store
        pstore(psg_ids, embid_output_path, high_protocol=True)
        pstore(psg_embs, emb_output_path, high_protocol=True)

        psg_ids = []
        psg_embs = []
        dist.barrier()
        
        if local_rank == 0:
            logger.info("Start merging all blocks...")
            merge_blocks(data_args.data_output_dir, cur_block_id)
            
    if local_rank == 0:
        logger.info("All docs have been stored in {} blocks.".format(cur_block_id + 1))
        

if __name__ == "__main__":
    main()
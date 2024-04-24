from IPython import embed
from dataclasses import dataclass, field
from typing import Optional, List, Literal

import os
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

from faiss_tool import FaissTool
from utils import write_running_args, mkdirs
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
    index_dir: str = field(
        metadata={"help": "the dir to store the index files"}
    )
    query_data_path_list: List[str] = field(default=None, metadata={"help": "the list of query dataset names"})
    max_q_len: int = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    embedding_size: int = field(
        default=768,
        metadata={
            "help": "the size of the embeddings"
        },
    )
    
    
    
    data_output_dir: str = field(default=None, metadata={"help": "the dir to store outputs, which can be used for various tasks"})
    force_emptying_dir: bool = field(default=False, metadata={"help": "force to emptying the dir"})
    
    def __post_init__(self):
        if self.query_data_path_list:
            self.query_data_path_list = self.query_data_path_list[0].split(' ')


@dataclass
class RetrievalArguments:
    top_n: int = field(default=100, metadata={"help": "the number of passages to retrieve for each query"})
    per_device_eval_batch_size: int = field(default=64, metadata={"help": "eval batch size per device for inference"})
    

class RetrievalInferenceDataset(Dataset):
    def __init__(self, 
                 query_data_path_list):
        
        self.query_data_path_list = query_data_path_list
        self.data = self._load_query_data()
        
    def _load_query_data(self):
        query_data = []
        for query_data_path in tqdm(self.query_data_path_list):
            with open(query_data_path) as f:
                for line in tqdm(f):
                    line = json.loads(line)
                    sample_idx = line['_id']
                    query = line['query']
                    query_data.append({'sample_idx': sample_idx, 'query': query})
        
        return query_data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class RetrievalCollator:
    def __init__(self, data_args, tokenizer):
        self.tokenizer = tokenizer
        self.max_q_len = data_args.max_q_len
        
    def __call__(self, batch):
        sample_ids = []
        query_inputs = []

        for i in range(len(batch)):
            sample_ids.append(batch[i]['sample_idx'])
            query_inputs.append(batch[i]['query'])
        
        query_input_encodings = self.tokenizer(query_inputs, padding="longest", max_length=self.max_q_len, truncation=True, return_tensors='pt')  
  
        return  {"query_input_encodings": query_input_encodings,
                 "sample_ids": sample_ids}


@torch.no_grad()
def get_query_embs(model, model_args, retrieval_dataloader):
    model.eval()
    model.to("cuda")
    
    query_embs = []
    sample_ids = []
    for batch in tqdm(retrieval_dataloader, desc="Encoding queries..."):
        inputs = batch['query_input_encodings']
        inputs = {k: v.to("cuda")  for k, v in inputs.items() if k not in {"sample_ids"}}
        embs = retriever_forward(model, model_args.model_type, inputs, model_args.normalize_emb)
        if embs.dtype != torch.float32:
            embs = embs.float()
        embs = embs.detach().cpu().numpy()
        for i, sample_idx in enumerate(batch['sample_ids']):
            sample_ids.append(sample_idx)
            query_embs.append(embs[i].reshape(1, -1))

    query_embs = np.concatenate(query_embs, axis=0)
    query_embs = query_embs.astype(np.float32) if query_embs.dtype != np.float32 else query_embs

    logger.info("#Total queries for evaluation: {}".format(len(query_embs)))
    
    model.to("cpu")
    torch.cuda.empty_cache()

    return sample_ids, query_embs


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, RetrievalArguments))
    model_args, data_args, retrieval_args = parser.parse_args_into_dataclasses()
    if data_args.data_output_dir:
        mkdirs([data_args.data_output_dir], force_emptying_dir=data_args.force_emptying_dir)
    write_running_args(data_args.data_output_dir, [model_args, data_args, retrieval_args])
   
    # 1. get query embeddings
    model, tokenizer = load_model(model_args, for_eval=True)
    
    # 2. load data
    retrieval_dataset = RetrievalInferenceDataset(data_args.query_data_path_list)
    
    retrieval_collator = RetrievalCollator(data_args, tokenizer)
    retrieval_dataloader = DataLoader(retrieval_dataset, 
                                      batch_size=retrieval_args.per_device_eval_batch_size, 
                                      shuffle=False, 
                                      collate_fn=retrieval_collator)

    # 3. get query embeddings
    sample_ids, query_embs = get_query_embs(model, model_args, retrieval_dataloader)
    sample_id_list = [sample_ids[i:i + 10000] for i in range(0, len(sample_ids), 10000)]
    query_emb_list = [query_embs[i:i + 10000] for i in range(0, len(query_embs), 10000)]
    
    # 4. faiss retrieval
    faiss_tool = FaissTool(data_args.embedding_size, data_args.index_dir, retrieval_args.top_n)
    for batch_idx in range(len(query_emb_list)):
        query_embs = query_emb_list[batch_idx]
        sample_ids = sample_id_list[batch_idx]
        scores_mat, psg_ids_mat = faiss_tool.search_on_blocks(query_embs)
    
        # 5. output retrieval results
        logger.info('begin to write the retrieval output...')
        output_path = os.path.join(data_args.data_output_dir, 'run.{}.json'.format(batch_idx))
        output_trec_path = os.path.join(data_args.data_output_dir, 'run.{}.trec'.format(batch_idx))
        with open(output_path, "w") as fw, open(output_trec_path, "w") as fw_trec:
            for i in range(len(sample_ids)):
                sample_idx = sample_ids[i]
                rank = 0
                for psg_idx, score in zip(psg_ids_mat[i], scores_mat[i]):
                    fw.write(json.dumps({"sample_idx": str(sample_idx), "psg": "", "psg_idx": psg_idx, "rank": rank, "retrieval_score": score}) + "\n")
                    fw_trec.write("{}\tQ0\t{}\t{}\t{}\t{}".format(sample_idx, psg_idx, rank, retrieval_args.top_n-rank, model_args.model_name_or_path) + "\n")
                    rank += 1
                    
        logger.info('finish writing the retrieval output for batch {}'.format(batch_idx))


if __name__ == '__main__':
    main()

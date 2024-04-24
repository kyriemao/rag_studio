from IPython import embed

import os
import json
import random
import argparse

from tqdm import tqdm
from collections import defaultdict

def gen_xy(args):
    # 1. load corpus
    corpus = {}
    with open(args.corpus_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            psg_id = line['_id']
            psg_text = line['text']
            corpus[psg_id] = psg_text
    
    # 2. transform data     
    qid = 0
    with open(args.xy_path, "w") as fw:
        with open(args.xy_raw_path) as f:
            for line in tqdm(f):
                line = json.loads(line)
                if not line['success']:
                    continue
                psg_id = line['_id']
                for pair in line['generated_questions']:
                    question, answer = pair['question'], pair['answer']
                    d = {"_id": qid, "question": question, "answer": answer, "pos": [(psg_id, corpus[psg_id])]}               
                    fw.write(json.dumps(d) + '\n')
                    qid += 1
                    
    print("Done!")


def incorporate_hard_neg_to_training_data(args):
    # 1. load corpus
    corpus = {}
    with open(args.corpus_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            psg_id = line['_id']
            psg_text = line['text']
            corpus[psg_id] = psg_text
    
    # 2. load initial training data
    qa_data = {}
    with open(args.initial_data_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            qid = line['_id']
            qa_data[str(qid)] = line

    # 3. randomly select neg context passages for each sample
    context_psgs = defaultdict(list)
    retrieved_files = os.listdir(args.retrieved_data_path)
    for retrieved_file in retrieved_files:
        if not retrieved_file.endswith(".trec"):
            continue
        with open(os.path.join(args.retrieved_data_path, retrieved_file)) as f:
            for line in tqdm(f, desc='loading retrieved trec file...'):
                qid, _, psg_idx, rank, _, _ = line.strip('\n').split('\t')
                rank = int(rank)
                if rank >= 20 and rank < 50:
                    context_psgs[qid].append(psg_idx)
    
    for qid in tqdm(context_psgs,desc="randomly select neg context passages..."):
        pool = context_psgs[qid]
        pos_psg_ids = set([pair[0] for pair in qa_data[qid]['context']['pos']])
        neg_psg_ids = set()
        n_context_psgs = args.n_context_psgs
        while True:
            neg_psg_ids = neg_psg_ids | set(random.sample(pool, min(n_context_psgs, len(pool))))
            neg_psg_ids = neg_psg_ids - pos_psg_ids
            if len(neg_psg_ids) == args.n_context_psgs or len(pool) < n_context_psgs:
                break
            else:
                n_context_psgs = args.n_context_psgs - len(neg_psg_ids)
                pool = list(set(pool) - neg_psg_ids)
            
        neg_psg_ids = list(neg_psg_ids)
        context_psgs[qid] = neg_psg_ids
    
    # 4. form the training data with hard negatives
    with open(args.hard_neg_train_data_path, "w") as fw:
        for qid in tqdm(qa_data, desc="Forming the training data with hard negatives..."):
            line = qa_data[qid]
            neg = [(psg_id, corpus[psg_id]) for psg_id in context_psgs[qid]]
            line['context']['neg'] = neg
            fw.write(json.dumps(line) + '\n')
    
    print("Done!")

def add_ctx(args):
    # 1. load corpus
    corpus = {}
    with open(args.corpus_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            psg_id = line['_id']
            psg_text = line['text']
            corpus[psg_id] = psg_text
    
    # 2. load initial training data
    qa_data = {}
    with open(args.xy_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            qid = line['_id']
            qa_data[str(qid)] = line

    # 3. select top neg context passages for each sample
    top_context_psgs = defaultdict(list)
    retrieved_files = os.listdir(args.retrieved_data_path)
    for retrieved_file in retrieved_files:
        if not retrieved_file.endswith(".trec"):
            continue
        with open(os.path.join(args.retrieved_data_path, retrieved_file)) as f:
            for line in tqdm(f, desc='loading retrieved trec file...'):
                qid, _, psg_idx, rank, _, _ = line.strip('\n').split('\t')
                rank = int(rank)
                if rank <= args.n_context_psgs+3:
                    top_context_psgs[qid].append(psg_idx)
    
    for qid in tqdm(top_context_psgs):
        pos_psg_ids = set([pair[0] for pair in qa_data[qid]['pos']])
        selected_top_psgs = []
        for top_psg_idx in top_context_psgs[qid]:
            if top_psg_idx not in pos_psg_ids:
                selected_top_psgs.append(top_psg_idx)
                if len(selected_top_psgs) == args.n_context_psgs:
                    break
        top_context_psgs[qid] = selected_top_psgs
        
    
    with open(args.xy_ctx_path, "w") as fw:
        for qid in tqdm(qa_data, desc="Forming the training data with top context..."):
            line = qa_data[qid]
            top_no_pos = [(psg_id, corpus[psg_id]) for psg_id in top_context_psgs[qid]]
            line['top_no_pos'] = top_no_pos
            fw.write(json.dumps(line) + '\n')
            
    print("Done!")


def incorporate_ctx_to_training_data(args):
    # 1. load corpus
    corpus = {}
    with open(args.corpus_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            psg_id = line['_id']
            psg_text = line['text']
            corpus[psg_id] = psg_text
    
    # 2. load initial training data
    qa_data = {}
    with open(args.initial_data_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            qid = line['_id']
            qa_data[str(qid)] = line

    # 3. randomly select neg context passages for each sample
    context_psgs = defaultdict(list)
    retrieved_files = os.listdir(args.retrieved_data_path)
    for retrieved_file in retrieved_files:
        if not retrieved_file.endswith(".trec"):
            continue
        with open(os.path.join(args.retrieved_data_path, retrieved_file)) as f:
            for line in tqdm(f, desc='loading retrieved trec file...'):
                qid, _, psg_idx, rank, _, _ = line.strip('\n').split('\t')
                rank = int(rank)
                if rank <= args.n_context_psgs:
                    context_psgs[qid].append(psg_idx)
    
    # 4. form the final data which has context passages
    with open(args.ctx_data_path, "w") as fw:
        for qid in tqdm(qa_data, desc="Forming the training data with top ctx..."):
            line = qa_data[qid]
            top = [(psg_id, corpus[psg_id]) for psg_id in context_psgs[qid]]
            if 'context' not in line:
                line['context'] = {}
            line['context']['top'] = top
            fw.write(json.dumps(line) + '\n')
    
    print("Done!")


def add_e_win_lose(args):
    e_win = {}
    with open(args.e_win_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            qid = line['_id']
            if line['success']:
                e_win[qid] = line['e_win']
    e_lose = {}
    with open(args.e_lose_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            qid = line['_id']
            if line['success']:
                e_lose[qid] = line['e_lose']

    with open(args.xy_ctx_path) as f, open(args.xy_e_win_lose_path, "w") as fw:
        for line in tqdm(f):
            line = json.loads(line)
            _id = line['_id']
            has_pos_id = "{}_has_pos".format(_id)
            no_pos_id = "{}_no_pos".format(_id)
            line['e_win'] = defaultdict(dict)
            line['e_lose'] = defaultdict(dict)
            
            if has_pos_id in e_win and has_pos_id in e_lose:
                line['e_win']['has_pos'] = e_win[has_pos_id]
                line['e_lose']['has_pos'] = e_lose[has_pos_id]

            if no_pos_id in e_lose:
                line['e_lose']['no_pos'] = e_lose[no_pos_id]
            if args.is_seed:
                line['e_win']['no_pos'] = {"helpful_sentences": [], "explanation": "None of the context is helpful to answer the question."} 
            elif no_pos_id in e_win:
                line['e_win']['no_pos'] = e_win[no_pos_id]
            
            fw.write(json.dumps(line) + '\n')

    print("Done!")
    


def main(args):
    if args.stage == "gen_xy":
        gen_xy(args)     
    elif args.stage == "incorporate_hard_neg":
        incorporate_hard_neg_to_training_data(args)
    elif args.stage == "incorporate_ctx":
        incorporate_ctx_to_training_data(args)
    elif args.stage == "dpo_train":
        dpo_train(args)
    elif args.stage == "add_ctx":
        add_ctx(args)
    elif args.stage == "add_e_win_lose":
        add_e_win_lose(args)
    else:
        raise ValueError("Invalid stage.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collection of funcs for data preparation.")
    parser.add_argument("--stage", type=str, choices=["gen_xy", "add_e_win_lose", "add_ctx"], help="Stage of the data preparation.")
    parser.add_argument("--corpus_path", type=str, help="Path to the corpus data.")
    parser.add_argument("--xy_raw_path", type=str, help="Path to the raw xy data.")
    parser.add_argument("--xy_path", type=str, help="Path to the xy data.")
    parser.add_argument("--xy_ctx_path", type=str, help="Path to the xy with top ctx data")
    parser.add_argument("--n_context_psgs", type=int, default=5, help="Number of context passages to use.")
    parser.add_argument("--retrieved_data_path", type=str, help="Path to the the retrieved data.")
    parser.add_argument("--e_win_path", type=str, help="Path to the win rationales data.")
    parser.add_argument("--e_lose_path", type=str, help="Path to the lose rationales data.")
    parser.add_argument("--xy_e_win_lose_path", type=str, help="Path to the xy with rationales data.")
    parser.add_argument("--is_seed", action='store_true', help="Indicates if this is the seed round.")
    
    # parser.add_argument("--initial_data_path", type=str, help="Path to the initial data. (train.jsonl or test.jsonl)")
    
    # parser.add_argument("--hard_neg_train_data_path", type=str, help="Path to the training data with hard negatives.")
    # parser.add_argument("--ctx_data_path", type=str, help="Path to the data with context for rag.")
    # parser.add_argument("--rationale_data_path", type=str, help="Path to the rationales data.")
    # parser.add_argument("--dpo_data_path", type=str, help="DPO training data path.")
    # parser.add_argument("--pred_data_path", type=str, help="Path to the prediction data.")
    # parser.add_argument("--output_path", type=str, help="Path to the output file.")
    # parser.add_argument("--answer_judge_data_path", type=str, help="Path to the answer judge data.")
    

    args = parser.parse_args()
    main(args)
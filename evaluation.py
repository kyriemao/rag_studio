import re
import json
import string
import argparse
from IPython import embed
from collections import Counter, defaultdict

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def trivia_qa_eval(args):
    preds = {} 
    with open(args.pred_path) as f:
        for line in f:
            line = json.loads(line)
            preds[line['_id']] = line['output']

    oracle = {}
    with open(args.oracle_path) as f:
        for line in f:
            line = json.loads(line)
            oracle[line['_id']] = line['gold_answers']

    # EM score
    scores = defaultdict(dict)
    em = []
    f1 = []
    
    for qid in preds:
        prediction = preds[qid]
        ground_truths = oracle[qid]
        em_score = metric_max_over_ground_truths(cal_em_score, prediction, ground_truths)
        f1_score = metric_max_over_ground_truths(cal_f1_score, prediction, ground_truths)
        scores[qid]['em'] = em_score
        scores[qid]['f1'] = f1_score
        em.append(em_score)
        f1.append(f1_score)
    
    return {"em_core": round(sum(em) / len(em), 4), "f1_score": round(sum(f1) / len(f1), 4), "n": len(em)}
    

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def cal_em_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def cal_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given the query and retrieved passages, perform RAG to generate the answer.")
    parser.add_argument("--oracle_path", type=str, required=True, help="Path to the oracle answers.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the pred answers.")
    args = parser.parse_args()
    
    res = trivia_qa_eval(args)
    print(res)
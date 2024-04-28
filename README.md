# RAG Studio: Self-aligned Retrieval-Augmented Generation

## An Optimization Loop
Given corpus D, initial retriever R, and generator G, the self-aligned optimization loop has the following steps:

1. Generate questions and answers (gen_xy)

G(c) -> x, y

Input: corpus.jsonl, G

Output: xy.jsonl

```sh
bash run_scripts/gen_xy.sh
```

2. Retrieve top passages (no_pos)

R(x) -> c1, c2, c3

Input: corpus.jsonl, xy.jsonl, R

Output: xy.ctx.jsonl

```sh
# index -> retrieve -> add_ctx
bash run_scripts/get_xy_ctx.sh
```

3. Generate training rationales (e_win and e_lose)

Two input versions:
- has_pos: G(c+c'+x+y) -> e, G(c+c'+x) -> e1+y1
- no_pos: G(c'+x+y) -> e', G(c'+x) -> e2+y2

Input: xy.ctx.jsonl, G

Output: e_win.(seed).jsonl, e_lose.jsonl, **xy.e_win_lose.jsonl**


```sh
bash run_scripts/gen_e_win.seed.sh # add `--is_seed` if this is the seed round.
bash run_scripts/gen_e_lose.sh
bash run_scripts/get_xy_e_win_lose.sh
```


4. Train generator with DPO

Two versions
- has_pos: DPO(c+c'+x, e+y, e1+y1) if y1 != y
- no_pos: DPO(c'+x, e'+y, e2+y2) if y2 != y


Input: xy.e_win_lose.jsonl, G

Output: G_{next}

Two versions
DPO(x_ha)

Input: xy.rationale.jsonl, G

Output: G_{next}

```sh
bash run_scripts/train_dpo.sh
```

5. Train retriever with hard negative mining from the rationales

**Step1**: judge whether the model predicted answer is correct. We only optimize the samples where the predicted answers are wrong.

Input: xy.jsonl, e_lose.jsonl, G

Output: judgment.jsonl

```sh
bash run_scripts/gen_j.sh
```

**Step2**: identify the passages that contributing to the wrong predicted answer and use them as the hard negatives. The gold passaage are always positives. Then, we form the training data file for training the retriever.
```json
{"_id": str, "question": str, "pos": [], "neg": []}
```

Input:

Output:

```sh
bash run_scripts/
```
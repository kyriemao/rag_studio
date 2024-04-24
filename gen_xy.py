from IPython import embed
import os
import re
import json
import asyncio
import aiohttp
import argparse
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

GEN_QA_PROMPT = """
Passage: 
{passage}

Generate {num_question} questions and its corresponding answers based on the passage provided. Each question should:
- Be Self-contained: Each question should stand alone without referring to the passage directly. Avoid phrases like "according to the passage," or "as mentioned in the passage." The question should read as if it is a general query, not specifically designed for the passage.
- Be Precisely Answerable: Ensure that the question can be clearly and completely answered using only the information from the passage.
- Vary in Complexity: Include a question that requires an inferential level of comprehension.
- Evaluate Informativeness: If the passage does not contain enough information to form a meaningful or good question, state that the passage is insufficient.
- Objective: The question should be framed in such a way that it seems the questioner does not know of the passage's existence, and the passage coincidentally contains the information necessary to answer the question.



Output Format:
- If the passage is suitable to generate good question, your output format should be:
Question_1: $question_1
Answer_1: $answer_1
...
- If the passage lacks sufficient information, your should output statement "The passage does not provide enough information to formulate a meaningful or good question".
""".strip('\n')

class GenQuestionDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class GenQuestionCollator:
    def __init__(self, prompt_template, num_question):
        self.prompt_template = prompt_template
        self.num_question = num_question
    
    def __call__(self, batch):
        ids = []
        prompts = []
        for sample in batch:
            ids.append(sample['_id'])
            psg = sample['text']
            prompt = self.prompt_template.format(num_question=self.num_question,
                                                 passage=psg)
            prompts.append(prompt)
        return {'_ids': ids, 'prompts': prompts}

async def generate_text(args,
                        session, 
                        config, 
                        prompt, 
                        parse_fn, 
                        sample_idx, 
                        output_result):
    """
    Asynchronously generates text based on the provided prompt and configuration.

    Args:
        args: Arguments passed to the function.
        session: Session object used for asynchronous operations.
        config (dict): Configuration settings provided as a dictionary.
        prompt (str): Initial text prompt to start generating text from.
        parse_fn: Function used for parsing text.
        sample_idx: Sample idx of this task.
        output_result: Variable to store the output.
    """

    response = await session.post(
        args.openai_api_base,
        headers={"Content-Type": "application/json"},
        json={
            "model": args.model_name_or_path,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            **config,
        },
    )
    response_json = await response.json()
    success, res = parse_fn(response_json)
    output_result[sample_idx]['_id'] = sample_idx
    output_result[sample_idx]['success'] = success
    output_result[sample_idx]['generated_questions'] = res
    
    return output_result

    
def parse_fn(result):
    def extract_questions_answers(text):
        questions = []
        responses = []

        pattern = r'Question_(\d+):\s*(.*?)\nAnswer_\1:\s*(.*?)\n'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            instruction_num, instruction, response = match
            questions.append(instruction)
            responses.append(response)

        return questions, responses

    res = result['choices'][0]['message']['content']
    success = True
    try:
        questions, answers = extract_questions_answers(res)
        generated_questions = [{"question": question, "answer": answer} for question, answer in zip(questions, answers)]
        return success, generated_questions
    except Exception as e:
        print(e)
        success = False
        return success, None
    

async def main(args):
    GEN_CONFIG = {"do_sample":True, "temperature": 0.8, "top_p": 0.95, "max_tokens": args.max_new_tokens}
    
    # 1. load exist psg_ids
    exist_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            for line in f:
                line = json.loads(line)
                if line['success']:
                    exist_ids.add(line['_id'])
                    
    data = []
    with open(args.corpus_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            if line['_id'] not in exist_ids:
                data.append(line)
            if len(data) == args.num_passage:
                break

    dataset = GenQuestionDataset(data)
    collator = GenQuestionCollator(prompt_template=GEN_QA_PROMPT,
                                      num_question=args.num_question)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)
    output_result =  defaultdict(dict)
                
    with open(args.output_path, "a+") as fw:
        async with aiohttp.ClientSession() as session:        
            tasks = []
            
            # collect batch tasks
            for batch in tqdm(dataloader, desc="Question and answer generation..."):
                for i in range(len(batch['prompts'])):
                    task = asyncio.create_task(generate_text(args, session, GEN_CONFIG, batch['prompts'][i], parse_fn, batch['_ids'][i], output_result))
                    tasks.append(task)

                # wait for the batch tasks to complete
                await asyncio.gather(*tasks)
                
                # write batch output
                for key in output_result:     
                    fw.write(json.dumps(output_result[key]) + '\n')
                    # fw.flush()

                output_result.clear()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions based on the given passage chunks.")
    parser.add_argument("--openai_api_base", type=str, default="http://localhost:8000/v1/chat/completions", help="OpenAI API base URL.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--model_type", type=str, required=True, help="model type")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max generated tokens for LLM.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    
    parser.add_argument("--corpus_path", type=str, help="Path to the input document corpus file. Has been chunked")
    parser.add_argument("--num_question", type=int, help="Number of questions to generate.")
    parser.add_argument("--num_passage", type=int, help="Number of passages used to generate.")
    parser.add_argument("--output_path", type=str, required=True, help="Default output path.")
    args = parser.parse_args()
    
    asyncio.run(main(args))
    
import re
from tqdm import tqdm
from datasets import load_dataset

import numpy as np

from pissa.benchmark.tools import generate_sample

# Prompts
ALPACA_PROMPT = \
        "Below is an instruction that describes a task. " \
        "Write a response that appropriately completes the request.\n\n" \
        "### Instruction:\n{instruction}\n\n### Response: "

COMMONSENSE_TEST_SUITE = {
    'boolq': {'file_path': 'data/commonsense_reasoning/boolq/test.json'},
    'siqa': {'file_path': 'data/commonsense_reasoning/social_i_qa/test.json'},
    'arc_c': {'file_path': 'data/commonsense_reasoning/ARC-Challenge/test.json'},
    'arc_e': {'file_path': 'data/commonsense_reasoning/ARC-Easy/test.json'},
    'winogrande': {'file_path': 'data/commonsense_reasoning/winogrande/test.json'},
}


def extract_answer(dataset, sentence: str) -> float:
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['siqa', 'arc_c', 'arc_e']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


def eval_commonsense(model, tokenizer, dataset_name, batch_size=4):
    if dataset_name in COMMONSENSE_TEST_SUITE:
        test = load_dataset("json", data_files=COMMONSENSE_TEST_SUITE[dataset_name]["file_path"])
        test = test['train']  # designate the split
    else:
        raise NotImplementedError(f"Unsupported dataset{dataset_name}")

    model.eval()

    acc_res = []
    dataset_size = len(test)
    for i in tqdm(range(0, dataset_size, batch_size), desc="Generating answers"):
        start = i
        end = min(dataset_size, i + batch_size)
        prompts = [ALPACA_PROMPT.format(instruction=test[start + idx]["instruction"]) for idx in range(end - start)]
        answers = [test[start + idx]["answer"] for idx in range(end - start)]
        completions = generate_sample(model, tokenizer, prompts, max_new_tokens=32)
        for completion, label in zip(completions, answers):
            pred = extract_answer(dataset_name, completion.split("### Response:")[1])
            acc = pred == label
            acc_res.append(acc)

    accuracy = np.mean(acc_res)
    print(f"{dataset_name} Accracy:", accuracy, flush=True)
    return accuracy

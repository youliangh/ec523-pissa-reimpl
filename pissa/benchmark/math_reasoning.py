import re
import torch
import numpy as np
import transformers
from datasets import load_from_disk, load_dataset
from tqdm import tqdm

from pissa.benchmark.tools import generate_sample

MATH_PROMPT = \
        "Below is an instruction that describes a task. " \
        "Write a response that appropriately completes the request.\n\n" \
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


# LLM-Adapters Test suite
MATH_REASONING_TEST_SUITE = {
    "AddSub": {'file_path': "data/math_reasoning/AddSub/test.json"},
    "AQuA": {'file_path': "data/math_reasoning/AQuA/test.json"},
    "MAWPS": {"file_path": "data/math_reasoning/mawps/test.json"},
    "SVAMP": {"file_path": "data/math_reasoning/SVAMP/test.json"},
    "gsm8k": {"file_path": "data/math_reasoning/gsm8k/test.json"},
    "SingleEq": {"file_path": "data/math_reasoning/SingleEq/test.json"}
}


def extract_answer_number(completion):
    completion = completion.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', completion)]
    if not pred:
        return float("inf")
    pred_answer = float(pred[-1])
    
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(completion: str) -> str:
    sentence_ = completion.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def is_correct(completion, answer, dataset_name):
    if dataset_name == "AQuA":
        pred = extract_answer_letter(completion)
        return pred == answer
    else:
        ans = answer
        if isinstance(ans, str):
            ans = float(ans)
    assert ans != INVALID_ANS, "No ground truth answer found in the document."
    return abs(extract_answer_number(completion) - ans) < 1e-3


def eval_math(model, tokenizer, dataset_name, batch_size=4):
    if dataset_name in MATH_REASONING_TEST_SUITE:
        test = load_dataset("json", data_files=MATH_REASONING_TEST_SUITE[dataset_name]["file_path"])
        test = test['train']  # designate the split
    else:
        raise NotImplementedError(f"Unsupported dataset {dataset_name}")

    model.eval()

    acc_res = []
    dataset_size = len(test)
    for i in tqdm(range(0, dataset_size, batch_size), desc="Generating answers"):
        start = i
        end = min(dataset_size, i + batch_size)
        prompts = [MATH_PROMPT.format(instruction=test[start + idx]["instruction"]) for idx in range(end - start)]
        answers = [test[start + idx]["answer"] for idx in range(end - start)]
        completions = generate_sample(model, tokenizer, prompts)
        for completion, answer in zip(completions, answers):
            acc = is_correct(completion.split("### Response:")[1].strip(), answer, dataset_name)
            acc_res.append(acc)

    accuracy = np.mean(acc_res)
    print(f"{dataset_name} Accracy:", accuracy, flush=True)
    return accuracy

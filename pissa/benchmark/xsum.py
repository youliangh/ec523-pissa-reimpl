import torch

import nltk

from tqdm import tqdm
from datasets import load_dataset
import evaluate
import transformers

from pissa.data import XSUM_PROMPT, XSUM_ANSWER

metric = evaluate.load("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def extract_summary(completion: str):
    split_list = completion.split("\n\n")
    if len(split_list) == 2:
        return split_list[1]
    elif len(split_list) < 2:
        print("Warning: Invalid xSum completion")
        return ""
    else:
        return "\n\n".join(split_list[1:])


def eval_xsum(model, tokenizer, source_max_len, cache_dir, batch_size=4):
    test = load_dataset("EdinburghNLP/xsum", split="test", cache_dir=cache_dir)

    model.eval()

    dataset_size = len(test)

    summary_pair_lists = []
    generation_config = transformers.GenerationConfig(
        do_sample=False,  # True
    )

    with torch.no_grad():
        for i in tqdm(range(0, dataset_size, batch_size), desc="Generating answers"):
            start = i
            end = min(dataset_size, i + batch_size)
            prompts = [XSUM_PROMPT.format(text=test[start + idx]["document"]) for idx in range(end - start)]
            targets = [XSUM_ANSWER.format(text=test[start + idx]["summary"], eos_token=tokenizer.eos_token) for idx in range(end - start)]

            # truncate
            current_device = torch.cuda.current_device()
            input_ids = tokenizer(
                prompts,
                max_length=source_max_len,
                truncation=True,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            ).to(current_device)

            with torch.no_grad():
                generated_ids = model.generate(**input_ids,
                                               generation_config=generation_config,
                                               max_new_tokens=64,
                                               pad_token_id=tokenizer.eos_token_id)

            if isinstance(generated_ids, tuple):
                generated_ids = generated_ids[0]

            decoded_preds = []
            for i in range(len(generated_ids)):
                decoded_preds.append(tokenizer.decode(generated_ids[i][input_ids["input_ids"].size(1):],
                                                      skip_special_tokens=True).strip())

            summary_pair_lists.append((decoded_preds, targets))

    for s_list, t_list in tqdm(summary_pair_lists, desc="Calculating scores"):
        preds, labels = postprocess_text(s_list, t_list)
        metric.add_batch(
            predictions=preds,
            references=labels,
        )

    result = metric.compute(use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}

    return result

# Evaluate perplexity
from datasets import load_dataset
from tqdm import tqdm

import torch
from pissa.data import load_data, format_dataset


def eval_ppl(model, tokenizer, args, dataset="wikitext2"):
    # wikitext2 by default
    if dataset in ["wikitext2", "wikitext"]:
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    else:
        dataset = load_data(dataset, args)
        dataset = format_dataset(tokenizer, dataset, None, args)["train"]
        test = dataset.map(lambda x: {"text": x["input"] + x["output"]})["text"]

    if dataset not in ["wikitext2", "wikitext"]:
        test = test[:1000]  # truncation

    model.eval()
    max_length = model.config.max_position_embeddings
    stride = max_length  #
    nlls = []
    encodings = tokenizer("\n\n".join(test), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl
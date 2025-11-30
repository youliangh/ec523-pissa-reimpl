import copy
import os
import warnings

from typing import Dict, Sequence, List
from dataclasses import dataclass

import datasets
import pandas as pd
import torch
import transformers

from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset, DatasetDict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


# Prompts
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

META_MATH_PROMPT = "Below is an instruction that describes a task. " \
                   "Write a response that appropriately completes the request.\n\n" \
                   "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."

# xSum
SUM_NAME_MAPPING = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

XSUM_PROMPT = "original text: {text}\n\n"
XSUM_ANSWER = "summary: {text}{eos_token}"


# MetaMath
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    args
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def preprocess_with_length_bound_sources_targets(
    sources: List[str],
    targets: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    args):
    tokenized_sources_with_prompt = tokenizer(
        sources,
        max_length=args.source_max_len,
        truncation=True,
        add_special_tokens=False,
    )
    tokenized_targets = tokenizer(
        targets,
        max_length=args.target_max_len,
        truncation=True,
        add_special_tokens=False,
    )

    # Build the input and labels for causal LM
    input_ids = []
    labels = []
    for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
    ):
        input_ids.append(torch.tensor(tokenized_source + tokenized_target))
        labels.append(
            torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
        )
    return dict(input_ids=input_ids, labels=labels)


def pretokenize_dataset_alpaca(examples, tokenizer, args):
    sources = []
    for instruction, inputs in zip(examples["instruction"], examples["input"]):
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"] if (inputs is not None and inputs != "") else ALPACA_PROMPT_DICT["prompt_no_input"]
        sources.append(prompt_format.format_map(dict(instruction=instruction, input=inputs)))
    targets = [f"{output}{tokenizer.eos_token}" for output in examples["output"]]
    data_dict = preprocess(sources, targets, tokenizer, args)
    return data_dict


def pretokenize_dataset_xsum(examples, tokenizer, text_column, summary_column, args):
    # Extract elements
    # max_source_length - 1024, max_target_length - 128
    sources = [f"{tokenizer.bos_token}original text: {example}\n\n" for example in examples[text_column]]
    targets = [f"summary: {example}{tokenizer.eos_token}" for example in examples[summary_column]]
    data_dict = preprocess_with_length_bound_sources_targets(sources, targets, tokenizer, args)
    return data_dict


def pretokenize_dataset(examples, tokenizer, template, query, response, args):
    sources = [template.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer, args)
    return data_dict


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate for supervised fine-tuning with prefetch."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out


def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset


def load_data(dataset_name, args) -> DatasetDict:
    dataset_dict = DatasetDict()
    if dataset_name == 'alpaca-clean':
        split = f"train[:{args.max_train_samples}]" if args.max_train_samples else None
        raw_dataset = load_dataset("json", cache_dir=args.cache_dir, data_files="data/alpaca_data_cleaned.json", split=split)
        if args.max_train_samples:
            dataset_dict['train'] = raw_dataset
        else:
            dataset_dict['train'] = raw_dataset['train']
        return dataset_dict
    elif dataset_name == 'MetaMath':
        split = f"train[:{args.max_train_samples}]" if args.max_train_samples else None
        raw_dataset = load_dataset("json", cache_dir=args.cache_dir, data_files="data/MetaMathQA-395K.json", split=split)
        if args.max_train_samples:
            dataset_dict['train'] = raw_dataset
        else:
            dataset_dict['train'] = raw_dataset['train']
        return dataset_dict
    elif dataset_name == 'xSum':
        split = f"train[:{args.max_train_samples}]" if args.max_train_samples else None
        raw_dataset = load_dataset("EdinburghNLP/xsum", cache_dir=args.cache_dir, split=split)
        if args.max_train_samples:
            dataset_dict['train'] = raw_dataset
            dataset_dict['validation'] = load_dataset("EdinburghNLP/xsum", cache_dir=args.cache_dir, split=datasets.Split.VALIDATION)
            dataset_dict['test'] = load_dataset("EdinburghNLP/xsum", cache_dir=args.cache_dir, split=datasets.Split.TEST)
        else:
            dataset_dict['train'] = raw_dataset['train']
        return dataset_dict
    elif dataset_name == 'math10k':
        split = f"train[:{args.max_train_samples}]" if args.max_train_samples else None
        raw_dataset = load_dataset("json", cache_dir=args.cache_dir, data_files="data/math_10k.json", split=split)
        if args.max_train_samples:
            dataset_dict['train'] = raw_dataset
        else:
            dataset_dict['train'] = raw_dataset['train']
        return dataset_dict
    elif dataset_name in ['commonsense170k', 'commonsense15k']:
        args.dataset_format = 'commonsense'
        filepath = 'data/commonsense_170k.json' if dataset_name == 'commonsense170k' else 'data/commonsense_15k.json'
        split = f"train[:{args.max_train_samples}]" if args.max_train_samples else None
        raw_dataset = load_dataset("json", cache_dir=args.cache_dir, data_files=filepath, split=split)
        if args.max_train_samples:
            dataset_dict['train'] = raw_dataset
        else:
            dataset_dict['train'] = raw_dataset['train']
        return dataset_dict
    else:
        if os.path.exists(dataset_name):
            try:
                args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                full_dataset = local_dataset(dataset_name)
                return full_dataset
            except:
                raise ValueError(f"Error loading dataset from {dataset_name}")
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")


def format_dataset(tokenizer, dataset, dataset_format, args):
    if (
        dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
        (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
    ):
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
    elif dataset_format == 'MetaMath' or (dataset_format is None and args.dataset == 'MetaMath'):
        dataset = dataset.map(lambda x: {
            'input': META_MATH_PROMPT.format_map(dict(instruction=x['query'])),
            'output': f"{x['response']}{tokenizer.eos_token}",
        })
    elif dataset_format == 'xSum' or (dataset_format is None and args.dataset == 'xSum'):
        dataset = dataset.map(lambda x: {
            'input': XSUM_PROMPT.format(text=x['document']),
            'output': XSUM_ANSWER.format(text=x['summary'], eos_token=tokenizer.eos_token),
        })
    elif dataset_format == 'math10k' or (dataset_format is None and args.dataset == 'math10k'):
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        print(dataset)
        return dataset
    elif dataset_format == 'commonsense' or (dataset_format is None and args.dataset in ['commonsense170k', 'commonsense15k']):
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        print(dataset)
        return dataset
    elif dataset_format == 'input-output':
        # leave as is
        pass

    dataset = dataset.remove_columns(
        [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
    )
    return dataset


def make_pretokenize_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca-cleaned, 51942 examples
        - MetaMath, 395,000 examples
    """

    def format_pretokenized_dataset(dataset, dataset_format):
        if (
                dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
                (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(pretokenize_dataset_alpaca,
                                  batched=True,
                                  batch_size=2048,
                                  load_from_cache_file=True,
                                  remove_columns=dataset.column_names,
                                  fn_kwargs={"tokenizer": tokenizer, "args": args})
            # raise NotImplementedError("Coming soon")
            # dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif args.dataset == 'MetaMath':
            dataset = dataset.map(pretokenize_dataset,
                                  batched=True,
                                  batch_size=2048,
                                  load_from_cache_file=True,
                                  remove_columns=dataset.column_names,
                                  fn_kwargs={"tokenizer": tokenizer, "template": META_MATH_PROMPT, "query": "query", "response": "response", "args": args})
        elif args.dataset == 'xSum':
            dataset = dataset.map(pretokenize_dataset_xsum,
                                  batched=True,
                                  batch_size=2048,
                                  load_from_cache_file=True,
                                  remove_columns=dataset.column_names,
                                  fn_kwargs={"tokenizer": tokenizer, "text_column": "document", "summary_column": "summary", "args": args})
        elif args.dataset in ['math10k', 'commonsense170k', 'commonsense15k']:
            dataset = dataset.map(pretokenize_dataset_alpaca,
                                  batched=True,
                                  batch_size=2048,
                                  load_from_cache_file=True,
                                  remove_columns=dataset.column_names,
                                  fn_kwargs={"tokenizer": tokenizer, "args": args})
        elif dataset_format == 'input-output':
            # leave as is
            pass
        return dataset

    # Load dataset.
    dataset = load_data(args.dataset, args)

    # Split train/eval, reduce size
    if args.do_eval:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        elif 'validation' in dataset:
            eval_dataset = dataset['validation']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset['train'].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']

        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    if args.do_train:
        training_dataset = dataset['train']

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=format_pretokenized_dataset(training_dataset, args.dataset_format) if args.do_train else None,
        eval_dataset=format_pretokenized_dataset(eval_dataset, args.dataset_format) if args.do_eval else None,
        data_collator=data_collator
    )


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:

    """
     # Load dataset.
    dataset = load_data(args.dataset, args)
    dataset = format_dataset(tokenizer, dataset, args.dataset_format, args)

    # Split train/eval, reduce size
    if args.do_eval:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        elif 'validation' in dataset:
            eval_dataset = dataset['validation']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=data_collator
    )


def prepare_dummy_data(tokenizer: transformers.PreTrainedTokenizer, args):
    dataset = load_data(args.dataset, args)
    dataset = format_dataset(tokenizer, dataset, args.dataset_format, args)

    training_dataset = dataset["train"]
    training_dataset = training_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )

    return training_dataset, data_collator

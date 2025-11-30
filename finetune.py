# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import time
import pickle
import warnings
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict
import logging
import tqdm

import numpy as np
import torch
import transformers
import argparse
import evaluate
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Seq2SeqTrainer,
    LlamaTokenizer,
    PreTrainedTokenizerBase,
    pipeline
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

def init_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)

from pissa.benchmark.commonsense_reasoning import eval_commonsense
from pissa.benchmark.math_reasoning import eval_math
from pissa.benchmark.xsum import eval_xsum

from pissa.data import DEFAULT_PAD_TOKEN, make_pretokenize_data_module, make_data_module
from pissa.utility import init_seed, easy_dump
from pissa.core import prepare_model


world_size = None
local_rank = None

def gen_random_prompts(
    tokenizer: PreTrainedTokenizerBase,
    lens_mean: int,
    lens_range: int,
    num_prompts: int,
):
    low = lens_mean - (lens_range // 2)
    high = lens_mean + (lens_range // 2)
    max_vocab_ids = max(tokenizer.get_vocab().values())

    def gen_prompt_tokens(length):
        return [random.randint(10, max_vocab_ids) for _ in range(length)]

    prompt_lens = list(map(lambda _: random.randint(low, high), range(num_prompts)))
    prompts_as_tokens = list(map(gen_prompt_tokens, prompt_lens))
    prompts = list(map(tokenizer.decode, prompts_as_tokens))

    # Because token does not map 1:1 to words, sometimes we get more or less tokens than desired.
    new_prompts = []
    encoded_prompts = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    for encoded, pmp_len in zip(encoded_prompts, prompt_lens):
        if len(encoded) > pmp_len:
            # This removes the additional tokens by tokenizing the prompt and cutting off additional tokens.
            encoded = encoded[:pmp_len]
        elif len(encoded) < pmp_len:
            # This left-pads the prompt with padding tokens.
            encoded = [tokenizer.pad_token_id] * (pmp_len - len(encoded)) + encoded
        decoded = tokenizer.decode(encoded)
        # encoded = tokenizer(decoded, add_special_tokens=False)["input_ids"]
        # assert (
        #     len(encoded) == pmp_len
        # ), f"Expected prompt to contain exactly {pmp_len} tokens, got {len(encoded)=}"
        new_prompts.append(decoded)

    return new_prompts, prompt_lens


def report0(*args):
    print(*args)

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'norm' in name:
            module.to(torch.bfloat16)
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )


@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=10000,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=128,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default="MetaMath",
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=False,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={
        "help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={
        "help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=-1, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={
        "help": 'The L2 weight decay rate of AdamW'})  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False,
                                        metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={
        "help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True,
                                         metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    do_eval: bool = field(default=True, metadata={"help": 'To eval'})
    lr_scheduler_type: str = field(default='constant', metadata={
        "help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10,
                               metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={
        "help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40,
                                  metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    enable_pissa: bool = field(default=False, metadata={"help": "Enable Pissa"})
    pretokenize: bool = field(default=False, metadata={"help": "Pretokenize the dataset"})
    eval_test: Optional[bool] = field(
        default=True,
        metadata={"help": "Evaluate on test set"}
    )


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        report0('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def get_accelerate_model(args, checkpoint_dir):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()

    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    if args.full_finetune:
        assert args.bits == 16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=None,
        dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code
    ).eval()

    model.config.torch_dtype = torch.bfloat16

    lora_target_modules = []
    if "llama" in args.model_name_or_path.lower():
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # edit here!
    elif "opt" in args.model_name_or_path.lower():
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]  # edit here!
    else:
        raise NotImplementedError("Model not supported yet.")
    lora_weights = prepare_model(model, target_modules=lora_target_modules, lora_rank=args.lora_r, bits=args.bits)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,  # Fast tokenizer giving issues.
        trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if 'llama' in args.model_name_or_path.lower() or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        report0('Adding special tokens.')
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id is not None and model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
        })

    if not args.full_finetune:
        if args.bits <= 8:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        else:
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()

        if checkpoint_dir is not None:
            report0("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            report0(f'adding LoRA modules...')
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=args.lora_dropout,
                init_lora_weights=True,  # init later with u and v
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            weight_t = torch.bfloat16
            module = module.to(weight_t)

            # Pissa
            #                  model.layers.0.mlp.gate_proj
            # base_model.model.model.layers.0.mlp.gate_proj
            # default - adapter name
            target = ".".join(name.split(".")[-5:])
            adapter_name = "default"
            current_device = torch.get_device(module.lora_A[adapter_name].weight)
            if target in lora_weights:
                u, s, v = lora_weights.pop(target)
                u = u @ s
                v = s @ v
                module.lora_A[adapter_name].weight = torch.nn.Parameter(v.to(weight_t).to(current_device))
                module.lora_B[adapter_name].weight = torch.nn.Parameter(u.to(weight_t).to(current_device))

        if 'norm' in name:
            module = module.to(torch.bfloat16)
        if 'lm_head' in name or 'embed_tokens' in name or 'embed_positions' in name:
            if hasattr(module, 'weight'):
                module = module.to(torch.bfloat16)

    trainable_ready, frozen_ready = get_trainable_parameters(args, model)
    return model, tokenizer


def get_trainable_parameters(args, model, do_print=True):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    frozen_params = 0
    for _, param in model.named_parameters():
        
        factor = 1
        if param.dtype == torch.uint8 and args.bits == 4:
            factor = 2

        if param.requires_grad:
            trainable_params += param.numel() * factor
        else:
            frozen_params += param.numel() * factor

    all_param = trainable_params + frozen_params

    if do_print:
        report0(
            f"trainable params: {trainable_params} || "
            f"frozen params: {frozen_params} "
            f"all params: {all_param} || "
            f"trainable: {100 * trainable_params / all_param:.4f}% || "
            f"frozen: {100 * frozen_params / all_param:.4f}%"
        )

    return trainable_params, frozen_params


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        report0(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def train():
    global world_size, local_rank

    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    print(args)

    # ensure reproductivity
    init_seed(args.seed)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        report0('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False

    if args.pretokenize:
        data_module = make_pretokenize_data_module(tokenizer=tokenizer, args=args)
    else:
        data_module = make_data_module(tokenizer=tokenizer, args=args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items()},
    )

    # # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes and parameter counts before training.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        report0(k, v, v / total)

    if args.do_eval:
        class MathEvalCallback(transformers.TrainerCallback):
            math_eval_cnt = 0
            math_eval_results = {}

            def on_evaluate(self, args, state, control, model, **kwargs):
                self.math_eval_cnt += 1

                results = {}

                trainer.model.eval()
                for dataset in ["AddSub", "AQuA", "MAWPS", "SVAMP", "gsm8k"]:
                    results[f"{dataset}_acc"] = eval_math(model, tokenizer, dataset, batch_size=2 if dataset=="AQuA" else 4)
                trainer.log(results)

                self.math_eval_results[self.math_eval_cnt * args.eval_steps] = results
                easy_dump(self.math_eval_results, args.output_dir, "math_eval_results")

        class CommonsenseEvalCallback(transformers.TrainerCallback):
            cs_eval_cnt = 0
            cs_eval_results = {}

            def on_evaluate(self, args, state, control, model, **kwargs):
                self.cs_eval_cnt += 1

                results = {}

                trainer.model.eval()
                for dataset in ["boolq", "siqa", "arc_c", "arc_e", "winogrande"]:
                    results[f"{dataset}_acc"] = eval_math(model, tokenizer, dataset, batch_size=4)
                trainer.log(results)

                self.cs_eval_results[self.cs_eval_cnt * args.eval_steps] = results
                easy_dump(self.cs_eval_results, args.output_dir, "commonsense_eval_results")

        class XSumEvalCallback(transformers.TrainerCallback):
            xsum_eval_cnt = 0
            xsum_eval_results = {}
            cache_dir = args.cache_dir
            rouge_metric = evaluate.load("rouge")  # use local
            source_max_len = args.source_max_len

            def on_evaluate(self, args, state, control, **kwargs):
                self.xsum_eval_cnt += 1

                trainer.model.eval()

                # speeding the inference
                torch.use_deterministic_algorithms(False)
                model.config.use_cache = True

                results = eval_xsum(model, tokenizer, source_max_len=self.source_max_len, cache_dir=self.cache_dir, batch_size=8)

                torch.use_deterministic_algorithms(True)
                model.config.use_cache = False

                trainer.log(results)

                self.xsum_eval_results[self.xsum_eval_cnt * args.eval_steps] = results
                easy_dump(self.xsum_eval_results, args.output_dir, "xSum_eval_results")

        if args.dataset in ["MetaMath", "math10k"]:
            trainer.add_callback(MathEvalCallback)
        elif args.dataset == "commonsense":
            trainer.add_callback(CommonsenseEvalCallback)
        elif args.dataset == "xSum":
            trainer.add_callback(XSumEvalCallback)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        init_seed(args.seed)
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    torch.use_deterministic_algorithms(False)
    model.config.use_cache = True

    # if args.full_finetune:
    #     model.save_pretrained(os.path.join(args.output_dir, "saved_model"))

    model.eval()

    warnings.filterwarnings("ignore")  # suppress the user warning of input length
    with torch.no_grad():
        if args.dataset in ["MetaMath", "math10k"]:
            generation_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
            dummy_prompt = "Question: what's the answer of 4 + 8 / 2?\n\nAnswer:"
            print(generation_pipeline(dummy_prompt, max_length=256, pad_token_id=tokenizer.eos_token_id))

    # Evaluation
    logger.info("*** Evaluate ***")
    results = {}
    trainer.model.eval()

    if args.eval_test:
        if args.dataset in ["MetaMath", "math10k"]:
            eval_batch_size = 8 if "70" in args.model_name_or_path else 16  # avoid oom
            eval_batch_size = 32 if "phi" in args.model_name_or_path else eval_batch_size
            for dataset in ["AddSub", "SingleEq", "MAWPS", "SVAMP", "gsm8k"]:
                results[f"{dataset}_acc"] = eval_math(model, tokenizer, dataset, batch_size=eval_batch_size)
            trainer.log(results)
            easy_dump(results, args.output_dir, "math_eval_results")
        elif args.dataset in ["commonsense15k", "commonsense170k"]:
            for dataset in ["boolq", "siqa", "arc_c", "arc_e", "winogrande"]:
                results[f"{dataset}_acc"] = eval_commonsense(model, tokenizer, dataset, batch_size=8)
            trainer.log(results)
            easy_dump(results, args.output_dir, "commonsense_eval_results")
        elif args.dataset == "xSum":
            results["xsum"] = eval_xsum(model, tokenizer, source_max_len=args.source_max_len, cache_dir=args.cache_dir, batch_size=8)
            trainer.log(results)
            easy_dump(results, args.output_dir, "xsum_eval_results")

    if (args.do_train or args.do_eval):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()

# ec523-pissa-reimpl
This is project repo for EC523 Deep Learning, containing our reimplementation of PiSSA, a parameter-efficient fine-tuning method that uses singular values and singular vectors to improve LoRA.
## Overview  
<img width="835" height="366" alt="overview" src="https://github.com/user-attachments/assets/a3e3524b-e091-4495-ab75-f42a4f61bf06" />  

We follow the main ideas from the PiSSA paper and implement:  
	•	SVD-based model decomposition  
	•	PiSSA-style LoRA initialization  
	•	4 / 8 bit quantization with BitsAndBytes  
	•	A basic fine-tuning pipeline for LLaMA-style models  
	•	Evaluation on math reasoning benchmarks  
The goal is to reproduce part of the results from the original paper while writing most of the core logic ourselves.  
## Goal  
### Completed:
  Finished SVD decomposition  
  Implemented quantization (4 / 8 / 16 bit)  
  Implemented PiSSA LoRA module  
### How-to-use:
**Step 1:** Download dataset `https://drive.google.com/file/d/1h1oDTeQQvzDt8zaSgP0DinEkXnXqmRBi/view?usp=drive_link` and unzip it in the project's root.

**Step 2:** Download huggingface models.
```
hf download meta-llama/Llama-2-7b-hf --local-dir "local path"
hf download hf download google/gemma-7b --local-dir "local path"
```

**Step 3:** Modify the script and launch fine-tuning.
```
# run_finetune.sh

# first 100k data points in MetaMath
dataset="MetaMath"

# lora_r should be always equal to lora_alpha per the paper
--lora_r 128
--lora_alpha 128

# 2e-5 per the paper
--learning_rate 0.0002

# batch size should be 128 per the paper
# NOTE: batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
CUDA_VISIBLE_DEVICES="0"
--per_device_train_batch_size 64
--gradient_accumulation_steps 2

# 0.03 per the paper
--warmup_ratio 0.03

# decrease if OOM during evaluation
--per_device_eval_batch_size 32

# 100k per the paper
--max_train_samples 100000

# default 16, 4 for quantization
--bits [16|4]
```

  Fine-tuning and evaluation in progress
## Reference
1) F. Meng, Z. Wang, and M. Zhang. PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. Advances in Neural Information Processing Systems 37 (2024): 121038-121072.
2) J. He, C. Zhou, X. Ma, T. Berg-Kirkpatrick, G. Neubig. Towards a Unified View of Parameter-Efficient Transfer Learning. ICLR, 2022.
3) E. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. LoRA: Low-rank Adaptation of Large Language Models. ICLR, 2022.

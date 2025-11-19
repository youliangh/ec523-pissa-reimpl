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
### Todo: 
  Fine-tuning and evaluation in progress
## Reference
1) F. Meng, Z. Wang, and M. Zhang. PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. Advances in Neural Information Processing Systems 37 (2024): 121038-121072.
2) J. He, C. Zhou, X. Ma, T. Berg-Kirkpatrick, G. Neubig. Towards a Unified View of Parameter-Efficient Transfer Learning. ICLR, 2022.
3) E. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. LoRA: Low-rank Adaptation of Large Language Models. ICLR, 2022.

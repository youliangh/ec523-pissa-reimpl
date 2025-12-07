import os
import functools
import torch
import bitsandbytes as bnb

from tqdm import tqdm
from typing import List

from pissa.utility import update_model_with_module

def quantize_module(model, name: str, module: torch.nn.Module, bits: int):
    if bits == 16:
        return  # No quantization needed for 16-bit

    has_bias = hasattr(module, "bias") and module.bias is not None
    if bits == 4:
        quantized_layer = bnb.nn.Linear4bit(module.in_features,
                                            module.out_features,
                                            bias=has_bias,
                                            compute_dtype=torch.bfloat16,
                                            quant_type="nf4")
    elif bits == 8:
        quantized_layer = bnb.nn.Linear8bitLt(module.in_features,
                                              module.out_features,
                                              bias=has_bias,
                                              has_fp16_weights=False,
                                              threshold=6.0)

    quantized_layer.load_state_dict(module.state_dict())
    
    update_model_with_module(model, name, quantized_layer.to(module.weight.device))


def prepare_model(model, target_modules: List[str],
                  lora_rank: int=32,
                  bits: int=4,
                  mode: str="pissa"):
        loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
        if loaded_in_kbit:
            raise NotImplementedError("Not supported for loading in 4-bit or 8-bit yet.")
        
        if bits not in [4, 8, 16]:
            raise ValueError("bits must be one of [4, 8, 16]")

        # Hack to mark the model as loaded in k-bit
        if bits <= 8:
            setattr(model, f"is_loaded_in_{bits}bit", True)
        
        if lora_rank <= 0:
            raise ValueError("lora_rank must be greater than 0")

        available_modules = []

        for name, module in model.named_modules():
            suffix = name.split(".")[-1]    
            if suffix not in target_modules:
                continue

            available_modules.append((name, module))

        progress_bar = tqdm(available_modules, desc="Decomposition of weights", unit="module")
        lora_weights = {}

        for name, module in progress_bar:
            if mode == "pissa":
                original_weight_type = module.weight.data.dtype
                original_weight = module.weight.data.to(torch.float32)
                u, s, v = torch.linalg.svd(original_weight, full_matrices=False)
                new_weight = u[:, lora_rank:] @ torch.diag(s[lora_rank:]) @ v[lora_rank:, :]
                module.weight.data = new_weight.contiguous().to(original_weight_type)

                quantize_module(model, name, module, bits)

                lora_weights[name] = ((
                    u[:, :lora_rank].to(original_weight_type),
                    torch.diag(torch.sqrt(s[:lora_rank])).to(original_weight_type),
                    v[:lora_rank, :].to(original_weight_type)
                ))
            elif mode == "lora":
                quantize_module(model, name, module, bits)

        return lora_weights
        
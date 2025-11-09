import os
import pickle
import random
import numpy as np
import gc
import torch
import json

from tqdm import tqdm

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

def get_module(model, name: str):
    parent_name = ".".join(name.split(".")[:-1])
    for n, p in model.named_modules():
        if n == parent_name:
            return p
    raise LookupError(name)

def get_exact_module(model, name: str):
    for n, p in model.named_modules():
        if n == name:
            return p
    raise LookupError(name)

def update_model_with_module(model, name: str, module: torch.nn.Module):
    with torch.no_grad():
        parent_module = get_module(model, name)
        submodule = name.split(".")[-1]
        delattr(parent_module, submodule)
        gc.collect()
        torch.cuda.empty_cache()
        setattr(parent_module, submodule, module)

def easy_dump(obj, dest, label):
    with open(os.path.join(dest, f"{label}.pkl"), "wb") as f:
        pickle.dump(obj, f)

    # also dump as json if it is a dict
    if isinstance(obj, dict):
        with open(os.path.join(dest, f"{label}.json"), "w") as f:
            f.write(json.dumps(obj, indent=4))
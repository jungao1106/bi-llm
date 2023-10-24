import os
import json
import torch
import numpy as np

def GlobalSeed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def load_dataset_from_json(filePath) -> None:
    return json.load(open(filePath, 'r', encoding='utf8'))

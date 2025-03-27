import yaml
from typing import Dict, Any
import torch
import random
import numpy as np 

def load_yml(path: str) -> Dict[str, Any]:
    with open(path) as f:
       config = yaml.safe_load(f) 
    return config
   

def seed_everything(random_state: int) -> None:
   random.seed(random_state)
   np.random.seed(random_state)
   torch.manual_seed(random_state)
   torch.cuda.manual_seed(random_state)
   torch.backends.cudnn.deterministic = False
   torch.backends.cudnn.benchmark = True



def get_torch_device() -> torch.device:
   if torch.cuda.is_available():
      return torch.device("cuda:0")
   elif torch.mps.is_available():
      return torch.device("mps")
   
   return torch.device("cpu")
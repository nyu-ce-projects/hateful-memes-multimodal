import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn import BatchNorm,GraphNorm

def get_device():
    n_gpus = 0
    if torch.cuda.is_available():
        device = 'cuda' 
        n_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device,n_gpus

def get_normalization(norm_type) -> Module:
    if norm_type=="graph_norm":
        return GraphNorm
    elif norm_type=="batch_norm":
        return BatchNorm
    else:
        raise NameError("invalid norm_type name")
    
def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")
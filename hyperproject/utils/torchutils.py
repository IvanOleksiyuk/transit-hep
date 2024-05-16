import torch

def tensor2numpy(x):
    return x.detach().cpu().numpy()
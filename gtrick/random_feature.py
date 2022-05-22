import torch

def random_feature(x):
    r = torch.rand(size=(len(x), 1)).to(x.device)
    x = torch.cat([x, r], dim=-1)
    return x
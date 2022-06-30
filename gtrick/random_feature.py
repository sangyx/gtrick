import torch

def random_feature(x):
    r"""Random feature comes from [Random Features Strengthen Graph Neural Networks](http://arxiv.org/abs/2002.03155). This trick adds a random feature to node features.

    Example:
        [RandomFeature (DGL)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/RandomFeature.ipynb), [RandomFeature (PyG)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/RandomFeature.ipynb) 

    Args:
        x (torch.Tensor): The node feature.
    
    Returns:
        (torch.Tensor): The node feature with a random feature.
    """

    r = torch.rand(size=(len(x), 1)).to(x.device)
    x = torch.cat([x, r], dim=-1)
    return x
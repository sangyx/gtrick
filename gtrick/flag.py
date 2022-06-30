"""Module for Free Large-scale Adversarial Augmentation on Graphs."""

import torch
import math

"""
The code are adapted from
https://github.com/devnkong/FLAG
"""

class FLAG:
    r"""FLAG is an adversarial data augmentation method for Graph Neural Networks, which comes from
    [Robust Optimization as Data Augmentation for Large-scale Graphs](https://arxiv.org/abs/2010.09891).

    This trick is helpful for **Node Level Task** and **Graph Level Task**.

    Example:
        [FLAG (DGL)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/dgl/FLAG.ipynb), [FALG (PyG)](https://nbviewer.org/github/sangyx/gtrick/blob/main/benchmark/pyg/FLAG.ipynb)

    Args:
        emb_dim (int): Node feature dim.
        loss_func (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer) : Optimizer.
        m (int): Ascent steps. Train the same minibatch m times.
        step_size (float): Ascent step size. If mag <= 0, perturb is initialized from uniform distribution [-step_size, step_size].
        mag (float): If mag > 0, it controls the max norm of perturb.

    """

    def __init__(self, emb_dim, loss_func, optimizer, m=3, step_size=1e-3, mag=-1) -> None:

        self.emb_dim = emb_dim
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.m = m
        self.step_size = step_size
        self.mag = mag

    def __call__(self, model, forward, num_nodes, y):
        r"""
        Args:
            model (torch.nn.Module): The model.
            forward (Callable[[torch.Tensor], torch.Tensor]): The function that inputs perturb and gets output.
            num_nodes (int): The number of nodes.
            y (torch.Tensor): The ground truth label.
        
        Returns:
            (torch.Tensor): The loss.
            (torch.Tensor): The output of the model.
        """

        model.train()
        self.optimizer.zero_grad()

        perturb_shape = (num_nodes, self.emb_dim)
        device = y.device

        if self.mag > 0:
            perturb = torch.FloatTensor(*perturb_shape).uniform_(-1, 1).to(device)
            perturb = perturb * self.mag / math.sqrt(perturb_shape[-1])
        else:
            perturb = torch.FloatTensor(
                *perturb_shape).uniform_(-self.step_size, self.step_size).to(device)
        perturb.requires_grad_()
        out = forward(perturb)

        loss = self.loss_func(out, y)
        loss /= self.m

        for _ in range(self.m - 1):
            loss.backward()
            perturb_data = perturb.detach() + self.step_size * torch.sign(perturb.grad.detach())
            if self.mag > 0:
                perturb_data_norm = torch.norm(perturb_data, dim=-1).detach()
                exceed_mask = (perturb_data_norm > self.mag).to(perturb_data)
                reweights = (self.mag / perturb_data_norm * exceed_mask +
                            (1 - exceed_mask)).unsqueeze(-1)
                perturb_data = (perturb_data * reweights).detach()

            perturb.data = perturb_data.data
            perturb.grad[:] = 0

            out = forward(perturb)
            loss = self.loss_func(out, y)
            loss /= self.m

        loss.backward()
        self.optimizer.step()

        return loss, out

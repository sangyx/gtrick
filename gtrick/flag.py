import torch
import math

class FLAG:
    def __init__(self, emb_dim, loss_func, optimizer, m=3, step_size=1e-3, mag=1e-3) -> None:
        self.emb_dim = emb_dim
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.m = m
        self.step_size = step_size
        self.mag = mag

    def __call__(self, model, forward, num_nodes, y, mask=None):
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
        if mask is not None:
            out = out[mask]

        loss = self.loss_func(out, y)
        loss /= self.m

        for _ in range(self.m - 1):
            loss.backward()
            perturb_data = perturb.detach() + self.step_size * torch.sign(perturb.grad.detach())
            if self.mag > 0:
                perturb_data_norm = torch.norm(perturb_data, dim=-1).detach()
                exceed_mask = (perturb_data_norm > self.mag).to(perturb_data)
                reweights = (self.mag / perturb_data_norm * exceed_mask +
                            (1-exceed_mask)).unsqueeze(-1)
                perturb_data = (perturb_data * reweights).detach()

            perturb.data = perturb_data.data
            perturb.grad[:] = 0

            out = forward(perturb)
            if mask is not None:
                out = out[mask]
            loss = self.loss_func(out, y)
            loss /= self.m

        loss.backward()
        self.optimizer.step()

        return loss, out


# def flag(model, forward, perturb_shape, y, optimizer, device, loss_func, m=3, step_size=1e-3, mag=1e-3, mask=None):
#     model.train()
#     optimizer.zero_grad()

#     if mag > 0:
#         perturb = torch.FloatTensor(*perturb_shape).uniform_(-1, 1).to(device)
#         perturb = perturb * mag / math.sqrt(perturb_shape[-1])
#     else:
#         perturb = torch.FloatTensor(
#             *perturb_shape).uniform_(-step_size, step_size).to(device)
#     perturb.requires_grad_()
#     out = forward(perturb).view(-1)
#     if mask is not None:
#         out = out[mask]

#     loss = loss_func(out, y)
#     loss /= m

#     for _ in range(m - 1):
#         loss.backward()
#         perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
#         if mag > 0:
#             perturb_data_norm = torch.norm(perturb_data, dim=-1).detach()
#             exceed_mask = (perturb_data_norm > mag).to(perturb_data)
#             reweights = (mag / perturb_data_norm * exceed_mask +
#                          (1 - exceed_mask)).unsqueeze(-1)
#             perturb_data = (perturb_data * reweights).detach()

#         perturb.data = perturb_data.data
#         perturb.grad[:] = 0

#         out = forward(perturb).view(-1)
#         if mask is not None:
#             out = out[mask]
#         loss = loss_func(out, y)
#         loss /= m

#     loss.backward()
#     optimizer.step()

#     return loss, out

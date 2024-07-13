import torch
from torch.optim import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def step(self, closure=None, itr=0):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if itr != 0:
            dp_list = []

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if itr != 0:
                    dp_list.append(d_p)

                p.data.add_(d_p, alpha=-lr)

        if itr != 0:
            return dp_list
        else:
            return loss

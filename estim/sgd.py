import torch
import torch.nn
import torch.multiprocessing

from .gestim import GradientEstimator


class SGDEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(SGDEstimator, self).__init__(*args, **kwargs)
        # many open files? torch.multiprocessing sharing file_system
        self.init_data_iter()

    def grad(self, model, in_place=False, data=None):
        if data is None:
            data = next(self.data_iter)

        loss = model.criterion(model, data)

        if in_place:
            loss.backward()
            return loss
        g = torch.autograd.grad(loss, model.parameters())
        return g

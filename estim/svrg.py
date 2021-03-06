import logging

import torch
import torch.nn
import torch.multiprocessing

from log_utils import Profiler
from .gestim import GradientEstimator
import models


class SVRGEstimator(GradientEstimator):
    def __init__(self, *args, **kwargs):
        super(SVRGEstimator, self).__init__(*args, **kwargs)
        self.init_data_iter()
        self.mu = []
        self.model = None

    def snap_batch(self, model):
        # model.eval()  # SVRG's trouble with dropout/batchnorm/data aug
        # self.model = model = copy.deepcopy(model)
        # deepcopy does not work with kfac, hooks are also copied
        if self.model is None:
            self.model = models.init_model(self.opt)
        self.model.load_state_dict(model.state_dict())
        model = self.model
        model.eval()
        self.mu = [torch.zeros_like(g) for g in model.parameters()]
        num = 0
        batch_time = Profiler()
        for batch_idx, data in enumerate(self.data_loader):
            idx = data[2]
            num += len(idx)
            if (self.opt.svrg_bsnap_num > 0
                    and num >= self.opt.svrg_bsnap_num):
                break
            loss = model.criterion(model, data, reduction='sum')
            grad_params = torch.autograd.grad(loss, model.parameters())
            for m, g in zip(self.mu, grad_params):
                m += g
            batch_time.toc('Time')
            batch_time.end()
            if batch_idx % 10 == 0:
                logging.info(
                        'SVRG Snap> [{0}/{1}]: {bt}'.format(
                            batch_idx, len(self.data_loader),
                            bt=str(batch_time)))
        # TODO: do it when summing?
        for m in self.mu:
            m /= num

    def grad(self, model_new, in_place=False):
        data = next(self.data_iter)

        model_old = self.model
        if model_old is None:
            import warnings
            warnings.warn('SVRG: using new model.')
            model_old = model_new

        # old grad
        loss = model_old.criterion(model_old, data)
        g_old = torch.autograd.grad(loss, model_old.parameters())

        # new grad
        loss = model_new.criterion(model_new, data)
        g_new = torch.autograd.grad(loss, model_new.parameters())

        if in_place:
            for m, go, gn, p in zip(
                    self.mu, g_old, g_new, model_new.parameters()):
                p.grad.copy_(m-go+gn)
            return loss
        ge = [m-go+gn for m, go, gn in zip(self.mu, g_old, g_new)]
        return ge

    def state_dict(self):
        return {'mu': [m.cpu() for m in self.mu]}

    def load_state_dict(self, state):
        if 'mu' not in state:
            return
        mu = state['mu']
        for mx, my in zip(mu, self.mu):
            mx.copy_(my)

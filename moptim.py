import logging
import utils

from data import get_gestim_loader
from log_utils import LogCollector
from estim.mgestim import GEstimatorMulti


class OptimizerMulti(object):

    def __init__(self, model, train_loader, tb_logger, opt):
        self.model = model
        self.opt = opt
        self.niters = 0
        self.optimizer = None
        self.logger = LogCollector(opt)
        gestim_loader = get_gestim_loader(train_loader, opt)
        self.gest = GEstimatorMulti(model, gestim_loader, opt, tb_logger)
        self.tb_logger = tb_logger
        self.init_snapshot = False
        self.gest_used = False
        self.gest_counter = 0
        self.last_log_iter = 0

    @property
    def param_groups(self):
        return self.gest.get_optim().param_groups

    def inc_niters(self):
        self.niters += 1
        self.gest.update_niters(self.niters)
        return self.niters

    def is_log_iter(self):
        niters = self.niters
        opt = self.opt
        if (niters-self.last_log_iter >= opt.gvar_log_iter
                and niters >= opt.gvar_start):
            self.last_log_iter = niters
            return True
        return False

    def log_var(self, model):
        tb_logger = self.tb_logger
        gviter = self.opt.gvar_estim_iter
        return self.gest.log_var(model, gviter, tb_logger)

    def snap_batch(self, model):
        # model.eval()  # done inside SVRG
        model.train()
        self.gest.snap_batch(model)
        self.init_snapshot = True
        self.gest_counter = 0

    def snap_online(self, model):
        # model.eval()  # TODO: keep train
        model.train()
        self.gest.snap_online(model)

    def use_sgd(self, niters):
        use_sgd = not self.opt.g_optim or niters < self.opt.g_optim_start
        if self.gest_used != (not use_sgd):
            self.gest.get_optim().niters = niters
            utils.adjust_lr(self.gest.get_optim(), self.opt)
        return use_sgd

    def grad(self):
        model = self.model
        model.train()
        use_sgd = self.use_sgd(self.niters)
        self.gest_used = not use_sgd
        if self.gest_used:
            self.gest_counter += 1
        return self.gest.grad(use_sgd, model, in_place=True)

    def step(self, profiler):
        opt = self.opt
        model = self.model

        # Rare snaps
        if ((self.niters - opt.gvar_start) % opt.g_bsnap_iter == 0
                and self.niters >= opt.gvar_start):
            logging.info('Batch Snapshot')
            self.snap_batch(model)
            profiler.toc('snap_batch')
        # Frequent snaps
        if ((self.niters - opt.gvar_start) % opt.g_osnap_iter == 0
                and self.niters >= opt.gvar_start):
            self.snap_online(model)
            profiler.toc('snap_online')

        self.zero_grad()

        pg_used = self.gest_used
        loss = self.grad()
        if self.gest_used != pg_used:
            logging.info('Optimizer changed.')
        self._step()
        profiler.toc('optim')
        return loss

    def state_dict(self):
        return self.gest.state_dict()

    def load_state_dict(self, state):
        self.gest.load_state_dict(state)
        self.init_snapshot = True

    def zero_grad(self):
        return self.gest.get_optim().zero_grad()

    def _step(self):
        return self.gest.get_optim().step()

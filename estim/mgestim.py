import torch
import torch.nn
import torch.multiprocessing

from estim.sgd import SGDEstimator
from estim.svrg import SVRGEstimator


def init_estimator(g_estim, opm, data_loader, opt, tb_logger):
    if g_estim == 'sgd':
        gest = SGDEstimator(data_loader, opt, tb_logger)
    elif g_estim == 'svrg':
        gest = SVRGEstimator(data_loader, opt, tb_logger)
    return gest


def init_optim(optim_name, model, opt):
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=opt.lr, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay,
                                    nesterov=opt.nesterov)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.lr,
                                     betas=opt.adam_betas,
                                     eps=opt.adam_eps,
                                     weight_decay=opt.weight_decay)
    return optimizer


class GEstimatorMulti(object):
    def __init__(self, model, data_loader, opt, tb_logger):
        self.gest_used = False
        self.optim = []
        self.gest = []
        self.opt = opt
        ges = opt.g_estim.split(',')
        for optim_name in opt.optim.split(','):
            opm = init_optim(optim_name, model, opt)
            opm.secondary_optim = len(self.optim) != 0
            self.optim += [(optim_name, opm)]
        for eid, g_estim in enumerate(ges):
            opm = (self.optim[0][1]
                   if len(self.optim) == 1
                   else self.optim[eid][1])
            self.gest += [(g_estim, init_estimator(
                g_estim, opm, data_loader, opt, tb_logger))]
        self.niters = 0

    def update_niters(self, niters):
        self.niters = niters
        for name, estim in self.gest:
            estim.update_niters(self.niters)
        for name, opm in self.optim:
            opm.steps = niters

    def snap_batch(self, model):
        for name, gest in self.gest:
            gest.snap_batch(model)

    def snap_online(self, model):
        for name, gest in self.gest:
            gest.snap_online(model)

    def snap_model(self, model):
        for name, gest in self.gest:
            gest.snap_model(model)

    def log_var(self, model, gviter, tb_logger):
        niters = self.niters
        Ege_s = []
        bias_str = ''
        var_str = ''
        snr_str = ''
        nvar_str = ''
        estim_str = '%s' % self.gest[self.gest_used][0]
        keys = []
        vals = []
        for i, (name, gest) in enumerate(self.gest):
            Ege, var_e, snr_e, nv_e = gest.get_Ege_var(model, gviter)
            if i == 0:
                tb_logger.log_value('sgd_var', float(var_e), step=niters)
                tb_logger.log_value('sgd_snr', float(snr_e), step=niters)
                tb_logger.log_value('sgd_nvar', float(nv_e), step=niters)
            if i == 1:
                tb_logger.log_value('est_var', float(var_e), step=niters)
                tb_logger.log_value('est_snr', float(snr_e), step=niters)
                tb_logger.log_value('est_nvar', float(nv_e), step=niters)
            tb_logger.log_value('est_var%d' % i, float(var_e), step=niters)
            tb_logger.log_value('est_snr%d' % i, float(snr_e), step=niters)
            tb_logger.log_value('est_nvar%d' % i, float(nv_e), step=niters)
            Ege_s += [Ege]
            var_str += '%s: %.8f ' % (name, var_e)
            snr_str += '%s: %.8f ' % (name, snr_e)
            nvar_str += '%s: %.8f ' % (name, nv_e)
            if i > 0:
                bias = torch.mean(torch.cat(
                    [(ee-gg).abs().flatten()
                     for ee, gg in zip(Ege_s[0], Ege_s[i])]))
                if i == 1:
                    tb_logger.log_value('grad_bias', float(bias), step=niters)
                tb_logger.log_value('grad_bias%d' % i,
                                    float(bias), step=niters)
                bias_str += '%s: %.8f\t' % (name, bias)
        keys = ['Estim used', 'Bias', 'Var', 'N-Var']
        vals = [estim_str, bias_str, var_str, nvar_str]  # snr_str
        s = ''
        for k, v in zip(keys, vals):
            s += '%s: %s\t\t' % (k, v)
        return s

    def grad(self, use_sgd, *args, **kwargs):
        self.gest_used = not use_sgd
        return self.get_estim().grad(*args, **kwargs)

    def state_dict(self):
        state = {
            'gest_used': self.gest_used,
            'optim': [],
            'gest': [],
            'niters': self.niters
        }
        for name, optim in self.optim:
            state['optim'] += [optim.state_dict()]
        for name, g_estim in self.gest:
            state['gest'] += [g_estim.state_dict()]
        return state

    def load_state_dict(self, state):
        self.gest_used = state['gest_used']
        self.niters = state['niters']
        for i, (name, optim) in enumerate(self.optim):
            optim.load_state_dict(state['optim'][i])
        for i, (name, g_estim) in enumerate(self.gest):
            g_estim.load_state_dict(state['gest'][i])

    def get_estim(self):
        estim_id = self.gest_used if len(self.gest) > 1 else 0
        return self.gest[estim_id][1]

    def get_optim(self):
        optim_id = self.gest_used if len(self.optim) > 1 else 0
        return self.optim[optim_id][1]

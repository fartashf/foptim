from collections import OrderedDict, defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import os
import gc


def hist_bins(val):
    for i in [10, 1, .5, .2, .1, .01]:
        q_bins = np.percentile(val, np.arange(0, 100+.01, i),
                               interpolation='higher')
        q_bins = np.unique(np.round(q_bins, decimals=7))
        if len(q_bins) >= 10:
            break
    nb = 1.*int(1000./(len(q_bins)-1))
    bins = [np.arange(q_bins[i], q_bins[i+1],
                      np.float32((q_bins[i+1]-q_bins[i])/nb))
            for i in range(len(q_bins)-1)]
    bins += [q_bins[-1]]
    bins = np.hstack(bins)
    bins = np.unique(bins)
    return bins


class TBXWrapper(object):
    def configure(self, logger_name, flush_secs=5, opt=None):
        self.writer = SummaryWriter(logger_name, flush_secs=flush_secs)
        self.logger_name = logger_name
        self.logobj = defaultdict(lambda: list())
        self.opt = opt

    def state_dict(self):
        return {'logobj': dict(self.logobj)}

    def load_state_dict(self, state):
        self.logobj.update(state['logobj'])

    def log_value(self, name, val, step):
        self.writer.add_scalar(name, val, step)
        self.logobj[name] += [(time.time(), step, float(val))]
        if self.opt.log_nex:
            self.writer.add_scalar(name+'_Nex', val, step*self.opt.batch_size)
            self.logobj[name+'_Nex'] += [(time.time(),
                                          step*self.opt.batch_size,
                                          float(val))]

    def add_scalar(self, name, val, step):
        self.log_value(name, val, step)

    def log_hist(self, name, val, step, log_scale=False, bins=100):
        # https://github.com/lanpa/tensorboard-pytorch/issues/42
        # bins = hist_bins(val)
        if log_scale:
            val = np.log(np.maximum(np.exp(-20), val))
            name += '_log'
        # disabled because of bad bins using doane
        self.writer.add_histogram(name, val, step, bins='doane')
        self.logobj[name] += [(time.time(), step,
                               np.histogram(val, bins=bins))]

    def log_img(self, name, val, step):
        self.writer.add_image(name, val, step)
        # self.logobj[name] += [(time.time(), step, np.array(val))]

    def save_log(self, filename='log.pth.tar'):
        try:
            os.makedirs(self.opt.logger_name)
        except os.error:
            pass
        torch.save(dict(self.logobj), self.opt.logger_name+'/'+filename)

    def log_obj(self, name, val):
        self.logobj[name] = val

    def log_objs(self, name, val, step=None):
        self.logobj[name] += [(time.time(), step, val)]

    def log_vector(self, name, val, step=None):
        name += '_v'
        if step is None:
            step = len(self.logobj[name])
        self.logobj[name] += [(time.time(), step, list(val.flatten()))]

    def close(self):
        self.writer.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        if self.count == 0:
            return '%d' % self.val
        return '%.4f (%.4f)' % (self.val, self.avg)

    def tb_log(self, tb_logger, name, step=None):
        tb_logger.log_value(name, self.val, step=step)


class TimeMeter(object):
    """Store last K times"""

    def __init__(self, k=1000):
        self.k = k
        self.reset()

    def reset(self):
        self.vals = [0]*self.k
        self.i = 0
        self.mu = 0

    def update(self, val):
        self.vals[self.i] = val
        self.i = (self.i + 1) % self.k
        self.mu = (1-1./self.k)*self.mu+(1./self.k)*val

    def __str__(self):
        # return '%.4f +- %.2f' % (np.mean(self.vals), np.std(self.vals))
        return '%.4f +- %.2f' % (self.mu, np.std(self.vals))

    def tb_log(self, tb_logger, name, step=None):
        tb_logger.log_value(name, self.vals[0], step=step)


class StatisticMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mu = AverageMeter()
        self.std = AverageMeter()
        self.min = AverageMeter()
        self.max = AverageMeter()
        self.med = AverageMeter()

    def update(self, val, n=0):
        val = np.ma.masked_invalid(val)
        val = val.compressed()
        n = min(n, len(val))
        if n == 0:
            return
        self.mu.update(np.mean(val), n=n)
        self.std.update(np.std(val), n=n)
        self.min.update(np.min(val), n=n)
        self.max.update(np.max(val), n=n)
        self.med.update(np.median(val), n=n)

    def __str__(self):
        # return 'mu:{}|med:{}|std:{}|min:{}|max:{}'.format(
        #     self.mu, self.med, self.std, self.min, self.max)
        return 'mu:{}|med:{}'.format(self.mu, self.med)

    def tb_log(self, tb_logger, name, step=None):
        self.mu.tb_log(tb_logger, name+'_mu', step=step)
        self.med.tb_log(tb_logger, name+'_med', step=step)
        self.std.tb_log(tb_logger, name+'_std', step=step)
        self.min.tb_log(tb_logger, name+'_min', step=step)
        self.max.tb_log(tb_logger, name+'_max', step=step)


class PercentileMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.nz = AverageMeter()
        self.perc10 = AverageMeter()
        self.perc50 = AverageMeter()
        self.perc90 = AverageMeter()

    def update(self, val, n=0):
        val = np.ma.masked_invalid(val)
        val = val.compressed()
        n = min(n, len(val))
        if n == 0:
            return
        p = np.percentile(val, [10, 50, 90])
        self.nz.update(np.isclose(val, 0).sum(), n=n)
        self.perc10.update(float(p[0]), n=n)
        self.perc50.update(float(p[1]), n=n)
        self.perc90.update(float(p[2]), n=n)

    def __str__(self):
        # return 'mu:{}|med:{}|std:{}|min:{}|max:{}'.format(
        #     self.mu, self.med, self.std, self.min, self.max)
        return '0:{}|10:{}|50:{}|90:{}'.format(
            self.nz, self.perc10, self.perc50, self.perc90)

    def tb_log(self, tb_logger, name, step=None):
        self.nz.tb_log(tb_logger, name+'_nz', step=step)
        self.perc10.tb_log(tb_logger, name+'_perc10', step=step)
        self.perc50.tb_log(tb_logger, name+'_perc50', step=step)
        self.perc90.tb_log(tb_logger, name+'_perc90', step=step)


class HistogramMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, log_scale=False, bins=100):
        self.reset()
        self.log_scale = log_scale
        self.bins = bins

    def reset(self):
        self.nz = AverageMeter()
        self.perc10 = AverageMeter()
        self.perc50 = AverageMeter()
        self.perc90 = AverageMeter()
        self.val = 0

    def update(self, val, n=0):
        val = np.ma.masked_invalid(val)
        val = val.compressed()
        n = min(n, len(val))
        if n == 0:
            return
        p = np.percentile(val, [10, 50, 90])
        self.nz.update(np.isclose(val, 0).sum(), n=n)
        self.perc10.update(float(p[0]), n=n)
        self.perc50.update(float(p[1]), n=n)
        self.perc90.update(float(p[2]), n=n)
        self.val = val

    def __str__(self):
        # return 'mu:{}|med:{}|std:{}|min:{}|max:{}'.format(
        #     self.mu, self.med, self.std, self.min, self.max)
        return '0:{}|10:{}|50:{}|90:{}'.format(
            self.nz, self.perc10, self.perc50, self.perc90)

    def tb_log(self, tb_logger, name, step=None):
        self.nz.tb_log(tb_logger, name+'_nz', step=step)
        tb_logger.log_hist(name, self.val, step=step, log_scale=self.log_scale,
                           bins=self.bins)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self, opt):
        self.meters = OrderedDict()
        self.log_keys = opt.log_keys.split(',')

    def reset(self):
        self.meters = OrderedDict()

    def update(self, k, v, n=0, hist=False, log_scale=False, bins=100):
        if k not in self.meters:
            if type(v).__module__ == np.__name__:
                if hist:
                    # self.meters[k] = PercentileMeter()
                    self.meters[k] = HistogramMeter(log_scale, bins)
                else:
                    self.meters[k] = StatisticMeter()
            else:
                self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if k in self.log_keys or 'all' in self.log_keys:
                if i > 0:
                    s += '  '
                s += k+': '+str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        for k, v in self.meters.items():
            v.tb_log(tb_logger, prefix+k, step=step)


class Profiler(object):
    def __init__(self, k=10):
        self.k = k
        self.meters = OrderedDict()
        self.start()

    def tic(self):
        self.t = time.time()

    def toc(self, name):
        end = time.time()
        if name not in self.times:
            self.times[name] = []
        self.times[name] += [end-self.t]
        self.tic()

    def start(self):
        self.times = OrderedDict()
        self.tic()

    def end(self):
        for k, v in self.times.items():
            if k not in self.meters:
                self.meters[k] = TimeMeter(self.k)
            self.meters[k].update(sum(v))
        self.start()

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k+': ' + str(v)
        return s


def get_all_tensors():
    # https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741
    A = []
    for obj in gc.get_objects():
        try:
            if (torch.is_tensor(obj)
                    or (hasattr(obj, 'data') and torch.is_tensor(obj.data))):
                A += [obj]
        except Exception:
            pass
    B = [a.numel() for a in A]
    II = np.argsort(B)
    return [A[i] for i in II]


def get_memory(A):
    return sum([a.numel() for a in A])*4/1024/1024/1024

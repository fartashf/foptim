from collections import OrderedDict


def mnist(args):
    dataset = 'mnist'
    module_name = 'main.gvar'
    log_dir = 'runs_%s' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch']
    shared_args = [('dataset', dataset),
                   ('lr', [.1, .05, .01]),  # 0.02
                   ('weight_decay', 0),
                   ('epochs', [
                       (30, OrderedDict([('lr_decay_epoch', 30)])),
                   ]),
                   ('arch', ['mlp', 'cnn']),
                   ('optim', ['sgd']),
                   ]
    gvar_args = [
        # ('gvar_estim_iter', 10),
        # ('gvar_log_iter', 100),
        ('gvar_start', 0),
        ('g_bsnap_iter', 1),
        ('g_epoch', ''),
        # ('g_optim', ''),
        # ('g_optim_start', 0),
    ]
    args_sgd = [('g_estim', ['sgd,sgd']),
                ('g_batch_size', [128, 256])]
    args += [OrderedDict(shared_args+gvar_args+args_sgd)]

    args_svrg = [('g_estim', ['sgd,svrg'])]
    args += [OrderedDict(shared_args+gvar_args+args_svrg)]

    jobs_0 = ['bolt3_gpu0', 'bolt3_gpu1', 'bolt3_gpu2',
              'bolt2_gpu0', 'bolt2_gpu1', 'bolt2_gpu2',   # 'bolt2_gpu3',
              'bolt1_gpu3', 'bolt1_gpu2',
              'bolt1_gpu0', 'bolt1_gpu1'
              ]
    njobs = [3]*3 + [4]*3 + [3]*4
    return args, log_dir, module_name, exclude, jobs_0, njobs

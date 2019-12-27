import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
import os


def get_loaders(opt):
    if opt.dataset == 'mnist':
        return get_mnist_loaders(opt)
    elif opt.dataset == 'cifar10':
        return get_cifar10_loaders(opt)
    elif opt.dataset == 'cifar100':
        return get_cifar100_loaders(opt)
    elif opt.dataset == 'svhn':
        return get_svhn_loaders(opt)
    elif opt.dataset.startswith('imagenet'):
        return get_imagenet_loaders(opt)
    elif opt.dataset == 'logreg':
        return get_logreg_loaders(opt)
    elif 'class' in opt.dataset:
        return get_logreg_loaders(opt)
    elif opt.dataset == 'linreg':
        return get_linreg_loaders(opt)
    elif opt.dataset == 'rcv1':
        return get_rcv1_loaders(opt)
    elif opt.dataset == 'covtype':
        return get_covtype_loaders(opt)
    elif opt.dataset == 'protein':
        return get_protein_loaders(opt)


def get_gestim_loader(train_loader, opt):
    kwargs = {'num_workers': opt.workers,
              'pin_memory': True} if opt.cuda else {}
    idxdataset = train_loader.dataset
    train_loader = torch.utils.data.DataLoader(
        idxdataset,
        batch_size=opt.g_batch_size,
        shuffle=True,
        drop_last=False, **kwargs)
    return train_loader


def dataset_to_loaders(train_dataset, test_dataset, opt):
    kwargs = {'num_workers': opt.workers,
              'pin_memory': True} if opt.cuda else {}
    idxdataset = IndexedDataset(train_dataset, opt, train=True)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        idxdataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        IndexedDataset(test_dataset, opt),
        batch_size=opt.test_batch_size, shuffle=False,
        **kwargs)

    train_test_loader = torch.utils.data.DataLoader(
        IndexedDataset(train_dataset, opt, train=True,
                       cr_labels=idxdataset.cr_labels),
        batch_size=opt.test_batch_size, shuffle=False,
        **kwargs)
    return train_loader, test_loader, train_test_loader


class IndexedDataset(data.Dataset):
    def __init__(self, dataset, opt, train=False, cr_labels=None):
        np.random.seed(2222)
        self.ds = dataset
        self.opt = opt

        # duplicates
        self.dup_num = 0
        self.dup_cnt = 0
        self.dup_ids = []
        if opt.duplicate != '' and train:
            params = map(int, self.opt.duplicate.split(','))
            self.dup_num, self.dup_cnt = params
            self.dup_ids = np.random.permutation(len(dataset))[:self.dup_num]

        # corrupt labels
        if cr_labels is not None:
            self.cr_labels = cr_labels
        else:
            self.cr_labels = np.random.randint(
                self.opt.num_class, size=len(self))
        cr_ids = np.arange(len(self))
        self.cr_ids = []
        if train:
            cr_num = int(1. * opt.corrupt_perc * len(dataset) / 100.)
            self.cr_ids = cr_ids[:cr_num]

    def __getitem__(self, index):
        subindex = index
        if index >= len(self.ds):
            subindex = self.dup_ids[(index-len(self.ds))//self.dup_cnt]
        img, target = self.ds[subindex]
        if int(index) in self.cr_ids:
            # target = torch.tensor(self.cr_labels[index])
            target = self.cr_labels[index]
        return img, target, index

    def __len__(self):
        return len(self.ds)+self.dup_num*self.dup_cnt


def get_mnist_loaders(opt, **kwargs):
    transform = transforms.ToTensor()
    if not opt.no_transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST(
        opt.data, train=True, download=True, transform=transform)

    test_dataset = datasets.MNIST(opt.data, train=False, transform=transform)
    return dataset_to_loaders(train_dataset, test_dataset, opt, **kwargs)


def get_cifar10_100_transform(opt):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2023, 0.1994, 0.2010))

    # valid_size=0.1
    # split = int(np.floor(valid_size * num_train))
    # indices = list(range(num_train))
    # train_idx, valid_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    if opt.data_aug:
        transform = [
            transforms.RandomAffine(10, (.1, .1), (0.7, 1.2), 10),
            transforms.ColorJitter(.2, .2, .2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        if opt.no_transform:
            transform = [transforms.ToTensor(),
                         normalize]
        else:
            transform = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
    return normalize, transform


def get_cifar10_loaders(opt):
    normalize, transform = get_cifar10_100_transform(opt)

    train_dataset = datasets.CIFAR10(root=opt.data, train=True,
                                     transform=transforms.Compose(transform),
                                     download=True)
    test_dataset = datasets.CIFAR10(
        root=opt.data, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt)


def get_cifar100_loaders(opt):
    normalize, transform = get_cifar10_100_transform(opt)

    train_dataset = datasets.CIFAR100(
        root=opt.data, train=True,
        transform=transforms.Compose(transform), download=True)
    test_dataset = datasets.CIFAR100(
        root=opt.data, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt)


def get_svhn_loaders(opt, **kwargs):
    normalize = transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))
    if opt.data_aug:
        transform = [
            transforms.RandomAffine(10, (.1, .1), (0.7, 1.), 10),
            transforms.ColorJitter(.2, .2, .2),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ]

    train_dataset = torch.utils.data.ConcatDataset(
        (datasets.SVHN(
            opt.data, split='train', download=True,
            transform=transforms.Compose(transform)),
         datasets.SVHN(
             opt.data, split='extra', download=True,
             transform=transforms.Compose(transform))))
    test_dataset = datasets.SVHN(opt.data, split='test', download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5),
                                                          (0.5, 0.5, 0.5))
                                 ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt)


def get_imagenet_loaders(opt):
    # Data loading code
    traindir = os.path.join(opt.data, 'train')
    valdir = os.path.join(opt.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    test_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    return dataset_to_loaders(train_dataset, test_dataset, opt)


class InfiniteLoader(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.data_iter = iter([])
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            if isinstance(self.data_loader, list):
                II = self.data_loader
                self.data_iter = (II[i] for i in torch.randperm(len(II)))
            else:
                self.data_iter = iter(self.data_loader)
            data = next(self.data_iter)
        return data

    def next(self):
        # for python2
        return self.__next__()

    def __len__(self):
        return len(self.data_loader)


def random_orthogonal_matrix(gain, shape, noortho=False):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are "
                           "supported.")

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    if noortho:
        return a.reshape(shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return np.asarray(gain * q, dtype=np.float)


class LinearClassificationDataset(data.Dataset):

    def __init__(self, C, D, num, dim, num_class, train=True):
        X = np.zeros((C.shape[0], num))
        Y = np.zeros((num,))
        for i in range(num_class):
            n = num // num_class
            e = np.random.normal(0.0, 1.0, (dim, n))
            X[:, i * n:(i + 1) * n] = np.dot(D[:, :, i], e) + C[:, i:i + 1]
            Y[i * n:(i + 1) * n] = i
        self.X = X
        self.Y = Y
        self.classes = range(num_class)

    def __getitem__(self, index):
        X = torch.Tensor(self.X[:, index]).float()
        Y = int(self.Y[index])
        return X, Y

    def __len__(self):
        return self.X.shape[1]


class PlainDataset(data.Dataset):
    def __init__(self, data, num_class=2, ratio=1, perm=None, xmean=None,
                 xstd=None):
        self.data = data
        if xmean is None:
            self.xmean = np.array(self.data[0].mean(0))
            E2x = self.xmean**2
            Ex2 = self.data[0].copy()
            Ex2.data **= 2
            Vx = Ex2.mean(0) - E2x
            self.xstd = np.array(np.sqrt(Vx))
            self.xmean, self.xstd = self.xmean.flatten(), self.xstd.flatten()
        else:
            self.xmean, self.xstd = xmean, xstd
        self.ymin = self.data[1].min()
        self.ymax = self.data[1].max()
        self.num_class = num_class
        self.classes = range(int(num_class))
        N = self.data[0].shape[0]
        if perm is None:
            perm = np.random.permutation(N)
            self.ids = perm[:int(N*ratio)]
            self.no_ids = perm[int(N*ratio):]
        else:
            self.ids = perm

    def __getitem__(self, index):
        index = self.ids[index]
        xnorm = (self.data[0][index].toarray().flat
                 - self.xmean)/(self.xstd+1e-5)
        X = torch.Tensor(xnorm).float()
        Y = int((self.data[1][index]-self.ymin)/self.num_class)
        return X, Y

    def __len__(self):
        return len(self.ids)


class LibSVMDataset(PlainDataset):
    def __init__(self, fpath, *args, **kwargs):
        import sklearn
        import sklearn.datasets
        data = sklearn.datasets.load_svmlight_file(fpath)
        super(LibSVMDataset, self).__init__(data, *args, **kwargs)


class CSVDataset(PlainDataset):
    def __init__(self, fpath, *args, **kwargs):
        train_data = []
        train_labels = []
        import csv
        with open() as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for row in csv_reader:
                train_data += [float(x) for x in row[3:]]
                train_labels += [int(row[2])]
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        data = [train_data, train_labels]
        super(CSVDataset, self).__init__(data, *args, **kwargs)


def get_logreg_loaders(opt, **kwargs):
    # np.random.seed(1234)
    np.random.seed(2222)
    # print("Create W")
    # Harder with noortho
    C = opt.c_const * random_orthogonal_matrix(1.0, (opt.dim, opt.num_class),
                                               noortho=True)
    D = opt.d_const * random_orthogonal_matrix(
        1.0, (opt.dim, opt.dim, opt.num_class), noortho=True)
    # print("Create train")
    train_dataset = LinearClassificationDataset(
        C, D, opt.num_train_data, opt.dim, opt.num_class, train=True)
    # print("Create test")
    test_dataset = LinearClassificationDataset(
        C, D, opt.num_test_data, opt.dim, opt.num_class, train=False)
    torch.save((train_dataset.X, train_dataset.Y,
                test_dataset.X, test_dataset.Y,
                C), opt.logger_name + '/data.pth.tar')

    return dataset_to_loaders(train_dataset, test_dataset, opt)


class LinearRegressionDataset(data.Dataset):

    def __init__(self, C, D, num, dim, num_class, train=True):
        X = np.zeros((dim, num))
        Y = np.zeros((num_class, num))
        for i in range(num_class):
            n = num // num_class
            e = np.random.normal(0.0, 1.0, (dim, n))
            # X[:, i * n:(i + 1) * n] = np.dot(D[:, :, i], e) + C[:, i:i + 1]
            X[:, i * n:(i + 1) * n] = D * e + C
            e = np.random.normal(0.0, .1, (num_class, n))
            Y[:, i * n:(i + 1) * n] = i/num_class*0 + e  # i + e
        self.X = X
        self.Y = Y
        self.classes = range(num_class)

    def __getitem__(self, index):
        X = torch.Tensor(self.X[:, index]).float()
        Y = torch.Tensor(self.Y[:, index]).float()
        return X, Y

    def __len__(self):
        return self.X.shape[1]


def get_linreg_loaders(opt, **kwargs):
    # np.random.seed(1234)
    np.random.seed(2222)
    # print("Create W")
    # C = opt.c_const * random_orthogonal_matrix(1.0, (opt.dim, opt.num_class))
    # D = opt.d_const * random_orthogonal_matrix(
    #     1.0, (opt.dim, opt.dim, opt.num_class))
    # print("Create train")
    C = opt.c_const
    D = opt.d_const
    train_dataset = LinearRegressionDataset(
        C, D, opt.num_train_data, opt.dim, opt.num_class, train=True)
    # print("Create test")
    test_dataset = LinearRegressionDataset(
        C, D, opt.num_test_data, opt.dim, opt.num_class, train=False)
    torch.save((train_dataset.X, train_dataset.Y,
                test_dataset.X, test_dataset.Y,
                C), opt.logger_name + '/data.pth.tar')

    return dataset_to_loaders(train_dataset, test_dataset, opt)


def get_rcv1_loaders(opt, **kwargs):
    # train_dataset = LibSVMDataset(
    #     os.path.join(opt.data, 'rcv1_train.binary.bz2'),
    #     num_class=opt.num_class, ratio=0.5)
    # perm = train_dataset.no_ids
    # test_dataset = LibSVMDataset(
    #     os.path.join(opt.data, 'rcv1_train.binary.bz2'),
    #     num_class=opt.num_class, perm=perm)
    train_dataset = LibSVMDataset(
        os.path.join(opt.data, 'rcv1_train.binary.bz2'),
        num_class=opt.num_class)
    xmean, xstd = train_dataset.xmean, train_dataset.xstd
    test_dataset = LibSVMDataset(
        # os.path.join(opt.data, 'rcv1_test.binary.bz2'),
        # xmean=xmean, xstd=xstd)
        os.path.join(opt.data, 'rcv1_train.binary.bz2'),
        xmean=xmean, xstd=xstd)
    return dataset_to_loaders(train_dataset, test_dataset, opt, **kwargs)


def get_covtype_loaders(opt, **kwargs):
    np.random.seed(2222)
    train_dataset = LibSVMDataset(
        os.path.join(opt.data, 'covtype.libsvm.binary.scale.bz2'),
        num_class=opt.num_class, ratio=0.5)
    xmean, xstd = train_dataset.xmean, train_dataset.xstd
    perm = train_dataset.no_ids
    test_dataset = LibSVMDataset(
        os.path.join(opt.data, 'covtype.libsvm.binary.scale.bz2'),
        num_class=opt.num_class, perm=perm, xmean=xmean, xstd=xstd)
    return dataset_to_loaders(train_dataset, test_dataset, opt, **kwargs)


def get_protein_loaders(opt, **kwargs):
    np.random.seed(2222)
    train_dataset = CSVDataset(
        os.path.join(opt.data, 'bio_train.dat'),
        num_class=opt.num_class, ratio=0.5)
    xmean, xstd = train_dataset.xmean, train_dataset.xstd
    perm = train_dataset.no_ids
    test_dataset = CSVDataset(
        os.path.join(opt.data, 'bio_train.dat'),
        num_class=opt.num_class, perm=perm, xmean=xmean, xstd=xstd)
    return dataset_to_loaders(train_dataset, test_dataset, opt, **kwargs)

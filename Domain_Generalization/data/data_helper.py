from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import StandardDataset
from data.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info, JigsawTestDatasetMultiple
from data.concat_dataset import ConcatDataset
from data.JigsawLoader import JigsawNewDataset, JigsawTestNewDataset

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'
nex = 'nex'

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
nex_datasets=['Jul','Augu','Sep','Oct','Nov','Dec','batch_2','batch_3',]
new_nex_datasets = ['Nex_trainingset','Jan2021','Feb2021','Mar2021']
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets + nex_datasets + new_nex_datasets
#office_paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in office_datasets}
#pacs_paths = {dataset: "/home/enoon/data/images/PACS/kfold/%s" % dataset for dataset in pacs_datasets}
#vlcs_paths = {dataset: "/home/enoon/data/images/VLCS/%s/test" % dataset for dataset in pacs_datasets}
#paths = {**office_paths, **pacs_paths, **vlcs_paths}

dataset_std = {mnist: (0.30280363, 0.30280363, 0.30280363),
               mnist_m: (0.2384788, 0.22375608, 0.24496263),
               svhn: (0.1951134, 0.19804622, 0.19481073),
               synth: (0.29410212, 0.2939651, 0.29404707),
               usps: (0.25887518, 0.25887518, 0.25887518),
               nex: (0.2203, 0.2203, 0.2203),
               }

dataset_mean = {mnist: (0.13909429, 0.13909429, 0.13909429),
                mnist_m: (0.45920207, 0.46326601, 0.41085603),
                svhn: (0.43744073, 0.4437959, 0.4733686),
                synth: (0.46332872, 0.46316052, 0.46327512),
                usps: (0.17025368, 0.17025368, 0.17025368),
                nex: (0.1407, 0.1407, 0.1407),
                }


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_dataloader(args, patches):
    dataset_list = args.source
    assert isinstance(dataset_list, list)

    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)
    limit = args.limit_source
    for dname in dataset_list:
        if dname in new_nex_datasets:
            index_root = data_root = '/import/home/share/from_Nexperia_April2021/%s' % dname
        else:
            index_root = join(dirname(__file__),'correct_txt_lists')
            data_root = join(dirname(__file__),'kfold')
        name_train, labels_train = _dataset_info(join(index_root, "%s_train.txt" % dname))
        name_val, labels_val = _dataset_info(join(index_root, "%s_val.txt" % dname))
        train_dataset = JigsawNewDataset(data_root, name_train, labels_train, patches=patches, img_transformer=img_transformer,
                            tile_transformer=tile_transformer, jig_classes=30, bias_whole_image=args.bias_whole_image)
        if limit:
            train_dataset = Subset(train_dataset, limit)
        datasets.append(train_dataset)
        val_datasets.append(JigsawTestNewDataset(data_root,name_val, labels_val, img_transformer=get_val_transformer(args),patches=patches, jig_classes=30)) 

    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader

def get_val_dataloader(args, patches=False):
    dname = args.target
    if dname in new_nex_datasets:
        index_root = data_root = '/import/home/share/from_Nexperia_April2021/%s' % dname
    else:
        index_root = join(dirname(__file__),'correct_txt_lists')
        data_root = join(dirname(__file__),'kfold')
    names, labels = _dataset_info(join(index_root, "%s_val.txt" % dname))
    img_tr = get_nex_val_transformer(args)
    val_dataset = JigsawTestNewDataset(data_root, names, labels, patches=patches, img_transformer=img_tr, jig_classes=30)
    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return loader

def get_tgt_dataloader(args, patches=False):
    '''
    Load whole domain dataset
    '''
    img_tr = get_nex_val_transformer(args)
    dname = args.target
    if dname in new_nex_datasets:
        index_root = data_root = '/import/home/share/from_Nexperia_April2021/%s' % dname
    else:
        index_root = join(dirname(__file__),'correct_txt_lists')
        data_root = join(dirname(__file__),'kfold')
    if args.downsample_target:
        name_train, labels_train = _dataset_info(join(index_root,"%s_train_down.txt" % dname))
        name_val, labels_val = _dataset_info(join(index_root, "%s_val_down.txt" % dname))
        name_test, labels_test = _dataset_info(join(index_root, "%s_test_down.txt" % dname))
    else:
        name_train, labels_train = _dataset_info(join(index_root,"%s_train.txt" % dname))
        name_val, labels_val = _dataset_info(join(index_root, "%s_val.txt" % dname))
        name_test, labels_test = _dataset_info(join(index_root, "%s_test.txt" % dname))

    tgt_train_dataset = JigsawTestNewDataset(data_root, name_train, labels_train, patches=patches, img_transformer=img_tr,jig_classes=30)
    tgt_val_dataset = JigsawTestNewDataset(data_root, name_val, labels_val, patches=patches, img_transformer=img_tr,
                                            jig_classes=30)
    tgt_test_dataset = JigsawTestNewDataset(data_root, name_test, labels_test, patches=patches, img_transformer=img_tr,
                                            jig_classes=30)

    tgt_dataset = ConcatDataset([tgt_train_dataset, tgt_val_dataset, tgt_test_dataset])
    loader = torch.utils.data.DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    img_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    img_tr.append(transforms.ToTensor())
    img_tr.append(transforms.Normalize(dataset_mean[nex], dataset_std[nex]))

    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

def get_nex_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.2203, 0.2203, 0.2203], std=[0.1407, 0.1407, 0.1407])]
    return transforms.Compose(img_tr)
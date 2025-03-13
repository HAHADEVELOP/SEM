import json
import os
import sys

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms as trans

sys.path.append('./')

from PIL import Image

from config.path_config import ImageNetTestPATH, ImageNetCSV, \
    ImageNetTestPATH2, ImageNetCSV2, \
    TianChiS2DataPATH, TianChiS2DataCSV
from config.path_config import ImageNet2012Path, Cifar10_path

from utils.data_augmentation.auto_augment_ops import ImageNetPolicy, CIFAR10Policy


class BaseClassificationDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 csv_file=None,
                 trans=None,
                 eval_num=1000,
                 inmemory=True,
                 ):
        super(BaseClassificationDataset, self).__init__()
        self.data_path = data_path
        self.trans = trans
        self.eval_num = eval_num
        self.csv = csv_file
        self.read_pairs()
        self.inmemory = inmemory
        if self.inmemory:
            self.dump_inmemory()

    def dump_inmemory(self):
        self.imgs = []
        for i in range(len(self)):
            img = Image.open(os.path.join(self.data_path, self.img_ids[i]))
            if self.trans is not None:
                img = self.trans(img)
            self.imgs.append(img)

    def read_pairs(self):
        self.img_ids = []
        self.sclasses = []
        self.tclasses = []

    def __getitem__(self, index):
        name = self.img_ids[index]
        sclass = self.sclasses[name]
        tclass = self.tclasses[name]
        if self.inmemory:
            img = self.imgs[index]
        else:
            img = Image.open(os.path.join(self.data_path, self.img_ids[index]))
            if self.trans is not None:
                img = self.trans(img)
        return img, sclass, tclass, name

    def __len__(self):
        assert len(self.img_ids) == self.eval_num
        return self.eval_num


class GeekPwnPairs(BaseClassificationDataset):
    def __init__(self,
                 data_path=ImageNetTestPATH,
                 csv_file=ImageNetCSV,
                 trans=None,
                 eval_num=1000,
                 inmemory=True,
                 ):
        super(GeekPwnPairs, self).__init__(data_path, csv_file, trans, eval_num, inmemory)

    def read_pairs(self):
        csv_file = self.csv
        data = pd.read_csv(csv_file)
        self.img_ids = list(data['ImageId'])[:self.eval_num]
        self.tclasses = {name: tclass - 1 for name, tclass in zip(list(data['ImageId']), list(data['TargetClass']))}
        self.sclasses = {name: tclass - 1 for name, tclass in zip(list(data['ImageId']), list(data['TrueLabel']))}


class FGSM_data(BaseClassificationDataset):
    def __init__(self,
                 data_path=ImageNetTestPATH2,
                 csv_file=ImageNetCSV2,
                 trans=None,
                 eval_num=1000,
                 inmemory=True,
                 ):
        super(FGSM_data, self).__init__(data_path, csv_file, trans, eval_num, inmemory)

    def read_pairs(self):
        csv_file = self.csv
        data = pd.read_csv(csv_file)
        self.img_ids = list(data['filename'])[:self.eval_num]
        self.sclasses = {name: label - 1 for name, label in zip(list(data['filename']), list(data['label']))}
        self.tclasses = {name: -1 for name, label in zip(list(data['filename']), list(data['label']))}


class TianChiS2Data(BaseClassificationDataset):
    def __init__(self,
                 data_path=TianChiS2DataPATH,
                 csv_file=TianChiS2DataCSV,
                 trans=None,
                 eval_num=1000,
                 inmemory=True,
                 ):
        super(TianChiS2Data, self).__init__(data_path, csv_file, trans, eval_num, inmemory)

    def read_pairs(self):
        csv_file = self.csv
        data = pd.read_csv(csv_file)
        self.img_ids = list(data['ImageId'])[:self.eval_num]
        self.tclasses = {name: tclass - 1 for name, tclass in zip(list(data['ImageId']), list(data['TargetClass']))}
        self.sclasses = {name: tclass - 1 for name, tclass in zip(list(data['ImageId']), list(data['TrueLabel']))}


class ImageNet_Index2Label:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__),
                               '../evaluate/eval_models/imagenet/imagenet_class_index.json'), 'r') as f:
            self.dict = json.load(f)

    def __call__(self, index):
        return self.dict[index]


def create_imagenet(
        path=ImageNet2012Path,
        train=False,
        data_transform=trans.Compose([
            trans.Resize((299, 299)),
            ImageNetPolicy(),
            trans.ToTensor()
        ])
):
    if train:
        dataset = datasets.ImageFolder(root=os.path.join(path, 'train'), transform=data_transform)
    else:
        dataset = datasets.ImageFolder(root=os.path.join(path, 'val'), transform=data_transform)

    return dataset


#################################### CIFAR ####################################

def create_cifar10(
        path=Cifar10_path,
        train=False,
        transform=trans.Compose([
            trans.Resize(256),
            CIFAR10Policy(),
            trans.ToTensor()
        ])
):
    return datasets.CIFAR10(path, train, transform, download=True)


if __name__ == '__main__':
    dta = create_cifar10()
    loader = DataLoader(dta, batch_size=2)
    for x in loader:
        pass

# coding: utf-8
import json
import os
import sys
import warnings

import matplotlib
import numpy as np
# import tensorflow as tf
import pandas as pd
import torch

warnings.filterwarnings(action='ignore')

sys.path.append('/')
matplotlib.use('Agg')

from functools import partial
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn import functional as F
from torchvision import transforms as trans
from PIL import Image
from pprint import pprint

from evaluate.defense import JpegCompression, TotalVarMin, FeatureDistillation, FeatureSqueezing, NRP_resnet_purifier, \
    Smooth, HGD
from data_.datasets import FGSM_data
from evaluate.eval_adv.eval_transferability.fgsm import evaluate
from config.path_config import ImageNetTestPATH2, ImageNetCSV2
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms import _pil_interp


class FGSM_data_defense(Dataset):
    def __init__(self,
                 path=ImageNetTestPATH2,
                 csv_file=ImageNetCSV2,
                 transforms=None,
                 targeted=False,
                 eval_num=1000,
                 defenseMethod=None,
                 use_saved=False,
                 ):
        self.path = path
        self.csv = csv_file
        self.eval_num = eval_num
        self.trans = transforms
        self.targeted = targeted
        assert targeted == False
        self.read_pairs()

        self.topil = trans.ToPILImage()
        self.totensor = trans.ToTensor()

        self.used_saved = use_saved
        if use_saved:
            self.temp_adv_dir = os.path.join('/xxx',
                                             path.replace('output/SE/single_model_integrate/', ''), defenseMethod)
            os.makedirs(self.temp_adv_dir, exist_ok=True)
            self.saved = len(os.listdir(self.temp_adv_dir))

    def read_pairs(self):
        csv_file = self.csv
        data = pd.read_csv(csv_file)
        self.img_ids = list(data['filename'])[:self.eval_num]
        self.tclasses = [label - 1 for label in list(data['label'])[:self.eval_num]]

    def __getitem__(self, index):
        if self.used_saved:
            try:
                img = Image.open(os.path.join(self.temp_adv_dir, self.img_ids[index]))
                img = self.totensor(img)
            except:
                img = Image.open(os.path.join(self.path, self.img_ids[index]))
                if self.trans is not None:
                    img = self.trans(img)
                    self.topil(img).save(os.path.join(self.temp_adv_dir, self.img_ids[index]), format='png')
                    self.saved += 1
                    print('saved +1, now', self.saved)
        else:
            img = Image.open(os.path.join(self.path, self.img_ids[index]))
            if self.trans is not None:
                img = self.trans(img)
        return img, self.tclasses[index], self.img_ids[index]

    def __len__(self):
        assert len(self.img_ids) == self.eval_num
        return self.eval_num


def evaluate_preprocessor(method, adv_path, model_name):
    print('evaluating ', method)
    if method == 'JPEG':
        preprocessor = JpegCompression(quality=75)
    elif method == 'TVM':
        preprocessor = TotalVarMin()
    elif method == 'FD':
        preprocessor = FeatureDistillation()
    elif method == 'Bit-Reduction':
        preprocessor = FeatureSqueezing(bit_depth=4)
    else:
        raise NotImplementedError

    def preprocessing(x):
        x = np.array(x, dtype=np.float32)[None, :] / 255
        x = preprocessor(x)[0]
        x = torch.from_numpy(x.transpose([2, 0, 1]))
        return x

    if method == 'TVM':
        adv_data = FGSM_data_defense(adv_path, transforms=preprocessing, defenseMethod=method, use_saved=True)
    else:
        adv_data = FGSM_data_defense(adv_path, transforms=preprocessing, defenseMethod=method)

    loader = DataLoader(adv_data, batch_size=125, num_workers=20)

    eval_names = [
        model_name
    ]
    eval_models = []
    eval_transforms = []
    for chosen_model in eval_names:
        model = create_model(chosen_model, pretrained=True).eval().cuda()
        input_config = resolve_data_config({}, model=model)

        current_tran = trans.Compose([
            trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
            trans.Normalize(input_config['mean'], input_config['std'])
        ])

        eval_transforms.append(current_tran)
        eval_models.append(model)

    res = evaluate(
        eval_models,
        eval_names,
        loader,
        eval_transforms
    )
    return res[0]


@torch.no_grad()
def evaluate_defensemodels(path):
    from models.ImageNetModels import model_name_dict
    denoise_models = [model_name_dict[name]().cuda().eval() for name in model_name_dict]

    dataset = FGSM_data(path,
                        trans=trans.Compose([
                            trans.Resize((299, 299)),
                            trans.ToTensor(),
                            trans.Normalize(mean=[.5, .5, .5],
                                            std=[.5, .5, .5])
                        ]))
    loader = DataLoader(dataset, batch_size=125, num_workers=32)
    return evaluate(
        denoise_models,
        list(model_name_dict.keys()),
        loader,
        transforms=[trans.Compose([]) for _ in model_name_dict]
    )


@torch.no_grad()
def evaluate_RandomSmooth(path, model_name):
    print('evaluating random smooth')

    model = create_model(model_name, pretrained=True).eval().cuda()
    input_config = resolve_data_config({}, model=model)

    current_tran = trans.Compose([
        trans.ToTensor(),
        trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
        trans.Normalize(input_config['mean'], input_config['std'])
    ])
    dataset = FGSM_data(
        path,
        trans=current_tran)

    smoother = Smooth(model)
    success_num = 0

    for _ in range(1000):
        x, y, name = dataset[_]
        x = x.cuda()
        y_pred = smoother.predict(x)
        if y_pred != y:
            success_num += 1
    print('final_res', success_num)

    return success_num / 1000


@torch.no_grad()
def evaluate_RP2(path, model_name, itr=30):
    print('evaluate_RP2', )
    model = create_model(model_name, pretrained=True).eval().cuda()
    input_config = resolve_data_config({}, model=model)
    from utils.data_augmentation.input_transformation import DI_diversity
    current_tran = trans.Compose([
        partial(DI_diversity, div_prob=1., low=1.037, constant=0),
        trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
        trans.Normalize(input_config['mean'], input_config['std'])
    ])
    dataset = FGSM_data(
        path,
        trans=trans.ToTensor(), )
    loader = DataLoader(dataset, batch_size=125, num_workers=32)
    transforms = [current_tran]
    res = []
    for name, model, transform in (zip([model_name], [model], transforms)):
        total_num = len(loader.dataset)
        try:
            a, b, c = loader.dataset[0]
            num = 3
        except:
            num = 2
        success_num = torch.zeros(1).cuda()
        for batch in loader:
            if num == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.cuda()
            y = y.cuda()
            y_ = 0
            for _ in range(itr):
                y_ += model(transform(x))
            y_ = torch.argmax(y_, 1)
            success_num += torch.sum(y_ != y)

        success_num = success_num.item()
        print('data_path {}, model {}, success rate {}'.format(
            loader.dataset.path if hasattr(loader.dataset, 'data_path') else None, name, success_num / total_num))
        res.append(success_num / total_num)
    return res[0]


@torch.no_grad()
def evaluate_HGD(path, model_name):
    print('evaluate_HGD', )
    model = create_model(model_name, pretrained=True).eval().cuda()
    input_config = resolve_data_config({}, model=model)
    denoiser = HGD().denoiser

    def return11(x):
        return (x - 0.5) * 2

    def return01(x):
        return (torch.clamp(x, -1., 1.) + 1) / 2

    current_tran = trans.Compose([
        trans.Resize((299, 299)),
        return11,
        denoiser.cuda().eval(),
        return01,
        trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
        trans.Normalize(input_config['mean'], input_config['std'])
    ])
    dataset = FGSM_data(
        path,
        trans=trans.ToTensor(), )

    return evaluate(
        [model],
        [model_name],
        loader=DataLoader(dataset, batch_size=125, num_workers=32),
        transforms=[current_tran]
    )[0]


def evaluate_SuperResolution(path, model_name):
    print('evaluate_SuperResolution', path)

    model = create_model(model_name, pretrained=True).eval().cuda()
    input_config = resolve_data_config({}, model=model)

    current_tran = trans.Compose([
        trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
        trans.Normalize(input_config['mean'], input_config['std'])
    ])
    dataset = FGSM_data(
        trans=trans.ToTensor(),
        inmemory=False
    )
    path = '-'.join(path.split('/')[-2:])
    adv_path = os.path.join('../super-resolution-adversarial-defense/src', path + '-super_resolution.pth')
    imgs = torch.load(adv_path)
    dataset.imgs = [imgs[name].cpu() for name in dataset.img_ids]
    dataset.inmemory = True

    return evaluate(
        [model],
        [model_name],
        loader=DataLoader(dataset, batch_size=125, num_workers=32),
        transforms=[current_tran]
    )[0]


def evaluate_ComDefend(path, model_name):
    print('evaluate_ComDefend', )

    model = create_model(model_name, pretrained=True).eval().cuda()
    input_config = resolve_data_config({}, model=model)

    current_tran = trans.Compose([
        trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
        trans.Normalize(input_config['mean'], input_config['std'])
    ])
    dataset = FGSM_data(
        path.replace('./output/SE/single_model_integrate', 'output/SE/defense') + '/ComDefend',
        trans=trans.ToTensor(), )

    return evaluate(
        [model],
        [model_name],
        loader=DataLoader(dataset, batch_size=125, num_workers=32),
        transforms=[current_tran]
    )[0]


def evaluate_PixelDeflection(path, model_name):
    print('evaluate_PixelDeflection', )

    model = create_model(model_name, pretrained=True).eval().cuda()
    input_config = resolve_data_config({}, model=model)

    current_tran = trans.Compose([
        trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
        trans.Normalize(input_config['mean'], input_config['std'])
    ])
    dataset = FGSM_data(
        path.replace('./output/SE/single_model_integrate', 'output/SE/defense') + '/PixelDeflection',
        trans=trans.ToTensor(), )

    return evaluate(
        [model],
        [model_name],
        loader=DataLoader(dataset, batch_size=125, num_workers=32),
        transforms=[current_tran]
    )[0]


def evaluate_NRP(path, model_name):
    print('evaluate_NRP', )
    model = create_model(model_name, pretrained=True).eval().cuda()
    input_config = resolve_data_config({}, model=model)
    purifier = NRP_resnet_purifier().netG.cuda().eval()
    # purifier = NRP_purifier().netG.cuda().eval()

    current_tran = trans.Compose([
        # trans.Resize((224, 224)),
        purifier,
        trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
        trans.Normalize(input_config['mean'], input_config['std'])
    ])
    dataset = FGSM_data(
        path,
        trans=trans.ToTensor(), )

    return evaluate(
        [model],
        [model_name],
        loader=DataLoader(dataset, batch_size=125, num_workers=32),
        transforms=[current_tran]
    )[0]


@torch.no_grad()
def scale_ensemble(path, model_name):
    print('evaluate_SE', )
    best_range = (0.7, 1.3)
    num = 100
    eval_model = model_name

    eval_names = [
        eval_model
    ]
    eval_models = []
    eval_transforms = []

    loader = DataLoader(FGSM_data(path,
                                  trans=trans.ToTensor(), eval_num=1000),
                        batch_size=125,
                        )

    for chosen_model in eval_names:
        model = create_model(chosen_model, pretrained=True).eval().cuda()
        input_config = resolve_data_config({}, model=model)

        current_tran = trans.Compose([
            # trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
            trans.Normalize(input_config['mean'], input_config['std'])
        ])

        eval_transforms.append(current_tran)
        eval_models.append(model)

        ori_size = input_config['input_size'][1]
        new_sizes = ori_size * np.array([best_range[0] + i * (best_range[1] - best_range[0]) / num for i in range(num)])
        new_sizes = np.unique(new_sizes.astype('int'))

        true_labels = []
        pred_labels0 = []
        pred_labels = []
        pred_labels2 = []
        success_num0 = 0
        success_num = 0
        success_num2 = 0
        for x, y, _ in tqdm(loader):
            x = x.cuda()
            true_labels.append(y.numpy())
            logits = 0
            logits2 = 0
            labels = []
            for new_size in new_sizes:
                logit = model(F.interpolate(current_tran(x), (new_size, new_size)))
                logits += logit
                logits2 += F.softmax(logit, 1)
                labels.append(torch.argmax(logit, 1).cpu().numpy())
            labels = np.array(labels)
            pred_labels0.append(labels)
            pred_labels.append(torch.argmax(logits, 1).cpu().numpy())
            pred_labels2.append(torch.argmax(logits2, 1).cpu().numpy())

        true_labels = np.concatenate(true_labels)
        pred_labels0 = np.concatenate(pred_labels0, 1).transpose([1, 0])
        pred_labels = np.concatenate(pred_labels)
        pred_labels2 = np.concatenate(pred_labels2)

        for true_label, pred_label in zip(true_labels, pred_labels0):
            pred_dict = {
                i: sum(np.where(pred_label == i, 1, 0)) for i in np.unique(pred_label)
            }
            pred = pred_label[0]
            for key in pred_dict.keys():
                if pred_dict[key] > pred_dict[pred]:
                    pred = key
            if pred != true_label:
                success_num0 += 1
        # print(success_num0 / len(pred_labels))
        SR1 = success_num0 / len(pred_labels)

        for true_label, pred_label in zip(true_labels, pred_labels):
            if pred_label != true_label:
                success_num += 1
        # print(success_num / len(pred_labels))
        SR2 = success_num / len(pred_labels)

        for true_label, pred_label in zip(true_labels, pred_labels2):
            if pred_label != true_label:
                success_num2 += 1
        # print(success_num2 / len(pred_labels))
        SR3 = success_num2 / len(pred_labels)
        return SR1, SR2, SR3


def evaluate_all():
    attack_names = [
        'inception_v3',
        'inception_v4',
        'inception_resnet_v2',
        'resnet50',
        'vgg16',
        'densenet121',
    ]

    with open('output/SE/defense/all_res.json', 'r') as f:
        all_res = json.load(f)

    eval_model = 'ens_adv_inception_resnet_v2'

    for src_model in attack_names:
        for opt in ['MI', 'NI']:
            for sub_attack in ['PB', 'ALL']:
                adv_path = './output/SE/single_model_integrate/{}/{}-{}-VTI-FGSM'.format(src_model, opt, sub_attack)

                if "{}-{}-{}-VTI-FGSM".format(src_model, opt, sub_attack) in all_res:
                    res = all_res["{}-{}-{}-VTI-FGSM".format(src_model, opt, sub_attack)]
                else:
                    res = {}

                if 'JPEG' not in res:
                    res['JPEG'] = evaluate_preprocessor('JPEG', adv_path, eval_model)
                if 'FD' not in res:
                    res['FD'] = evaluate_preprocessor('FD', adv_path, eval_model)
                if 'Bit-Reduction' not in res:
                    res['Bit-Reduction'] = evaluate_preprocessor('Bit-Reduction', adv_path, eval_model)
                if 'TVM' not in res:
                    res['TVM'] = evaluate_preprocessor('TVM', adv_path, eval_model)
                if 'ComDefend' not in res:
                    res['ComDefend'] = evaluate_ComDefend(adv_path, eval_model)
                if 'R&P' not in res:
                    res['R&P'] = evaluate_RP2(adv_path, eval_model)
                # if 'SR' not in res:
                res['SR'] = evaluate_SuperResolution(adv_path, eval_model)
                if 'NRP' not in res:
                    res['NRP'] = evaluate_NRP(adv_path, eval_model)
                if 'HGD' not in res:
                    res['HGD'] = evaluate_HGD(adv_path, eval_model)
                if 'RS' not in res:
                    res['RS'] = evaluate_RandomSmooth(adv_path, eval_model)
                if 'PD' not in res:
                    res['PD'] = evaluate_PixelDeflection(adv_path, eval_model)
                if 'SE' not in res:
                    res['SE'] = scale_ensemble(adv_path, eval_model)

                all_res["{}-{}-{}-VTI-FGSM".format(src_model, opt, sub_attack)] = res
    pprint(all_res)
    with open('output/SE/defense/all_res.json', 'w') as f:
        json.dump(all_res, f)


def compute_results():
    attack_names = [
        'inception_v3',
        # 'inception_v4',
        'inception_resnet_v2',
        'resnet50',
        # 'vgg16',
        'densenet121',
    ]
    all_defense = [
        'JPEG', 'FD', 'Bit-Reduction', 'TVM', 'HGD', 'R&P', 'RS', 'SR', 'PD',
        'NRP'
    ]
    print('\t' + ' & '.join(all_defense))
    a = 0
    b = 0
    with open('output/SE/defense/all_res.json', 'r') as f:
        all_res = json.load(f)
    for src_idx, src_model in enumerate(attack_names):
        for opt in ['MI', 'NI']:
            for sub_attack in ['PB', 'ALL']:
                key = "{}-{}-{}-VTI-FGSM".format(src_model, opt, sub_attack)
                scores = all_res[key]

                if sub_attack == 'ALL':
                    print('\t & SE-PB-{}-FGSM'.format(opt), end='\t')
                else:
                    print('\t & PB-{}-FGSM'.format(opt), end='\t')

                for method in all_defense:
                    score = scores[method]
                    if method == 'SE':
                        score = score[0]

                    if sub_attack == 'ALL':
                        b += score
                    else:
                        a += score

                    if score == max(
                            [all_res["{}-{}-{}-VTI-FGSM".format(src_model, opt, x)][method] for x in ['PB', 'ALL']]):
                        print(' &  \\textbf{%.1f}' % (score * 100), end='')
                    else:
                        print(' & %.1f' % (score * 100), end='')

                print(' \\\\')
            print('\t \\cline{2-12}')
        print()
    print(b / len(all_defense) / len(attack_names) / 2, a / len(all_defense) / len(attack_names) / 2)


def evaluate_adv(path, model_name):
    print('evaluate', path)
    model = create_model(model_name, pretrained=True).eval().cuda()
    input_config = resolve_data_config({}, model=model)

    current_tran = trans.Compose([
        trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
        trans.Normalize(input_config['mean'], input_config['std'])
    ])
    dataset = FGSM_data(
        path,
        trans=trans.ToTensor(), )

    return evaluate(
        [model],
        [model_name],
        loader=DataLoader(dataset, batch_size=125, num_workers=32),
        transforms=[current_tran]
    )


if __name__ == '__main__':
    # test()
    # evaluate_all()
    compute_results()

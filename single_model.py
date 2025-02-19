"""
Code for single model attack

"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import json
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms import _pil_interp
from torch.utils.data import DataLoader
from torchvision import transforms as trans

sys.path.append('/')

from datasets import FGSM_data
from fgsm import evaluate, fgsm
from tqdm import tqdm

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

def attack(model_names, distributed, out_dir, eval_num, batch_size, best_params):
    for chosen_model in tqdm(model_names):
        model = create_model(chosen_model, pretrained=True).eval().cuda()
        input_config = resolve_data_config({}, model=model)

        current_tran = trans.Compose([
            trans.Normalize(input_config['mean'], input_config['std'])
        ])
        attack_dta = FGSM_data("/home/xxx", transform=trans.Compose([
            trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
        ]), max_samples=1000)
        
        loader = torch.utils.data.DataLoader(
                attack_dta,
                batch_size=batch_size, shuffle=False,
                num_workers=4)

        best_param = best_params[chosen_model]
        spatial_list = tuple([best_param[0] + (best_param[1] - best_param[0]) * i / 50 for i in range(50)])

        for opt in ['MI', 'NI']:
            if opt == 'NI':
                Nestrov = True
            else:
                Nestrov = False
            
            fgsm(
                [model],
                loader,
                [current_tran],
                os.path.join(out_dir, chosen_model, opt + '-SE-FGSM'),
                momentum=1.,
                nestrov=Nestrov,
                scale_list=[1.],
                tim_kernel=1,
                input_transform=lambda x: x,
                vt_number=0,
                ori_sizes=(input_config['input_size'][1],),
                spatial_lists=(spatial_list,),
                vr_m=1,
                attack_iter=10,
            )


def eval(attack_names, eval_names, distributed, out_dir, eval_num, batch_size):

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

    all_res = {}
    for attack_name in attack_names:
        for opt in ['MI', 'NI']:
            for transform_method in ['VR', 'DI', 'SI']:
                eval_dta = FGSM_data(os.path.join(out_dir, attack_name, opt + '-' + transform_method + '-FGSM'), 
                                     transform=trans.Compose([
                                        #  trans.ToTensor(),
                                     ]),
                                     max_samples=1000)
                loader = DataLoader(eval_dta, batch_size=batch_size)
                all_res[attack_name + '-' + opt + '-' + transform_method + '-FGSM'] = \
                    evaluate(
                        eval_models,
                        eval_names,
                        loader,
                        eval_transforms,
                    )

    with open(os.path.join(out_dir, 'all_evalate_results.json'), 'w') as f:
        json.dump(all_res, f)


def main():
    distributed = False

    debug = 0
    if debug:
        distributed = False
        eval_num = 4
        batch_size = 2
    else:
        eval_num = 1000
        batch_size = 64

    attack_names = [
        'inception_v3',
        'inception_resnet_v2',
        'resnet50',
        'densenet121',
    ]

    eval_names = [
        'inception_v3',
        'inception_resnet_v2',
        'resnet50',
        'densenet121',
        'adv_inception_v3',
        'ens_adv_inception_resnet_v2'
    ]

    best_params = {
        'inception_v3': (0.7, 1.3),
        'inception_resnet_v2': (0.5, 1.1),
        'resnet50': (0.7, 1.3),
        'densenet121': (0.7, 1.3),
    }

    out_dir = '/home/xxx'

    attack(attack_names, distributed, out_dir, eval_num, batch_size, best_params)
    

def collect_results():
    out_dir = '/home/xxx'
    res_file = os.path.join(out_dir, 'all_evalate_results.json')
    with open(res_file, 'r') as f:
        all_res = json.load(f)
    attack_names = [
        'inception_v3',
        'inception_resnet_v2',
        'resnet50',
        'densenet121',
    ]
    for attackid, attack_name in enumerate(attack_names):
        for opt in ['MI', 'NI']:
            compare_scores = []
            for transform_method in ['VR', 'SE-VR', 'DI', 'SE-DI', 'SI', 'SE-SI']:
                scores = all_res[attack_name + '-' + opt + '-' + transform_method + '-FGSM']
                scores = np.concatenate([scores[:2], scores[2:5], scores[5:]], 0)
                compare_scores.append(scores)
            compare_scores = np.array(compare_scores)

            out = []
            for idx, transform_method in enumerate(['VR', 'SE-VR', 'DI', 'SE-DI', 'SI', 'SE-SI']):
                print('\t& ' + transform_method + '-' + opt + '-FGSM &', end='\t')
                inner_out = []
                for i in range(len(compare_scores[0])):
                    score = compare_scores[idx][i]
                    is_max = score == np.max(compare_scores[:, i])
                    score = "%.1f" % (score * 100)
                    if i == attackid:
                        score += '*'
                    if is_max:
                        score = '\\textbf{' + score + '}'
                    inner_out.append(score)
                print(' & '.join(inner_out), end=' \\\\\n')
            print('\t\\cline{2-8}')
        print('\n\n')


if __name__ == '__main__':
    main()
    # attack()
    # collect_results()

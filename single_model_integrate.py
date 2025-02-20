"""
Code for single model attack, integrated attack

compare DI, SI, SE, based on MI and NI, combined with TI and VT



"""

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

from data_.datasets import FGSM_data
from evaluate.eval_adv.eval_transferability.fgsm import evaluate, fgsm

if not os.path.exists('/root/.cache/torch/hub'):
    os.system(
        'mkdir -p /root/.cache/torch/hub ; ln -s /xxx/timm_ckpts /root/.cache/torch/hub/checkpoints')

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)


def attack(model_names, distributed, out_dir, eval_num, batch_size, best_params):
    for chosen_model in model_names:
        model = create_model(chosen_model, pretrained=True).eval().cuda()
        input_config = resolve_data_config({}, model=model)

        current_tran = trans.Compose([
            # trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
            trans.Normalize(input_config['mean'], input_config['std'])
        ])
        attack_dta = FGSM_data(trans=trans.Compose([
            trans.Resize(input_config['input_size'][1:], interpolation=_pil_interp(input_config['interpolation'])),
            trans.ToTensor(),
        ]), eval_num=eval_num)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                attack_dta,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False)
            loader = torch.utils.data.DataLoader(
                attack_dta,
                batch_size=batch_size, sampler=sampler,
                num_workers=4)
            half_loader = torch.utils.data.DataLoader(
                attack_dta,
                batch_size=batch_size // 2 + 1, sampler=sampler,
                num_workers=4)
        else:
            loader = torch.utils.data.DataLoader(
                attack_dta,
                batch_size=batch_size, shuffle=False,
                num_workers=4)

            half_loader = torch.utils.data.DataLoader(
                attack_dta,
                batch_size=batch_size // 2 + 1, shuffle=False,
                num_workers=4)

        best_param = best_params[chosen_model]
        spatial_list = tuple([best_param[0] + (best_param[1] - best_param[0]) * i / 50 for i in range(50)])

        for opt in ['MI', 'NI']:
            if opt == 'NI':
                nestrov = True
            else:
                nestrov = False

            fgsm(
                [model],
                loader,
                [current_tran],
                os.path.join(out_dir, chosen_model, opt + '-VR-VTI-FGSM'),
                momentum=1.,
                nestrov=nestrov,
                scale_list=[1.],
                # tim_kernel=1,
                input_transform=lambda x: x,
                # vt_number=0,
                ori_sizes=(input_config['input_size'][1],),
                spatial_lists=((1,),),
                # vr_m=1,
                attack_iter=10,
            )

            fgsm(
                [model],
                loader,
                [current_tran],
                os.path.join(out_dir, chosen_model, opt + '-DI-VTI-FGSM'),
                momentum=1.,
                nestrov=nestrov,
                scale_list=[1.],
                # tim_kernel=1,
                # input_transform=lambda xs:xs,
                # vt_number=0,
                ori_sizes=(input_config['input_size'][1],),
                spatial_lists=((1,),),
                vr_m=1,
                attack_iter=10,
            )

            fgsm(
                [model],
                loader,
                [current_tran],
                os.path.join(out_dir, chosen_model, opt + '-SI-VTI-FGSM'),
                momentum=1.,
                nestrov=nestrov,
                # scale_list=[1.],
                # tim_kernel=1,
                input_transform=lambda x: x,
                # vt_number=0,
                ori_sizes=(input_config['input_size'][1],),
                spatial_lists=((1,),),
                vr_m=1,
                attack_iter=10,
            )

            if chosen_model in ['inception_resnet_v2']:
                _loader = half_loader
            else:
                _loader = loader

            fgsm(
                [model],
                _loader,
                [current_tran],
                os.path.join(out_dir, chosen_model, opt + '-SE-VTI-FGSM'),
                momentum=1.,
                nestrov=nestrov,
                scale_list=[1.],
                # tim_kernel=1,
                input_transform=lambda x: x,
                # vt_number=0,
                ori_sizes=(input_config['input_size'][1],),
                spatial_lists=(spatial_list,),
                vr_m=1,
                attack_iter=10,
            )

            fgsm(
                [model],
                _loader,
                [current_tran],
                os.path.join(out_dir, chosen_model, opt + '-ALL-VTI-FGSM'),
                momentum=1.,
                nestrov=nestrov,
                # scale_list=[1.],
                # tim_kernel=1,
                # input_transform=lambda xs:xs,
                # vt_number=0,
                ori_sizes=(input_config['input_size'][1],),
                spatial_lists=(spatial_list,),
                vr_m=1,
                attack_iter=10,
            )

            fgsm(
                [model],
                _loader,
                [current_tran],
                os.path.join(out_dir, chosen_model, opt + '-PB-VTI-FGSM'),
                momentum=1.,
                nestrov=nestrov,
                # scale_list=[1.],
                # tim_kernel=1,
                # input_transform=lambda xs:xs,
                # vt_number=0,
                ori_sizes=(input_config['input_size'][1],),
                spatial_lists=((1,),),
                # vr_m=1,
                attack_iter=10,
            )


def eval(attack_names, eval_names, distributed, out_dir, eval_num, batch_size):
    if distributed and dist.get_rank():
        return

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
    with open(os.path.join(out_dir, 'all_evalate_results.json'), 'r') as f:
        all_res = json.load(f)
    for attack_name in attack_names:
        for opt in ['MI', 'NI']:
            for transform_method in ['VR', 'DI', 'SI', 'SE', 'PB', 'ALL']:

                if attack_name + '-' + opt + '-' + transform_method + '-VTI-FGSM' in all_res:
                    continue

                eval_dta = FGSM_data(
                    path=os.path.join(out_dir, attack_name, opt + '-' + transform_method + '-VTI-FGSM'),
                    trans=trans.Compose([
                        trans.ToTensor(),
                    ]),
                    eval_num=eval_num)
                loader = DataLoader(eval_dta, batch_size=batch_size)
                all_res[attack_name + '-' + opt + '-' + transform_method + '-VTI-FGSM'] = \
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
        batch_size = 125

    if distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device("cuda:{}".format(dist.get_rank() % torch.cuda.device_count()))

    attack_names = [
        'inception_v3',
        'inception_v4',
        'inception_resnet_v2',
        'resnet50',
        'vgg16',
        'densenet121',
    ]

    eval_names = [
        'inception_v3',
        'inception_v4',
        'inception_resnet_v2',
        'resnet50',
        'vgg16',
        'densenet121',
        'adv_inception_v3',
        'ens_adv_inception_resnet_v2'
    ]

    best_params = {
        'inception_v3': (0.7, 1.3),
        'inception_v4': (0.5, 1.1),
        'inception_resnet_v2': (0.5, 1.1),
        'resnet50': (0.7, 1.3),
        'vgg16': (0.5, 1.1),
        'densenet121': (0.7, 1.3),

    }

    out_dir = '/xxx'

    # attack(attack_names, distributed, out_dir, eval_num, batch_size, best_params)

    eval(attack_names, eval_names, distributed, out_dir, eval_num, batch_size)


def collect_results():
    out_dir = '/xxx'
    res_file = os.path.join(out_dir, 'all_evalate_results.json')
    with open(res_file, 'r') as f:
        all_res = json.load(f)
    attack_names = [
        'inception_v3',
        # 'inception_v4',
        'inception_resnet_v2',
        'resnet50',
        # 'vgg16',
        'densenet121',
    ]
    for attackid, attack_name in enumerate(attack_names):
        for opt in ['MI', 'NI']:
            compare_scores = []
            for transform_method in ['PB', 'ALL', ]:
                scores = all_res[attack_name + '-' + opt + '-' + transform_method + '-VTI-FGSM']
                # scores = scores[:-2]  # remove adverasrial trained models
                scores = np.concatenate([scores[:1], scores[2:4], scores[5:]], 0)
                compare_scores.append(scores)
            compare_scores = np.array(compare_scores)

            out = []
            for idx, transform_method in enumerate(['PB', 'SE-PB', ]):
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
    # main()
    collect_results()

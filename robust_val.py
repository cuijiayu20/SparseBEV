"""鲁棒性基准测试评估脚本。

支持三种噪声类型：
  - clean:       无噪声基准
  - drop:        丢帧测试
  - extrinsics:  外参扰动测试
  - occlusion:   遮挡测试

用法示例:
  # 基准测试
  python robust_val.py --config configs/r50_nuimg_704x256.py --weights ckpts/r50_nuimg_704x256.pth --noise-type clean

  # 丢帧测试 (10%)
  python robust_val.py --config configs/r50_nuimg_704x256.py --weights ckpts/r50_nuimg_704x256.pth \
      --noise-type drop --drop-ratio 10 --noise-pkl data/nuscenes/nuscenes_infos_val_with_noise_Drop.pkl

  # 外参扰动测试 (L2, single)
  python robust_val.py --config configs/r50_nuimg_704x256.py --weights ckpts/r50_nuimg_704x256.pth \
      --noise-type extrinsics --extrinsics-level L2 --extrinsics-type single \
      --noise-pkl data/nuscenes/nuscenes_infos_val_with_noise.pkl

  # 遮挡测试 (exp=2.0)
  python robust_val.py --config configs/r50_nuimg_704x256.py --weights ckpts/r50_nuimg_704x256.pth \
      --noise-type occlusion --occlusion-exp 2.0 \
      --noise-pkl data/nuscenes/nuscenes_infos_val_with_noise.pkl \
      --mask-dir robust_benchmark/Mud_Mask_selected
"""

import os
import sys
import json
import utils
import logging
import argparse
import importlib
import torch
import torch.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed, multi_gpu_test, single_gpu_test
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from models.utils import VERSION


def evaluate(dataset, results):
    """评估并返回指标字典。"""
    metrics = dataset.evaluate(results, jsonfile_prefix='submission')

    result = {
        'mAP': metrics['pts_bbox_NuScenes/mAP'],
        'mATE': metrics['pts_bbox_NuScenes/mATE'],
        'mASE': metrics['pts_bbox_NuScenes/mASE'],
        'mAOE': metrics['pts_bbox_NuScenes/mAOE'],
        'mAVE': metrics['pts_bbox_NuScenes/mAVE'],
        'mAAE': metrics['pts_bbox_NuScenes/mAAE'],
        'NDS': metrics['pts_bbox_NuScenes/NDS'],
    }

    logging.info('--- Evaluation Results ---')
    for k, v in result.items():
        logging.info(f'{k}: {v:.4f}')

    return result


def modify_config_for_noise(cfgs, args):
    """根据噪声类型动态修改配置。"""

    if args.noise_type == 'clean':
        # 使用原始配置，无需修改
        logging.info('[Robust] Mode: CLEAN (baseline)')
        return

    elif args.noise_type == 'drop':
        # 丢帧测试：替换数据集类型为 NuScenesNoiseDataset
        logging.info(f'[Robust] Mode: FRAME DROP, ratio={args.drop_ratio}%, type=discrete')
        cfgs.data.val.type = 'NuScenesNoiseDataset'
        cfgs.data.val.noise_nuscenes_ann_file = args.noise_pkl
        cfgs.data.val.drop_frames = True
        cfgs.data.val.drop_ratio = args.drop_ratio
        cfgs.data.val.drop_type = 'discrete'
        cfgs.data.val.extrinsics_noise = False

    elif args.noise_type == 'extrinsics':
        # 外参扰动测试
        logging.info(f'[Robust] Mode: EXTRINSICS NOISE, level={args.extrinsics_level}, type={args.extrinsics_type}')
        cfgs.data.val.type = 'NuScenesNoiseDataset'
        cfgs.data.val.noise_nuscenes_ann_file = args.noise_pkl
        cfgs.data.val.extrinsics_noise = True
        cfgs.data.val.extrinsics_noise_type = args.extrinsics_type
        cfgs.data.val.extrinsics_noise_level = args.extrinsics_level
        cfgs.data.val.drop_frames = False

    elif args.noise_type == 'occlusion':
        # 遮挡测试：替换图像加载 Pipeline
        logging.info(f'[Robust] Mode: OCCLUSION, exp={args.occlusion_exp}')

        # 找到 test_pipeline 中的 LoadMultiViewImageFromFiles 并替换
        new_pipeline = []
        for step in cfgs.data.val.pipeline:
            if step['type'] == 'LoadMultiViewImageFromFiles':
                new_pipeline.append(dict(
                    type='LoadMaskMultiViewImageFromFiles',
                    to_float32=step.get('to_float32', False),
                    color_type=step.get('color_type', 'color'),
                    noise_nuscenes_ann_file=args.noise_pkl,
                    mask_file=args.mask_dir,
                    exp=args.occlusion_exp,
                ))
            else:
                new_pipeline.append(step)
        cfgs.data.val.pipeline = new_pipeline

    else:
        raise ValueError(f'Unknown noise type: {args.noise_type}')


def get_output_filename(args):
    """根据测试类型生成输出文件名。"""
    if args.noise_type == 'clean':
        return 'clean'
    elif args.noise_type == 'drop':
        return f'drop_{args.drop_ratio}'
    elif args.noise_type == 'extrinsics':
        return f'extrinsics_{args.extrinsics_level}_{args.extrinsics_type}'
    elif args.noise_type == 'occlusion':
        return f'occlusion_exp{args.occlusion_exp}'
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(description='Robustness benchmark evaluation')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--weights', required=True, help='Checkpoint file path')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)

    # 噪声参数
    parser.add_argument('--noise-type', type=str, required=True,
                       choices=['clean', 'drop', 'extrinsics', 'occlusion'],
                       help='Noise type: clean/drop/extrinsics/occlusion')
    parser.add_argument('--noise-pkl', type=str, default='',
                       help='Path to noise PKL file')
    parser.add_argument('--drop-ratio', type=int, default=10,
                       help='Frame drop ratio (10-90)')
    parser.add_argument('--extrinsics-level', type=str, default='L1',
                       choices=['L1', 'L2', 'L3', 'L4'],
                       help='Extrinsics noise level')
    parser.add_argument('--extrinsics-type', type=str, default='single',
                       choices=['single', 'all'],
                       help='Extrinsics noise type: single or all cameras')
    parser.add_argument('--occlusion-exp', type=float, default=3.0,
                       help='Occlusion mask exp parameter (1.0/2.0/3.0/5.0)')
    parser.add_argument('--mask-dir', type=str, default='robust_benchmark/Occlusion_mask',
                       help='Path to mud mask directory')
    parser.add_argument('--output-dir', type=str, default='robust_results',
                       help='Directory to save results')
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)

    # register custom modules
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    # you need GPUs
    assert torch.cuda.is_available()

    # determine local_rank and world_size
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if local_rank == 0:
        utils.init_logging(None, cfgs.debug)
    else:
        logging.root.disabled = True

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(local_rank))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        logging.info('Initializing DDP with %d GPUs...' % world_size)
        dist.init_process_group('nccl', init_method='env://')

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    # 根据噪声类型修改配置
    modify_config_for_noise(cfgs, args)

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=False,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()
    model.fp16_enabled = True

    if world_size > 1:
        model = MMDistributedDataParallel(model, [local_rank], broadcast_buffers=False)
    else:
        model = MMDataParallel(model, [0])

    logging.info('Loading checkpoint from %s' % args.weights)
    checkpoint = load_checkpoint(
        model, args.weights, map_location='cuda', strict=True,
        logger=logging.Logger(__name__, logging.ERROR)
    )

    if 'version' in checkpoint:
        VERSION.name = checkpoint['version']

    if world_size > 1:
        results = multi_gpu_test(model, val_loader, gpu_collect=True)
    else:
        results = single_gpu_test(model, val_loader)

    if local_rank == 0:
        metrics = evaluate(val_dataset, results)

        # 保存结果到 JSON
        os.makedirs(args.output_dir, exist_ok=True)
        output_name = get_output_filename(args)
        output_path = os.path.join(args.output_dir, f'{output_name}.json')

        output_data = {
            'noise_type': args.noise_type,
            'config': args.config,
            'weights': args.weights,
            'metrics': metrics,
        }

        # 添加噪声特定信息
        if args.noise_type == 'drop':
            output_data['drop_ratio'] = args.drop_ratio
        elif args.noise_type == 'extrinsics':
            output_data['extrinsics_level'] = args.extrinsics_level
            output_data['extrinsics_type'] = args.extrinsics_type
        elif args.noise_type == 'occlusion':
            output_data['occlusion_exp'] = args.occlusion_exp

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logging.info(f'Results saved to {output_path}')


if __name__ == '__main__':
    main()

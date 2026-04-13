import os
import numpy as np
import mmcv
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from pyquaternion import Quaternion


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):

    def collect_sweeps(self, index, into_past=60, into_future=60):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation,
        )

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict


@DATASETS.register_module()
class NuScenesNoiseDataset(CustomNuScenesDataset):
    """扩展数据集，支持丢帧和外参扰动噪声测试。
    
    复用 robust_benchmark/Toolkit 中定义的噪声数据结构。
    
    Args:
        noise_nuscenes_ann_file (str): 噪声PKL文件路径
        extrinsics_noise (bool): 是否启用外参噪声
        extrinsics_noise_type (str): 'single' 或 'all'
        extrinsics_noise_level (str): 'L1'/'L2'/'L3'/'L4'
        drop_frames (bool): 是否启用丢帧
        drop_ratio (int): 丢帧比例 10-90
        drop_type (str): 'discrete' 或 'consecutive'
    """

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 test_mode=False,
                 use_valid_flag=False,
                 # 噪声参数
                 noise_nuscenes_ann_file='',
                 extrinsics_noise=False,
                 extrinsics_noise_type='single',
                 extrinsics_noise_level='L3',
                 drop_frames=False,
                 drop_ratio=0,
                 drop_type='discrete',
                 **kwargs):
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            test_mode=test_mode,
            use_valid_flag=use_valid_flag,
            **kwargs)

        self.extrinsics_noise = extrinsics_noise
        assert extrinsics_noise_type in ['all', 'single']
        self.extrinsics_noise_type = extrinsics_noise_type
        self.extrinsics_noise_level = extrinsics_noise_level
        self.drop_frames = drop_frames
        self.drop_ratio = drop_ratio
        self.drop_type = drop_type

        if self.extrinsics_noise or self.drop_frames:
            assert noise_nuscenes_ann_file != '', \
                'noise_nuscenes_ann_file must be provided when noise is enabled'
            noise_data = mmcv.load(noise_nuscenes_ann_file, file_format='pkl')
            self.noise_camera_data = noise_data.get('camera', {})
            self.noise_lidar_data = noise_data.get('lidar', {})
        else:
            self.noise_camera_data = {}
            self.noise_lidar_data = {}

        print('[NuScenesNoiseDataset] Noise settings:')
        if self.drop_frames:
            print(f'  Frame drop: ratio={self.drop_ratio}%, type={self.drop_type}')
        if self.extrinsics_noise:
            print(f'  Extrinsics noise: type={self.extrinsics_noise_type}, level={self.extrinsics_noise_level}')
        if not self.drop_frames and not self.extrinsics_noise:
            print('  Clean (no noise)')

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []

            for cam_type, cam_info in info['cams'].items():
                cam_data_path = cam_info['data_path']
                file_name = os.path.basename(cam_data_path)

                # --- 丢帧处理 ---
                if self.drop_frames and file_name in self.noise_camera_data:
                    noise_info = self.noise_camera_data[file_name]['noise']
                    drop_info = noise_info.get('drop_frames', {})
                    ratio_info = drop_info.get(self.drop_ratio, {})
                    type_info = ratio_info.get(self.drop_type, {})
                    
                    if type_info.get('stuck', False):
                        replace_file = type_info.get('replace', '')
                        if replace_file != '':
                            cam_data_path = cam_data_path.replace(file_name, replace_file)

                img_paths.append(os.path.relpath(cam_data_path))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # --- 外参扰动处理 ---
                if self.extrinsics_noise and file_name in self.noise_camera_data:
                    noise_ext = self.noise_camera_data[file_name]['noise'].get('extrinsics_noise', {})
                    
                    # 构建噪声键名：{level}_{type}_noise_sensor2lidar_rotation
                    level = self.extrinsics_noise_level
                    ntype = self.extrinsics_noise_type
                    rot_key = f'{level}_{ntype}_noise_sensor2lidar_rotation'
                    trans_key = f'{level}_{ntype}_noise_sensor2lidar_translation'
                    
                    # 若找不到带级别前缀的键，回退到旧格式 (L3兼容)
                    if rot_key not in noise_ext:
                        rot_key = f'{ntype}_noise_sensor2lidar_rotation'
                        trans_key = f'{ntype}_noise_sensor2lidar_translation'
                    
                    sensor2lidar_rotation = noise_ext[rot_key]
                    sensor2lidar_translation = noise_ext[trans_key]
                else:
                    sensor2lidar_rotation = cam_info['sensor2lidar_rotation']
                    sensor2lidar_translation = cam_info['sensor2lidar_translation']

                # 计算 lidar2img 矩阵
                lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
                lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

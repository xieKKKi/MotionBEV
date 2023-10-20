#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
from torch.utils import data


class SemKITTI(data.Dataset):
    def __init__(self, data_config_path, data_path, imageset='train', return_ref=False, residual=1,
                 residual_path=None, drop_few_static_frames=True):
        self.return_ref = return_ref
        with open(data_config_path, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            self.split = semkittiyaml['split']['train']
        elif imageset == 'val':
            self.split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            self.split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.scan_files = {}
        self.residual = residual
        self.residual_files = {}
        for seq in self.split:
            seq = '{0:02d}'.format(int(seq))
            scan_files = []
            scan_files += absoluteFilePaths('/'.join([data_path, str(seq).zfill(2), 'velodyne']))
            scan_files.sort()
            self.scan_files[seq] = scan_files
            if self.residual > 0:
                residual_files = []
                residual_files += absoluteFilePaths('/'.join(
                    [residual_path, str(seq).zfill(2), 'residual_images']))  # residual_images_4  residual_images
                residual_files.sort()
                self.residual_files[seq] = residual_files

        if imageset == 'train' and drop_few_static_frames:
            self.remove_few_static_frames()

        scan_files = []
        residual_files = []
        for seq in self.split:
            seq = '{0:02d}'.format(int(seq))
            scan_files += self.scan_files[seq]
            if self.residual > 0:
                residual_files += self.residual_files[seq]
        self.scan_files = scan_files
        if self.residual > 0:
            self.residual_files = residual_files

    def remove_few_static_frames(self):
        # Developed by Jiadai Sun 2021-11-07
        # This function is used to clear some frames, because too many static frames will lead to a long training time

        remove_mapping_path = "config/train_split_dynamic_pointnumber.txt"
        with open(remove_mapping_path) as fd:
            lines = fd.readlines()
            lines = [line.strip() for line in lines]

        pending_dict = {}  # 加载到dict中
        for line in lines:
            if line != '':
                seq, fid, _ = line.split()
                if int(seq) in self.split:
                    if seq in pending_dict.keys():
                        if fid in pending_dict[seq]:
                            raise ValueError(f"!!!! Duplicate {fid} in seq {seq} in .txt file")
                        pending_dict[seq].append(fid)
                    else:
                        pending_dict[seq] = [fid]

        total_raw_len = 0
        total_new_len = 0
        for seq in self.split:
            seq = '{0:02d}'.format(int(seq))
            if seq in pending_dict.keys():
                raw_len = len(self.scan_files[seq])

                # lidar scan files
                scan_files = self.scan_files[seq]
                useful_scan_paths = [path for path in scan_files if os.path.split(path)[-1][:-4] in pending_dict[seq]]
                self.scan_files[seq] = useful_scan_paths

                if self.residual:
                    residual_files = self.residual_files[seq]
                    useful_residual_paths = [path for path in residual_files if
                                             os.path.split(path)[-1][:-4] in pending_dict[seq]]
                    self.residual_files[seq] = useful_residual_paths
                    assert (len(useful_scan_paths) == len(useful_residual_paths))
                new_len = len(self.scan_files[seq])
                print(f"Seq {seq} drop {raw_len - new_len}: {raw_len} -> {new_len}")
                total_raw_len += raw_len
                total_new_len += new_len
        print(f"Totally drop {total_raw_len - total_new_len}: {total_raw_len} -> {total_new_len}")

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.scan_files)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.scan_files[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.scan_files[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)

        if self.residual > 0:
            residual_data = np.load(self.residual_files[index])
            data_tuple += (residual_data,)  # (x y z), label, ref, residual_n

        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class voxel_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, return_test=False,
                 fixed_volume_space=True, max_volume_space=[50, 50, 1.5], min_volume_space=[-50, -50, -3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.ignore_label = 0
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        max_bound = np.percentile(xyz, 100, axis=0)
        min_bound = np.percentile(xyz, 0, axis=0)

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


class spherical_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size,
                 rotate_aug=False, flip_aug=False, transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 return_test=False,
                 fixed_volume_space=True,
                 max_volume_space=[50.15, np.pi, 2], min_volume_space=[1.85, -np.pi, -4],
                 is_livox=False):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.transform = transform_aug
        self.trans_std = trans_std
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        # max_volume_space = [50.15, np.pi, 2]  # 最里面一格和最外面用来收集ignore的点，防止与不ignore的点放在一起
        # min_volume_space = [1.85, -np.pi, -4]
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.is_livox = is_livox

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:  # with reflectivity
            xyz, labels, sig = data
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)  # 删除单维度
        elif len(data) == 4:  # with residual
            xyz, labels, sig, residual = data
        else:
            raise Exception('Return invalid data tuple')

        # 因为只变化了坐标没有变索引，所以标签不需要变
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            if self.is_livox:
                flip_type = np.random.choice(2, 1)
                if flip_type == 1:
                    xyz[:, 1] = -xyz[:, 1]
            else:
                flip_type = np.random.choice(4, 1)
                if flip_type == 1:
                    xyz[:, 0] = -xyz[:, 0]
                elif flip_type == 2:
                    xyz[:, 1] = -xyz[:, 1]
                elif flip_type == 3:
                    xyz[:, :2] = -xyz[:, :2]
        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),  # [0.1, 0.1, 0.1]
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        else:
            max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
            min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
            max_bound = np.max(xyz_pol[:, 1:], axis=0)
            min_bound = np.min(xyz_pol[:, 1:], axis=0)
            max_bound = np.concatenate(([max_bound_r], max_bound))
            min_bound = np.concatenate(([min_bound_r], min_bound))

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any():
            print("Zero interval!")
        # 得到每一个点对应的voxel的索引[rho_idx, theta_yaw, pitch_idx]
        # Clip (limit) the values in an array.
        # np.floor向下取整
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process labels
        self.ignore_label = 255
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)  # 每一个点对应的格子和label
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]  # 按照pitch yaw rho顺序从小到大排序
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        # center data on each voxel for PTnet
        # 这里是每一个点所处的voxel中心的位置，在同一个voxel的点的位置是一样的
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        # 每一个点的位置相对于其voxel中心的偏移量
        return_xyz = xyz_pol - voxel_centers
        # [bias_rho, bias_yaw, bias_pitch, rho, yaw, pitch, x, y, z]
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)
        elif len(data) == 4:  # reflectivity residual
            # [bias_rho, bias_theta, bias_z, rho, theta, z, x, y, reflectivity, residual(1-?)]
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis], residual), axis=1)
        else:
            raise NotImplementedError

        if self.return_test:
            return torch.from_numpy(processed_label).type(torch.LongTensor), \
                   torch.from_numpy(grid_ind), \
                   labels, \
                   torch.from_numpy(return_fea).type(torch.FloatTensor), index
        else:
            return torch.from_numpy(processed_label).type(torch.LongTensor), \
                   torch.from_numpy(grid_ind), \
                   labels, \
                   torch.from_numpy(return_fea).type(torch.FloatTensor)


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):  # 每个栅格赋予出现次数最多的label
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)  # counter计算每个label的数量
    counter[sorted_label_voxel_pair[0, 3]] = 1  # 第一个初始化，先加一
    cur_sear_ind = sorted_label_voxel_pair[0, :3]  # 目标点的栅格坐标索引
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]  # 当前点的栅格坐标索引
        if not np.all(np.equal(cur_ind, cur_sear_ind)):  # 索引不一致，要移动到下一个栅格
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)  # 栅格使用出现次数最多的label
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1  # label计数
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label_dynamic(processed_label, sorted_label_voxel_pair):  # 每个栅格赋予出现次数最多的label, 超过3个点视为dynamic
    label_size = 2
    counter = np.zeros((label_size,), dtype=np.uint16)  # counter计算每个label的数量
    counter[sorted_label_voxel_pair[0, 3]] = 1  # 第一个初始化，先加一
    cur_sear_ind = sorted_label_voxel_pair[0, :3]  # 目标点的栅格坐标索引
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]  # 当前点的栅格坐标索引
        if not np.all(np.equal(cur_ind, cur_sear_ind)):  # 索引不一致，要移动到下一个栅格
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)  # 栅格使用出现次数最多的label
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1  # label计数
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def collate_fn_BEV(data):
    label2stack = torch.stack([d[0] for d in data])
    grid_ind_stack = [d[1] for d in data]
    point_label = [d[2] for d in data]
    xyz = [d[3] for d in data]
    return label2stack, grid_ind_stack, point_label, xyz


def collate_fn_BEV_test(data):
    label2stack = torch.stack([d[0] for d in data])
    grid_ind_stack = [d[1] for d in data]
    point_label = [d[2] for d in data]
    xyz = [d[3] for d in data]
    index = [d[4] for d in data]
    return label2stack, grid_ind_stack, point_label, xyz, index


# load Semantic KITTI class info
def get_SemKITTI_label_name(label_mapping):  #
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    inv_learning_map = semkittiyaml['learning_map_inv']
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        map_i = semkittiyaml['learning_map'][i]
        map_inv_i = semkittiyaml['learning_map_inv'][map_i]
        SemKITTI_label_name[map_i] = semkittiyaml['labels'][map_inv_i]

    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[:]
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]
    return unique_label, unique_label_str, inv_learning_map

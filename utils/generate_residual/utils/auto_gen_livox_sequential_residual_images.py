#!/usr/bin/env python3
# Developed by Jiapeng Xie
# Brief: This script generates residual images

import os
import random

os.environ["OMP_NUM_THREADS"] = "16"
import yaml
import numpy as np

from tqdm import tqdm
from icecream import ic
from kitti_utils import load_poses, load_calib, load_files, load_vertex
from kitti_utils import polar_projection_livox
from queue import Queue


def check_and_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_yaml(path):
    if yaml.__version__ >= '5.1':
        config = yaml.load(open(path), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(path))
    return config


def process_one_seq(config):
    # specify parameters
    num_frames = config['num_frames']
    occlusion_block = config['occlusion_block']
    num_prev_n = config['num_prev_n']
    num_last_n = config['num_last_n']

    # specify the output folders
    residual_image_folder = config['residual_image_folder']
    check_and_makedirs(residual_image_folder)

    # load poses
    pose_file = config['pose_file']
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    calib_file = config['calib_file']
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)

    # load LiDAR scans
    scan_folder = config['scan_folder']
    scan_paths = load_files(scan_folder)

    # test for the first N scans
    if num_frames >= len(poses) or num_frames <= 0:
        print('generate training data for all frames with number of: ', len(poses))
    else:
        poses = poses[:num_frames]
        scan_paths = scan_paths[:num_frames]

    polar_image_params = config['polar_image']

    prev_len_Que = Queue()
    last_len_Que = Queue()
    prev_sub_map = None
    last_sub_map = None
    prev_diff_map = None
    last_diff_map = None
    prev_index = None
    last_index = None
    # generate residual images for the whole sequence
    for frame_idx in tqdm(range(len(scan_paths))):
        if frame_idx < num_prev_n + num_last_n - 1:
            continue
        else:
            # load current scan
            current_scan = load_vertex(scan_paths[frame_idx])
            # current_scan = current_scan[:random.randint(100, 200)]  # down sample for debug
            current_pose = poses[frame_idx]
            if frame_idx == num_prev_n + num_last_n - 1:  # initialize
                for i in range(-num_last_n - num_prev_n + 1, -num_last_n + 1):
                    prev_pose = poses[frame_idx + i]
                    prev_scan = load_vertex(scan_paths[frame_idx + i])
                    # prev_scan = prev_scan[:random.randint(100, 200)]  # down sample for debug
                    prev_scan_transformed = np.linalg.inv(current_pose).dot(prev_pose).dot(prev_scan.T).T
                    if prev_sub_map is None:
                        prev_sub_map = prev_scan_transformed
                        prev_diff_map = np.full((prev_scan.shape[0], num_prev_n + num_last_n), 0, dtype=np.float32)
                        prev_index = np.full(prev_scan.shape[0], num_last_n, dtype=int)
                    else:
                        prev_sub_map = np.concatenate((prev_sub_map, prev_scan_transformed), axis=0)
                        prev_diff_map = np.concatenate(
                            (prev_diff_map, np.full((prev_scan.shape[0], num_prev_n + num_last_n), 0,
                                                    dtype=np.float32)), axis=0)
                        prev_index = prev_index + 1
                        prev_index = np.concatenate((prev_index, np.full(prev_scan.shape[0], num_last_n, dtype=int)),
                                                    axis=0)
                    prev_len_Que.put(prev_scan.shape[0])
                for i in range(-num_last_n + 1, 1):
                    last_pose = poses[frame_idx + i]
                    last_scan = load_vertex(scan_paths[frame_idx + i])  # (x, y, z, 1)
                    # last_scan = last_scan[:random.randint(100, 200)]  # down sample for debug
                    last_scan_transformed = np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T
                    if last_sub_map is None:
                        last_sub_map = last_scan_transformed
                        last_diff_map = np.full((last_scan.shape[0], num_prev_n + num_last_n), 0, dtype=np.float32)
                        last_index = np.full(last_scan.shape[0], 0, dtype=int)
                    else:
                        last_sub_map = np.concatenate((last_sub_map, last_scan_transformed), axis=0)
                        last_diff_map = np.concatenate(
                            (last_diff_map, np.full((last_scan.shape[0], num_prev_n + num_last_n), 0,
                                                    dtype=np.float32)), axis=0)
                        last_index = last_index + 1
                        last_index = np.concatenate((last_index, np.full(last_scan.shape[0], 0, dtype=int)), axis=0)
                    last_len_Que.put(last_scan.shape[0])
            else:
                last_pose = poses[frame_idx - 1]
                # transform to current coordinate
                prev_sub_map = np.linalg.inv(current_pose).dot(last_pose).dot(prev_sub_map.T).T
                last_sub_map = np.linalg.inv(current_pose).dot(last_pose).dot(last_sub_map.T).T
                last_sub_map = np.concatenate((last_sub_map, current_scan), axis=0)
                last_diff_map = np.concatenate(
                    (last_diff_map, np.full((current_scan.shape[0], num_prev_n + num_last_n), 0,
                                            dtype=np.float32)), axis=0)
                last_len_Que.put(current_scan.shape[0])
                prev_index = (prev_index + 1)  # % num_prev_n + num_last_n
                last_index = (last_index + 1)  # % num_prev_n + num_last_n
                last_index = np.concatenate((last_index, np.full(current_scan.shape[0], 0, dtype=int)), axis=0)

            # print(np.max(prev_index), np.max(last_index), np.min(prev_index), np.min(last_index))
            prev_proj_z_delta, prev_mask_valid, prev_proj_y, prev_proj_x, prev_occlusion_mask = \
                polar_projection_livox(current_vertex=prev_sub_map.astype(np.float32),
                                       proj_H=polar_image_params['height'],
                                       proj_W=polar_image_params['width'],
                                       max_range=polar_image_params['max_range'],
                                       min_range=polar_image_params['min_range'],
                                       yaw_limit=polar_image_params['fov_h_half'],
                                       max_z=polar_image_params['max_z'],
                                       min_z=polar_image_params['min_z'],
                                       return_occlusion=config['occlusion_block'])

            last_proj_z_delta, last_mask_valid, last_proj_y, last_proj_x, last_occlusion_mask = \
                polar_projection_livox(current_vertex=last_sub_map.astype(np.float32),
                                       proj_H=polar_image_params['height'],
                                       proj_W=polar_image_params['width'],
                                       max_range=polar_image_params['max_range'],
                                       min_range=polar_image_params['min_range'],
                                       yaw_limit=polar_image_params['fov_h_half'],
                                       max_z=polar_image_params['max_z'],
                                       min_z=polar_image_params['min_z'],
                                       return_occlusion=config['occlusion_block'])

            # generate residual image
            residual_proj_z_delta = prev_proj_z_delta - last_proj_z_delta
            residual_proj_z_delta[0.2 * prev_proj_z_delta < last_proj_z_delta] = 0
            residual_proj_z_delta[prev_proj_z_delta < 0.4] = 0
            residual_proj_z_delta[prev_proj_z_delta > 3] = 0
            if config['occlusion_block']:
                residual_proj_z_delta[last_occlusion_mask] = 0  # residual_proj_z_delta[last_proj_z_delta == 0] = 0
            prev_diff_map[prev_mask_valid, prev_index[prev_mask_valid]] += residual_proj_z_delta[
                prev_proj_y, prev_proj_x]

            residual_proj_z_delta = last_proj_z_delta - prev_proj_z_delta
            residual_proj_z_delta[0.2 * last_proj_z_delta < prev_proj_z_delta] = 0
            residual_proj_z_delta[last_proj_z_delta < 0.4] = 0
            residual_proj_z_delta[last_proj_z_delta > 3] = 0
            if occlusion_block:
                residual_proj_z_delta[prev_occlusion_mask] = 0  # residual_proj_z_delta[prev_proj_z_delta == 0] = 0
            last_diff_map[last_mask_valid, last_index[last_mask_valid]] += residual_proj_z_delta[
                last_proj_y, last_proj_x]

            prev_scan_size = prev_len_Que.get()
            prev_sub_map = prev_sub_map[prev_scan_size:]
            diff_scan = prev_diff_map[:prev_scan_size]
            prev_diff_map = prev_diff_map[prev_scan_size:]
            prev_index = prev_index[prev_scan_size:]

            last_scan_size = last_len_Que.get()
            prev_len_Que.put(last_scan_size)
            prev_sub_map = np.concatenate((prev_sub_map, last_sub_map[:last_scan_size]), axis=0)
            last_sub_map = last_sub_map[last_scan_size:]
            prev_diff_map = np.concatenate((prev_diff_map, last_diff_map[:last_scan_size]), axis=0)
            last_diff_map = last_diff_map[last_scan_size:]
            prev_index = np.concatenate((prev_index, last_index[:last_scan_size]), axis=0)
            last_index = last_index[last_scan_size:]

            # save
            file_name = os.path.join(residual_image_folder, str(frame_idx - num_last_n - num_prev_n + 1).zfill(6))
            np.save(file_name, diff_scan)

    print("Saving last few files...")
    for frame_idx in range(len(scan_paths) - num_last_n - num_prev_n + 1, len(scan_paths)):
        if not prev_len_Que.empty():
            prev_scan_size = prev_len_Que.get()
            diff_scan = prev_diff_map[:prev_scan_size]
            prev_diff_map = prev_diff_map[prev_scan_size:]
        else:
            prev_scan_size = last_len_Que.get()
            diff_scan = last_diff_map[:prev_scan_size]
            last_diff_map = last_diff_map[prev_scan_size:]

        file_name = os.path.join(residual_image_folder, str(frame_idx).zfill(6))
        np.save(file_name, diff_scan)
    print("Done.")


if __name__ == '__main__':
    # load config file
    config_filename = '../config/data_preparing_livox_sequential.yaml'
    config = load_yaml(config_filename)

    scan_folder = config['scan_folder']
    pose_file = config['scan_folder']
    calib_file = config['scan_folder']
    residual_image_folder = config['residual_image_folder']
    num_prev_n = config['num_prev_n']
    num_last_n = config['num_last_n']

    for seq in range(0, 8):  # sequences id
        # Update the value in config to facilitate the iterative loop
        config['scan_folder'] = scan_folder + f"sequences/{'%02d' % seq}/velodyne"
        config['pose_file'] = pose_file + f"sequences/{'%02d' % seq}/poses.txt"
        config['calib_file'] = calib_file + f"sequences/{'%02d' % seq}/calib.txt"
        config['residual_image_folder'] = residual_image_folder + f"{'%02d' % seq}/residual_images"
        ic(config)
        process_one_seq(config)

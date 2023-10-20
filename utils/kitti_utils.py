#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# This file is covered by the LICENSE file in the root of this project.
# Brief: some utilities

import os
import math
import numpy as np
import numba as nb
from scipy.spatial.transform import Rotation as R

np.random.seed(0)


def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
        Args:
            pose_path: (Complete) filename for the pose file
        Returns:
            A numpy array of size nx4x4 with n poses as 4x4 transformation
            matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']

    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')

    return np.array(poses)


def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)


def range_projection(current_vertex, proj_H=64, proj_W=2048, fov_up=3.0, fov_down=-25.0, max_range=50, min_range=2):
    """ Project a pointcloud into a spherical projection, range image.
        Args:
            current_vertex: raw point clouds
        Returns:
            # proj_vertex: each pixel contains the corresponding point (x, y, z, depth)
            proj_idx, proj_range
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth_ori = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    mask = (depth_ori > min_range) & (depth_ori < max_range)
    current_vertex = current_vertex[mask]  # get rid of [0, 0, 0] points
    depth = depth_ori[mask]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    intensity = current_vertex[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)  # 向下取整
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    proj_x_orig = np.copy(proj_x)

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    proj_y_orig = np.copy(proj_y)

    # order in decreasing depth
    order = np.argsort(depth)[::-1]  # 排序并提取索引，深度较大值会被较小值覆盖
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices_ori = np.arange(depth_ori.shape[0])
    indices = indices_ori[mask]
    # indices = np.arange(depth_ori.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), 0, dtype=np.float32)  # [H,W] range (-1 is no data)
    # proj_vertex = np.full((proj_H, proj_W, 4), 0, dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1, dtype=np.int32)  # [H,W] index (-1 is no data)
    # proj_intensity = np.full((proj_H, proj_W), 0, dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    # proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, depth]).T
    proj_idx[proj_y, proj_x] = indices
    # proj_intensity[proj_y, proj_x] = intensity

    return proj_idx, proj_range


def polar_projection(current_vertex, proj_H=360, proj_W=480, max_range=50.0, min_range=2.0, max_z=2.0, min_z=-4.0,
                     return_occlusion=False):
    # get scan components
    rho = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    yaw = np.arctan2(current_vertex[:, 1], current_vertex[:, 0])
    scan_z = current_vertex[:, 2]

    mask_valid = (rho > min_range) & (rho < max_range) & (scan_z < max_z) & (scan_z > min_z)
    rho_valid = rho[mask_valid]
    yaw_valid = yaw[mask_valid]
    z_valid = scan_z[mask_valid]

    # get projections in image coords
    proj_x = 1.0 - (rho_valid - min_range) / (max_range - min_range)  # in [0.0, 1.0]
    proj_y = 0.5 * (yaw_valid / np.pi + 1.0)  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)  # 向下取整
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    order_max = np.argsort(z_valid)  # 排序并提取索引，z较大值会被较小值覆盖
    scan_z_max = z_valid[order_max]
    proj_y_max = proj_y[order_max]
    proj_x_max = proj_x[order_max]
    proj_z_max = np.full((proj_H, proj_W), 0, dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_z_max[proj_y_max, proj_x_max] = scan_z_max

    order_min = order_max[::-1]  # z较小值会被较大值覆盖
    scan_z_min = z_valid[order_min]
    proj_y_min = proj_y[order_min]
    proj_x_min = proj_x[order_min]
    proj_z_min = np.full((proj_H, proj_W), 0, dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_z_min[proj_y_min, proj_x_min] = scan_z_min

    proj_z_delta = np.abs(proj_z_max - proj_z_min)

    occlusion_mask = np.full(proj_z_delta.shape, False, dtype=bool)
    if return_occlusion:
        get_occlusion_mask(proj_z_delta, occlusion_mask)

    return proj_z_delta, mask_valid, proj_y, proj_x, occlusion_mask


def polar_projection_livox(current_vertex, proj_H=360, proj_W=480, max_range=50.0, min_range=2.0, yaw_limit=np.pi,
                           max_z=2.0, min_z=-4.0, return_occlusion=False):
    # get scan components
    rho = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    yaw = np.arctan2(current_vertex[:, 1], current_vertex[:, 0])
    scan_z = current_vertex[:, 2]

    mask_valid = (rho > min_range) & (rho < max_range) & \
                 (scan_z < max_z) & (scan_z > min_z) & \
                 (yaw < yaw_limit) & (yaw > -yaw_limit)
    rho_valid = rho[mask_valid]
    yaw_valid = yaw[mask_valid]
    z_valid = scan_z[mask_valid]

    # get projections in image coords
    proj_x = 1.0 - (rho_valid - min_range) / (max_range - min_range)  # in [0.0, 1.0]
    proj_y = 0.5 * (yaw_valid / yaw_limit + 1.0)  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)  # 向下取整
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    order_max = np.argsort(z_valid)  # 排序并提取索引，z较大值会被较小值覆盖
    scan_z_max = z_valid[order_max]
    proj_y_max = proj_y[order_max]
    proj_x_max = proj_x[order_max]
    proj_z_max = np.full((proj_H, proj_W), 0, dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_z_max[proj_y_max, proj_x_max] = scan_z_max

    order_min = order_max[::-1]  # z较小值会被较大值覆盖
    scan_z_min = z_valid[order_min]
    proj_y_min = proj_y[order_min]
    proj_x_min = proj_x[order_min]
    proj_z_min = np.full((proj_H, proj_W), 0, dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_z_min[proj_y_min, proj_x_min] = scan_z_min

    proj_z_delta = np.abs(proj_z_max - proj_z_min)

    occlusion_mask = np.full(proj_z_delta.shape, False, dtype=bool)
    if return_occlusion:
        get_occlusion_mask(proj_z_delta, occlusion_mask)

    return proj_z_delta, mask_valid, proj_y, proj_x, occlusion_mask


@nb.jit('(float32[:,:],bool_[:,:])', nopython=True, cache=True, parallel=False)
def get_occlusion_mask(proj_arr, occlusion_mask):
    for y in range(proj_arr.shape[0]):  # 遍历每一个yaw
        # for x in range(proj_arr.shape[1] - 1, -1, -1):  # 从后往前遍历rho，把后几位为零的位置mask置为true，表示被遮挡
        for x in range(proj_arr.shape[1]):  # 从远往近遍历rho，把后几位为零的位置mask置为true，表示被遮挡。这里本来顺序就是反的
            if proj_arr[y, x] == 0:
                occlusion_mask[y, x] = True
            else:
                break


def cart_projection(current_vertex, proj_H=480, proj_W=480,
                    max_x=50.0, min_x=-50.0,
                    max_y=50.0, min_y=-50.0,
                    max_z=2.0, min_z=-4.0):
    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    mask_valid = (scan_x < max_x) & (scan_x > min_x) & \
                 (scan_y < max_y) & (scan_y > min_y) & \
                 (scan_z < max_z) & (scan_z > min_z)
    x_valid = scan_x[mask_valid]
    y_valid = scan_y[mask_valid]
    z_valid = scan_z[mask_valid]

    # get projections in image coords
    proj_x = 1.0 - (x_valid - min_x) / (max_x - min_x)  # in [0.0, 1.0]
    proj_y = 1.0 - (y_valid - min_y) / (max_y - min_y)  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)  # 向下取整
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    order_max = np.argsort(z_valid)  # 排序并提取索引，z较大值会被较小值覆盖
    scan_z_max = z_valid[order_max]
    proj_y_max = proj_y[order_max]
    proj_x_max = proj_x[order_max]
    proj_z_max = np.full((proj_H, proj_W), 0, dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_z_max[proj_y_max, proj_x_max] = scan_z_max

    order_min = order_max[::-1]  # z较小值会被较大值覆盖
    scan_z_min = z_valid[order_min]
    proj_y_min = proj_y[order_min]
    proj_x_min = proj_x[order_min]
    proj_z_min = np.full((proj_H, proj_W), 0, dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_z_min[proj_y_min, proj_x_min] = scan_z_min

    proj_z_delta = np.abs(proj_z_max - proj_z_min)

    return proj_z_delta, mask_valid, proj_y, proj_x


def sphere_projection(current_vertex, voxel_shape=None,
                      fov_up=3.0, fov_down=-25.0,
                      max_range=50, min_range=2,
                      fov_left=-180, fov_right=180):
    # laser parameters
    if voxel_shape is None:
        voxel_shape = [480, 360, 64]
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov_left = fov_left / 180.0 * np.pi
    fov_right = fov_right / 180.0 * np.pi
    max_bound = np.asarray([max_range, fov_right, fov_up])
    min_bound = np.asarray([min_range, fov_left, fov_down])

    rho = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    yaw = np.arctan2(current_vertex[:, 1], current_vertex[:, 0])
    pitch = np.arcsin(current_vertex[:, 2] / rho)

    mask_valid = (pitch > fov_down) & (pitch < fov_up) & (rho > min_range) & (rho < max_range)

    current_vertex_valid = current_vertex[mask_valid]
    rho_valid = np.linalg.norm(current_vertex_valid[:, :3], 2, axis=1)
    yaw_valid = np.arctan2(current_vertex_valid[:, 1], current_vertex_valid[:, 0])
    pitch_valid = np.arcsin(current_vertex_valid[:, 2] / rho_valid)

    xyz_pol_valid = np.stack((rho_valid, yaw_valid, pitch_valid), axis=1)

    crop_range = max_bound - min_bound
    intervals = crop_range / (np.asarray(voxel_shape) - 1)  # 每一格的宽度

    if (intervals == 0).any():
        print("Zero interval!")
    # 得到每一个点对应的voxel的索引[rho_idx, theta_yaw, pitch_idx]
    # Clip (limit) the values in an array.
    # np.floor向下取整
    grid_ind_valid = (np.floor((np.clip(xyz_pol_valid, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

    # 排序
    # grid_ind_valid = grid_ind_valid[np.lexsort((grid_ind_valid[:, 2], grid_ind_valid[:, 1], grid_ind_valid[:, 0]))]

    '''
    return_index：如果为true，返回新列表元素在旧列表中的位置（下标），并以列表形式存储。
    return_inverse：如果为true，返回旧列表元素在新列表中的位置（下标），并以列表形式存储。
    return_counts：如果为true，返回去重数组中的元素在原数组中的出现次数。'''
    unique_grid_ind, index, inverse, counts = np.unique(grid_ind_valid, return_index=True, return_inverse=True,
                                                        return_counts=True, axis=0)
    # 每个点对应的voxel有几个点
    # point_ind_count_valid = counts[inverse]
    # point_ind_count = np.zeros(rho.shape, dtype=np.int)
    # point_ind_count[mask_valid] = point_ind_count_valid

    # 每个voxel里有几个点
    voxel_count = np.zeros(voxel_shape, dtype=np.float32)
    x = grid_ind_valid[index]
    voxel_count[x.T[0], x.T[1], x.T[2]] = counts

    # point_ind_count_valid2 = voxel_count[x.T[0], x.T[1], x.T[2]][inverse]

    return mask_valid, grid_ind_valid, voxel_count, index, inverse


def gen_normal_map(current_range, current_vertex, proj_H=64, proj_W=900):
    """ Generate a normal image given the range projection of a point cloud.
        Args:
            current_range:  range projection of a point cloud, each pixel contains the corresponding depth
            current_vertex: range projection of a point cloud,
                                            each pixel contains the corresponding point (x, y, z, 1)
        Returns:
            normal_data: each pixel contains the corresponding normal
    """
    normal_data = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)

    # iterate over all pixels in the range image
    for x in range(proj_W):
        for y in range(proj_H - 1):
            p = current_vertex[y, x][:3]
            depth = current_range[y, x]

            if depth > 0:
                wrap_x = wrap(x + 1, proj_W)
                u = current_vertex[y, wrap_x][:3]
                u_depth = current_range[y, wrap_x]
                if u_depth <= 0:
                    continue

                v = current_vertex[y + 1, x][:3]
                v_depth = current_range[y + 1, x]
                if v_depth <= 0:
                    continue

                u_norm = (u - p) / np.linalg.norm(u - p)
                v_norm = (v - p) / np.linalg.norm(v - p)

                w = np.cross(v_norm, u_norm)
                norm = np.linalg.norm(w)
                if norm > 0:
                    normal = w / norm
                    normal_data[y, x] = normal

    return normal_data


def wrap(x, dim):
    """ Wrap the boarder of the range image.
    """
    value = x
    if value >= dim:
        value = (value - dim)
    if value < 0:
        value = (value + dim)
    return value


def euler_angles_from_rotation_matrix(R):
    """ From the paper by Gregory G. Slabaugh, Computing Euler angles from a rotation matrix,
        psi, theta, phi = roll pitch yaw (x, y, z).
        Args:
            R: rotation matrix, a 3x3 numpy array
        Returns:
            a tuple with the 3 values psi, theta, phi in radians
    """

    def isclose(x, y, rtol=1.e-5, atol=1.e-8):
        return abs(x - y) <= atol + rtol * abs(y)

    phi = 0.0
    if isclose(R[2, 0], -1.0):
        theta = math.pi / 2.0
        psi = math.atan2(R[0, 1], R[0, 2])
    elif isclose(R[2, 0], 1.0):
        theta = -math.pi / 2.0
        psi = math.atan2(-R[0, 1], -R[0, 2])
    else:
        theta = -math.asin(R[2, 0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
        phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
    return psi, theta, phi


def load_vertex(scan_path):
    """ Load 3D points of a scan. The fileformat is the .bin format used in
        the KITTI dataset.
        Args:
            scan_path: the (full) filename of the scan file
        Returns:
            A nx4 numpy array of homogeneous points (x, y, z, 1).
    """
    current_vertex = np.fromfile(scan_path, dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 4))  # x,y,z,intesity
    current_points = current_vertex[:, 0:3]
    current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
    current_vertex[:, :-1] = current_points
    return current_vertex


def load_vertex_intensity(scan_path):
    current_vertex_raw = np.fromfile(scan_path, dtype=np.float32)
    current_vertex_raw = current_vertex_raw.reshape((-1, 4))  # x,y,z,intesity
    current_points = current_vertex_raw[:, 0:3]
    current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
    current_vertex[:, :-1] = current_points
    return current_vertex, current_vertex_raw


def load_files(folder):
    """ Load all files in a folder and sort.
    """
    file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(folder)) for f in fn]
    file_paths.sort()
    return file_paths


def load_labels(label_path):
    """ Load semantic and instance labels in SemanticKitti format.
    """
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half

    # sanity check
    assert ((sem_label + (inst_label << 16) == label).all())

    return sem_label, inst_label


def rotation_matrix_from_euler_angles(yaw, degrees=True):
    """ Generate rotation matrix given yaw angle.
        Args:
            yaw: yaw angle
        Returns:
            rotation matrix
    """
    return R.from_euler('z', yaw, degrees=degrees).as_matrix()


def gen_transformation(yaw, translation):
    """ Generate transformation from given yaw angle and translation.
        Args:
            current_range: range image
            current_vertex: point clouds
        Returns:
            normal image
    """
    rotation = R.from_euler('zyx', [[yaw, 0, 0]], degrees=True)
    rotation = rotation.as_dcm()[0]
    transformation = np.identity(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = [translation[0], translation[1], translation[2]]

    return transformation

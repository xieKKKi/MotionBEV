#!/usr/bin/env python3
# @author: Jiapeng Xie

import numpy as np
import argparse
import open3d as o3d
from kitti_utils import load_vertex, load_labels, load_files


def visualize(data_path, prediction_dir, seq, use_ground_truth):
    # frame_id = [1624, 2827]
    scan_paths = load_files(f'{data_path}/sequences/{seq}/velodyne')
    frame_id = list(range(0, len(scan_paths)))  # all frames
    point_size = 5

    def show_pointcloud(vis, id):
        id = "%06d" % id
        gt_path = f'{data_path}/sequences/{seq}/labels/{id}.label'
        prediction_path = f"{prediction_dir}/sequences/{seq}/predictions/{id}.label"
        scan_path = f'{data_path}/sequences/{seq}/velodyne/{id}.bin'
        label_path = f'{data_path}/sequences/{seq}/labels/{id}.label'

        scan = load_vertex(scan_path)
        rho = np.sqrt(scan[:, 0] ** 2 + scan[:, 1] ** 2)
        z = scan[:, 2]
        # valid_mask = (-10 < z) & (z < 2) & (rho < 60) & (rho > 0)
        # scan = scan[valid_mask]

        pcd.points = o3d.utility.Vector3dVector(scan[:, :3])
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # true negative
        colors = np.array(pcd.colors)

        prediction, _ = load_labels(prediction_path)
        # prediction = prediction[valid_mask]
        if use_ground_truth:
            label, _ = load_labels(gt_path)
            # label = label[valid_mask]
            # colors[label > 250] = [0.0, 1.0, 0.0]
            colors[(prediction > 250) & (label > 250)] = [0.0, 1.0, 0.0]  # true positive
            colors[(prediction > 250) & (label <= 250)] = [1.0, 0.0, 0.0]  # false positive
            colors[(prediction <= 250) & (label > 250)] = [0.0, 0.0, 1.0]  # false negative
        else:
            colors[prediction > 250] = [0.0, 1.0, 0.0]
            #colors[(prediction >= 10) & (prediction <= 32)] = [1.0, 0.0, 0.0]

        pcd.colors = o3d.utility.Vector3dVector(colors)
        # vis.clear_geometries()
        vis.add_geometry(pcd)

        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters("utils/viewpoint/viewpoint_kitti.json")  # viewpoint_livox.json
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.poll_events()
        vis.update_renderer()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f'kitti_seq{seq}', width=1000, height=1000)
    vis.get_render_option().point_size = point_size

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    for id in frame_id:
        show_pointcloud(vis, id)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./evaluate_mos.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset dir. No Default',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        required=True,
        help='Prediction dir. Same organization as dataset, but predictions in'
             'each sequences "prediction" directory. No Default.'
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        required=False,
        default="08",
        help='Target sequence, default to 08.',
    )
    parser.add_argument(
        '--use_ground_truth', '-gt',
        type=bool,
        required=False,
        default=False,
        help='Use ground truth for visualization, default to False.',
    )

    FLAGS, unparsed = parser.parse_known_args()
    visualize(FLAGS.dataset, FLAGS.predictions, FLAGS.sequence, FLAGS.use_ground_truth)

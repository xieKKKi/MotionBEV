#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Developed by Xieyuanli Chen
# Brief: This script combines moving object segmentation with semantic information

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

from kitti_utils import load_files, load_labels

if __name__ == '__main__':
    # specify moving object segmentation results folder
    mos_pred_root = '/home/ubuntu/Desktop/MotionBEV/prediction_save_dir_KITTI_amcm'

    # specify semantic segmentation results folder
    semantic_pred_root = '/home/ubuntu/xjp/PolarSeg/out/SemKITTI_test'

    # create a new folder for combined results
    combined_results_root = '/home/ubuntu/Desktop/MotionBEV/combine_semantics_preditions_amcm'

    # create output folder
    seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    if not os.path.exists(os.path.join(combined_results_root, "sequences")):
        os.makedirs(os.path.join(combined_results_root, "sequences"))

    for seq in seqs:
        seq = '{0:02d}'.format(int(seq))
        if not os.path.exists(os.path.join(combined_results_root, "sequences", seq, "predictions")):
            os.makedirs(os.path.join(combined_results_root, "sequences", seq, "predictions"))

    for seq in seqs:
        # load moving object segmentation files
        mos_pred_seq_path = os.path.join(mos_pred_root, "sequences", seq, "predictions")
        mos_pred_files = load_files(mos_pred_seq_path)

        # load semantic segmentation files
        semantic_pred_seq_path = os.path.join(semantic_pred_root, "sequences", seq, "predictions")
        semantic_pred_files = load_files(semantic_pred_seq_path)

        print('processing seq:', seq)

        for frame_idx in tqdm(range(len(mos_pred_files))):
            mos_pred, _ = load_labels(mos_pred_files[frame_idx])  # mos_pred should be 9/251 for static/dynamic
            semantic_pred, _ = load_labels(semantic_pred_files[frame_idx])  # mos_pred should be full classes
            semantic_pred_mapped = np.ones(len(mos_pred), dtype=np.uint32) * 9
            combine_pred = np.ones(len(mos_pred), dtype=np.uint32) * 9

            # mapping semantic into static and movable classes
            movable_mask = (semantic_pred > 0) & (semantic_pred < 9)
            semantic_pred_mapped[movable_mask] = 251

            # if consistent keep the same, otherwise labeled as static
            combined_mask = (semantic_pred_mapped == mos_pred)
            combine_pred[combined_mask] = mos_pred[combined_mask]

            '''if len(mos_pred[mos_pred == 251]) > 0:
                print("moving points: ", len(mos_pred[mos_pred == 251]))
                print("movable points: ", len(semantic_pred_mapped[movable_mask]))
                print("moving and movable points: ", len(combine_pred[combine_pred == 251]))'''
            if len(mos_pred[mos_pred == 251]) > 0 and len(mos_pred[mos_pred == 251]) != len(combine_pred[combine_pred == 251]):
                print("moving points: ", len(mos_pred[mos_pred == 251]))
                print("movable points: ", len(semantic_pred_mapped[movable_mask]))
                print("moving and movable points: ", len(combine_pred[combine_pred == 251]))

            file_name = os.path.join(combined_results_root, "sequences", seq, "predictions", str(frame_idx).zfill(6))
            combine_pred.reshape((-1)).astype(np.uint32)
            combine_pred.tofile(file_name + '.label')

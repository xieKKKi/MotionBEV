#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Developed by Jiapeng Xie
import os
import time
import numpy as np
import torch
from tqdm import tqdm

from network.CA_BEV_Unet import CA_Unet
from network.A_BEV_Unet import BEV_Unet
from network.ptBEVnet import ptBEVnet
from dataloader.dataset import collate_fn_BEV, collate_fn_BEV_test, SemKITTI, spherical_dataset, \
    voxel_dataset, get_SemKITTI_label_name
from config.config import load_config_data
# ignore weird np warning
import warnings

warnings.filterwarnings("ignore")


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 1)
    hist = hist[unique_label, :]
    hist = hist[:, unique_label]
    return hist


def main(arch_config, data_config):
    print("arch_config: ", arch_config)
    print("data_config: ", data_config)

    # parameters
    configs = load_config_data(arch_config)
    data_cfg = configs['data_loader']
    model_cfg = configs['model_params']
    train_cfg = configs['train_params']
    fea_compre = model_cfg['grid_size'][2]
    pytorch_device = torch.device('cuda:0')

    test_batch_size = 1
    prediction_save_dir = './prediction_save_dir_KITTI'
    val = True  # False #True
    test = True  # True #False

    if data_cfg['dataset_type'] == 'polar':
        fea_dim = 9
        circular_padding = True
    elif data_cfg['dataset_type'] == 'traditional':
        fea_dim = 7
        circular_padding = False
    else:
        raise NotImplementedError

    # prepare miou fun
    unique_label, unique_label_str, inv_learning_map = get_SemKITTI_label_name(data_config)

    # prepare model
    if model_cfg['use_co_attention']:
        my_BEV_model = CA_Unet(n_class=len(unique_label),
                               n_height=fea_compre,
                               residual=data_cfg['residual'],
                               input_batch_norm=model_cfg['use_norm'],
                               dropout=model_cfg['dropout'],
                               circular_padding=circular_padding)
    else:
        my_BEV_model = BEV_Unet(n_class=len(unique_label),
                                n_height=fea_compre,
                                residual=data_cfg['residual'],
                                input_batch_norm=model_cfg['use_norm'],
                                dropout=model_cfg['dropout'],
                                circular_padding=circular_padding)
    my_model = ptBEVnet(my_BEV_model,
                        grid_size=model_cfg['grid_size'],
                        fea_dim=fea_dim,
                        ppmodel_init_dim=model_cfg['ppmodel_init_dim'],
                        kernal_size=1,
                        fea_compre=fea_compre)
    model_load_path = train_cfg['model_load_path']
    if os.path.exists(model_load_path):
        print("Load model from: " + model_load_path)
        my_model.load_state_dict(torch.load(model_load_path))
    else:
        print(model_load_path, " : not exist!")
        exit()
    my_model.to(pytorch_device)

    # prepare dataset
    test_pt_dataset = SemKITTI(data_config_path=data_config,
                               data_path=data_cfg['data_path'] + '/sequences/',
                               imageset='test',
                               return_ref=data_cfg['return_ref'],
                               residual=data_cfg['residual'],
                               residual_path=data_cfg['residual_path'],
                               drop_few_static_frames=False)
    val_pt_dataset = SemKITTI(data_config_path=data_config,
                              data_path=data_cfg['data_path'] + '/sequences/',
                              imageset='val',
                              return_ref=data_cfg['return_ref'],
                              residual=data_cfg['residual'],
                              residual_path=data_cfg['residual_path'],
                              drop_few_static_frames=False)

    if data_cfg['dataset_type'] == 'polar':
        test_dataset = spherical_dataset(test_pt_dataset,
                                         grid_size=model_cfg['grid_size'],
                                         fixed_volume_space=data_cfg['fixed_volume_space'],
                                         return_test=True)
        val_dataset = spherical_dataset(val_pt_dataset,
                                        grid_size=model_cfg['grid_size'],
                                        fixed_volume_space=data_cfg['fixed_volume_space'],
                                        return_test=True)
    elif data_cfg['dataset_type'] == 'traditional':
        test_dataset = voxel_dataset(test_pt_dataset,
                                     grid_size=model_cfg['grid_size'],
                                     fixed_volume_space=data_cfg['fixed_volume_space'],
                                     return_test=True)
        val_dataset = voxel_dataset(val_pt_dataset,
                                    grid_size=model_cfg['grid_size'],
                                    fixed_volume_space=data_cfg['fixed_volume_space'],
                                    return_test=True)
    else:
        raise NotImplementedError
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=test_batch_size,
                                                      collate_fn=collate_fn_BEV_test,
                                                      shuffle=False,
                                                      num_workers=data_cfg['num_workers'],
                                                      pin_memory=True)
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=test_batch_size,
                                                     collate_fn=collate_fn_BEV_test,
                                                     shuffle=False,
                                                     num_workers=data_cfg['num_workers'],
                                                     pin_memory=True)

    # validation
    if val:
        print('*' * 80)
        print('Test network performance on validation split')
        print('*' * 80)
        pbar = tqdm(total=len(val_dataset_loader))
        my_model.eval()
        hist_list = []
        time_list = []
        with torch.no_grad():
            for i_iter_val, (val_vox_label, val_grid, val_pt_labs, val_pt_fea, val_index) in enumerate(
                    val_dataset_loader):
                val_pt_fea_ten = [i.to(pytorch_device) for i in val_pt_fea]
                val_grid_ten = [i.to(pytorch_device) for i in val_grid]
                # val_vox_label_ten = val_vox_label.to(pytorch_device)

                torch.cuda.synchronize()
                start_time = time.time()
                predict_labels, pt_out = my_model(val_pt_fea_ten, val_grid_ten, pytorch_device)
                torch.cuda.synchronize()
                time_list.append(time.time() - start_time)

                predict_labels = torch.argmax(predict_labels, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()
                for count, i_val_grid in enumerate(val_grid):
                    hist_list.append(fast_hist_crop(
                        predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]],
                        val_pt_labs[count], unique_label))

                    inv_labels = np.vectorize(inv_learning_map.__getitem__)(
                        predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]])
                    inv_labels = inv_labels.astype('uint32')
                    # print(predict_labels)

                    save_dir = val_pt_dataset.scan_files[val_index[count]]
                    _, dir2 = save_dir.split('/sequences/', 1)
                    new_save_dir = prediction_save_dir + '/sequences/' + dir2.replace('velodyne', 'predictions')[
                                                                         :-3] + 'label'
                    if not os.path.exists(os.path.dirname(new_save_dir)):
                        os.makedirs(os.path.dirname(new_save_dir))

                    inv_labels.tofile(new_save_dir)

                pbar.update(1)
        iou = per_class_iu(sum(hist_list))
        print('Validation per class iou: ')
        for class_name, class_iou in zip(unique_label_str, iou):
            print('%s : %.2f%%' % (class_name, class_iou * 100))
        val_miou = np.nanmean(iou) * 100
        del val_vox_label, val_grid, val_pt_fea, val_grid_ten
        pbar.close()
        print('Current val miou is %.3f ' % val_miou)
        print('Inference time per %d is %.4f seconds\n' % (test_batch_size, np.mean(time_list)))

    # test
    if test:
        print('*' * 80)
        print('Generate predictions for test split')
        print('*' * 80)
        pbar = tqdm(total=len(test_dataset_loader))
        with torch.no_grad():
            for i_iter_test, (_, test_grid, _, test_pt_fea, test_index) in enumerate(test_dataset_loader):
                # predict
                test_pt_fea_ten = [i.to(pytorch_device) for i in test_pt_fea]
                test_grid_ten = [i.to(pytorch_device) for i in test_grid]

                predict_labels, pt_out = my_model(test_pt_fea_ten, test_grid_ten, pytorch_device)
                predict_labels = torch.argmax(predict_labels, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()

                # write to label file
                for count, i_test_grid in enumerate(test_grid):
                    test_pred_label = np.vectorize(inv_learning_map.__getitem__)(
                        predict_labels[count, test_grid[count][:, 0], test_grid[count][:, 1], test_grid[count][:, 2]])
                    test_pred_label = test_pred_label.astype('uint32')

                    save_dir = test_pt_dataset.scan_files[test_index[count]]
                    _, dir2 = save_dir.split('/sequences/', 1)
                    new_save_dir = prediction_save_dir + '/sequences/' + dir2.replace('velodyne', 'predictions')[
                                                                         :-3] + 'label'
                    if not os.path.exists(os.path.dirname(new_save_dir)):
                        os.makedirs(os.path.dirname(new_save_dir))

                    test_pred_label.tofile(new_save_dir)
                pbar.update(1)
        del test_grid, test_pt_fea, test_index
        pbar.close()
    print('Predicted test labels are saved in %s. ' % prediction_save_dir)


if __name__ == '__main__':
    arch_config_path = "config/MotionBEV-semantickitti.yaml"
    data_config_path = "config/semantic-kitti-MOS.yaml"
    main(arch_config_path, data_config_path)

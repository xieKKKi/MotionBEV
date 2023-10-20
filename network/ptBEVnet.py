# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class ptBEVnet(nn.Module):

    def __init__(self, BEV_net, grid_size, fea_dim=3, kernal_size=3,
                 ppmodel_init_dim=32, fea_compre=None):
        super(ptBEVnet, self).__init__()

        self.fea_dim = fea_dim
        self.pt_model_in = PPmodel_in(fea_dim=fea_dim, init_dim=ppmodel_init_dim)
        # self.pt_model_in_res = PPmodel_in(fea_dim=1, init_dim=ppmodel_init_dim//8)
        # self.pt_model_out = PPmodel_out(n_class=2, init_dim=ppmodel_init_dim)

        self.BEV_model = BEV_net
        self.fea_compre = fea_compre
        self.grid_size = grid_size

        # NN stuff
        if kernal_size != 1:
            self.local_pool_op = torch.nn.MaxPool2d(kernal_size, stride=1, padding=(kernal_size - 1) // 2,
                                                    dilation=1)
        else:
            self.local_pool_op = None

        # parametric pooling        
        self.pool_dim = ppmodel_init_dim * 8

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xyz_ind, cur_dev):
        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xyz_ind)):
            cat_pt_ind.append(
                F.pad(xyz_ind[i_batch], (1, 0), 'constant', value=i_batch))  # xy_ind[i_batch]的最后一个维度做padding，前补i_batch

        cat_pt_fea = torch.cat(pt_fea, dim=0)  # 所有batch cat起来
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)

        # shuffle the data
        # shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        # cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        # cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind[:, :3], return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        cat_pt_pos_fea = cat_pt_fea[:, :self.fea_dim]
        cat_pt_res_fea = cat_pt_fea[:, self.fea_dim:]

        # process feature
        processed_cat_pt_fea, pt_fea1, pt_fea2, pt_fea3 = self.pt_model_in(cat_pt_pos_fea)
        # processed_cat_pt_fea_res, pt_fea1_res, pt_fea2_res, pt_fea3_res = self.pt_model_in_res(cat_pt_res_fea)

        # 把index相同的輸入相加  https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]
        pooled_res_data = torch_scatter.scatter_max(cat_pt_res_fea, unq_inv, dim=0)[0]  # scatter_mean

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        # stuff pooled data into 4D tensor
        out_data_dim = [len(pt_fea), self.grid_size[0], self.grid_size[1], self.pt_fea_dim]  # 4, 480, 360, 32
        out_data = torch.zeros(out_data_dim, dtype=torch.float32).to(cur_dev)
        out_data[unq[:, 0], unq[:, 1], unq[:, 2], :] = processed_pooled_data  # 给每一格赋值
        out_data = out_data.permute(0, 3, 1, 2)  # 维度 0,1,2,3 -> 0,3,1,2  即4, 32, 480, 360
        if self.local_pool_op != None:
            out_data = self.local_pool_op(out_data)

        res_data_dim = [len(pt_fea), self.grid_size[0], self.grid_size[1], cat_pt_res_fea.shape[1]]  # 4, 480, 360, 1
        res_data = torch.zeros(res_data_dim, dtype=torch.float32).to(cur_dev)
        res_data[unq[:, 0], unq[:, 1], unq[:, 2], :] = pooled_res_data  # 给每一格赋值
        res_data = res_data.permute(0, 3, 1, 2)  # 维度 0,1,2,3 -> 0,3,1,2  即4, 1, 480, 360

        # run through network
        net_return_voxel_data = self.BEV_model(
            out_data, res_data)

        # net_return_pt_fea = net_return_voxel_data[cat_pt_ind[:, 0], :, cat_pt_ind[:, 1], cat_pt_ind[:, 2], cat_pt_ind[:, 3]]
        # net_return_pt_data = self.pt_model_out(net_return_pt_fea, pt_fea1, pt_fea2, pt_fea3)
        net_return_pt_data = None

        return net_return_voxel_data, net_return_pt_data


class PPmodel_in(nn.Module):
    def __init__(self, fea_dim=9, init_dim=32):
        super(PPmodel_in, self).__init__()
        self.bn1 = nn.BatchNorm1d(fea_dim)

        self.layer1 = nn.Sequential(
            nn.Linear(fea_dim, init_dim),
            nn.BatchNorm1d(init_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(init_dim, init_dim * 2),
            nn.BatchNorm1d(init_dim * 2),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(init_dim * 2, init_dim * 4),
            nn.BatchNorm1d(init_dim * 4),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Linear(init_dim * 4, init_dim * 8)

    def forward(self, pt_fea):
        pt_fea = self.bn1(pt_fea)
        pt_fea1 = self.layer1(pt_fea)
        pt_fea2 = self.layer2(pt_fea1)
        pt_fea3 = self.layer3(pt_fea2)
        processed_pt_fea = self.out(pt_fea3)
        return processed_pt_fea, pt_fea1, pt_fea2, pt_fea3


class PPmodel_out(nn.Module):
    def __init__(self, n_class=2, init_dim=32):
        super(PPmodel_out, self).__init__()
        self.input = nn.Linear(n_class, init_dim * 4)

        self.layer1 = nn.Sequential(
            nn.Linear(init_dim * 4, init_dim * 2),
            nn.BatchNorm1d(init_dim * 2),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(init_dim * 2, init_dim),
            nn.BatchNorm1d(init_dim),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(init_dim, init_dim // 2),
            nn.BatchNorm1d(init_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Linear(init_dim // 2, n_class)
        # self.out = nn.Conv1d(init_dim // 2, n_class, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, return_pt_fea, res_pt_fea1, res_pt_fea2, res_pt_fea3):
        pt_fea = self.input(return_pt_fea)

        pt_fea = self.layer1(pt_fea + res_pt_fea3)
        pt_fea = self.dropout(pt_fea)

        pt_fea = self.layer2(pt_fea + res_pt_fea2)
        pt_fea = self.dropout(pt_fea)

        pt_fea = self.layer3(pt_fea + res_pt_fea1)
        pt_fea = self.dropout(pt_fea)
        processed_pt_fea = self.out(pt_fea)
        return processed_pt_fea

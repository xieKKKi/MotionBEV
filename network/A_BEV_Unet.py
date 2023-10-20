# -*- coding: utf-8 -*-
# Developed by Jiapeng Xie
import torch.nn as nn
from network.basic_blocks import inconv, outconv, down, up


class BEV_Unet(nn.Module):

    def __init__(self, n_class, n_height, residual, dilation=1, group_conv=False, input_batch_norm=False, dropout=0.,
                 circular_padding=False, dropblock=True, use_vis_fea=False):
        super(BEV_Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        if use_vis_fea:
            self.network = UNet(n_class * n_height, 2 * n_height, residual, dilation, group_conv, input_batch_norm,
                                dropout,
                                circular_padding, dropblock)
        else:
            self.network = UNet(n_class * n_height, n_height, residual, dilation, group_conv, input_batch_norm, dropout,
                                circular_padding, dropblock)

    def forward(self, x, res):
        x = self.network(x, res)

        x = x.permute(0, 2, 3, 1)  # 维度 0,1,2,3 -> 0,2,3,1  即4, 480, 360, 32
        new_shape = list(x.size())[:3] + [self.n_height, self.n_class]
        x = x.view(new_shape)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class UNet(nn.Module):
    def __init__(self, n_class, n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock):
        super(UNet, self).__init__()
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc = inconv(residual, residual * 2, dilation, input_batch_norm, circular_padding)
        self.res_down1 = down(residual * 2, residual * 4, dilation, group_conv, circular_padding)
        self.res_down2 = down(residual * 4, residual * 8, dilation, group_conv, circular_padding)
        self.res_down3 = down(residual * 8, residual * 16, dilation, group_conv, circular_padding)
        #self.res_down4 = down(512, 512, dilation, group_conv, circular_padding)

        #dropblock = False
        self.up1 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up2 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up3 = up(128, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)
        self.up4 = up(64, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout)

        self.dropout = nn.Dropout(p=0. if dropblock else dropout)
        #self.dropout = nn.Dropout(p=0)
        self.outc = outconv(32, n_class)

        self.use_attention = "MGA"
        if self.use_attention == "MGA":
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.conv1x1_conv1_channel_wise = nn.Conv2d(32, 32, 1, bias=True)
            self.conv1x1_conv1_spatial = nn.Conv2d(residual * 2, 1, 1, bias=True)

            self.conv1x1_layer0_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
            self.conv1x1_layer0_spatial = nn.Conv2d(residual * 4, 1, 1, bias=True)

            self.conv1x1_layer1_channel_wise = nn.Conv2d(128, 128, 1, bias=True)
            self.conv1x1_layer1_spatial = nn.Conv2d(residual * 8, 1, 1, bias=True)

            self.conv1x1_layer2_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
            self.conv1x1_layer2_spatial = nn.Conv2d(residual * 16, 1, 1, bias=True)

            #self.conv1x1_layer3_channel_wise = nn.Conv2d(512, 512, 1, bias=True)
            #self.conv1x1_layer3_spatial = nn.Conv2d(512, 1, 1, bias=True)

        else:
            raise NotImplementedError

    def forward(self, x, res):
        res1 = self.res_inc(res)
        res2 = self.res_down1(res1)
        res3 = self.res_down2(res2)
        res4 = self.res_down3(res3)
        # res5 = self.res_down4(res4)

        x1 = self.inc(x)
        # Bridging two specific branches using MotionGuidedAttention
        if self.use_attention == "MGA":
            x1 = self.encoder_attention_module_MGA_tmc(x1, res1, self.conv1x1_conv1_channel_wise,
                                                       self.conv1x1_conv1_spatial)
            x2 = self.down1(x1)
            x2 = self.encoder_attention_module_MGA_tmc(x2, res2, self.conv1x1_layer0_channel_wise,
                                                       self.conv1x1_layer0_spatial)
            x3 = self.down2(x2)
            x3 = self.encoder_attention_module_MGA_tmc(x3, res3, self.conv1x1_layer1_channel_wise,
                                                       self.conv1x1_layer1_spatial)
            x4 = self.down3(x3)

            x4 = self.encoder_attention_module_MGA_tmc(x4, res4, self.conv1x1_layer2_channel_wise,
                                                       self.conv1x1_layer2_spatial)
            x5 = self.down4(x4)

            '''x5 = self.encoder_attention_module_MGA_tmc(x5, res5, self.conv1x1_layer3_channel_wise,
                                                       self.conv1x1_layer3_spatial)
            x5 = self.down5(x5)'''
        else:
            raise NotImplementedError

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(self.dropout(x))
        return x

    def encoder_attention_module_MGA_tmc(self, img_feat, flow_feat, conv1x1_channel_wise, conv1x1_spatial):
        """
            flow_feat_map:  [bsize, 1, h, w]
            feat_vec:       [bsize, channel, 1, 1]
            channel_attentioned_img_feat:  [bsize, channel, h, w]
        """
        # spatial attention
        flow_feat_map = conv1x1_spatial(flow_feat)
        flow_feat_map = nn.Sigmoid()(flow_feat_map)
        spatial_attentioned_img_feat = flow_feat_map * img_feat

        # channel-wise attention
        feat_vec = self.avg_pool(spatial_attentioned_img_feat)
        feat_vec = conv1x1_channel_wise(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec

        final_feat = channel_attentioned_img_feat + img_feat
        return final_feat


import torch
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt
from spectral import spy_colors


def pred_cls_map_dl(sample_list, net, gt_hsi, path, model_type_flag, device):
    pred_sample = []
    pred_label = []

    net.eval()
    if len(sample_list) == 1:
        iter = sample_list[0]
        if model_type_flag == 1:  # data for single spatial net
            for X_spa, y in iter:
                X_spa = X_spa.to(device)
                pre_y = net(X_spa).cpu().argmax(axis=1).detach().numpy()
                pred_sample.extend(pre_y + 1)
        elif model_type_flag == 2:  # data for single spectral net
            for X_spe, y in iter:
                X_spe = X_spe.to(device)
                pre_y = net(X_spe).cpu().argmax(axis=1).detach().numpy()
                pred_sample.extend(pre_y + 1)
        elif model_type_flag == 3:
            for X_spa, X_spe, y in iter:
                X_spa, X_spe, y = X_spa.to(device), X_spe.to(device), y.to(device)
                pre_y = net(X_spa, X_spe).cpu().argmax(axis=1).detach().numpy()
                pred_sample.extend(pre_y + 1)
    elif len(sample_list) == 2:
        iter, index = sample_list[0], sample_list[1]
        if model_type_flag == 1:  # data for single spatial net
            for X_spa, y in iter:
                X_spa = X_spa.to(device)
                pre_y = net(X_spa).cpu().argmax(axis=1).detach().numpy()
                pred_label.extend(pre_y + 1)
        elif model_type_flag == 2:  # data for single spectral net
            for X_spe, y in iter:
                X_spe = X_spe.to(device)
                pre_y = net(X_spe).cpu().argmax(axis=1).detach().numpy()
                pred_label.extend(pre_y + 1)
        elif model_type_flag == 3:
            for X_spa, X_spe, y in iter:
                X_spa, X_spe, y = X_spa.to(device), X_spe.to(device), y.to(device)
                pre_y = net(X_spa, X_spe).cpu().argmax(axis=1).detach().numpy()
                pred_label.extend(pre_y + 1)

        gt = np.ravel(gt_hsi)
        pred_sample = np.zeros(gt.shape)
        pred_sample[index] = pred_label
    pred_hsi = np.reshape(pred_sample, (gt_hsi.shape[0], gt_hsi.shape[1]))
    spy.save_rgb(path + '_' + str(len(sample_list)) + '_pre.png', pred_hsi, colors=spy_colors)
    print('------Get pred classification maps successful-------')

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:48:37 2020

@author: cxy
"""

import numpy as np
from numpy.linalg import norm
from skimage.measure import compare_psnr, compare_ssim, compare_mse

def mpsnr(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的均方误差
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    n_bands = x_true.shape[2]
    p = [compare_psnr(x_true[:, :, k], x_pred[:, :, k], dynamic_range=np.max(x_true[:, :, k])) for k in range(n_bands)]
    return np.mean(p)


def sam(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape
    sam_rad = np.zeros([x_pred.shape[0],x_pred.shape[1]])
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            a = tmp_pred / (norm(tmp_pred)+1e-13)
            b = tmp_true / (norm(tmp_true)+1e-13)
            sam_rad[x, y] = np.arccos(np.dot(a.T,b))
    #sam_deg = sam_rad.mean() * 180 / np.pi
    return sam_rad.mean()


def ergas(x_true, x_pred):
    m, n, k = np.shape(x_true)
    mm, nn, kk = np.shape(x_pred)
    m = np.min([m,mm])
    n = np.min([n,nn])
    k = np.min([k,kk])
    x_true = x_true[0:m,0:n,0:k]
    x_pred = x_pred[0:m,0:n,0:k]
    
    ergas = 0
    for i in range(0,k):
        ergas = ergas + (np.power(norm(x_true[:,:,i]-x_pred[:,:,i]),2)/np.power(m*n,2))/np.power(np.mean(x_true[:,:,i]),2)
        
    ergas_value = 100 * np.sqrt(ergas/k)
    return ergas_value


def mssim(x_true,x_pred):
    """
        :param x_true: 高光谱图像：格式：(H, W, C)
        :param x_pred: 高光谱图像：格式：(H, W, C)
        :return: 计算原始高光谱数据与重构高光谱数据的结构相似度
    """
    SSIM = compare_ssim(X=x_true, Y=x_pred, multichannel=True)
    return SSIM
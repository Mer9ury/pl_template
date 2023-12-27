import pandas as pd
from scipy import stats
import numpy as np
from skimage import io, color

def psnr(x, y, max_val=1.):
    """Peak Signal-to-Noise Ratio"""
    x, y = np.float32(x), np.float32(y)
    mse = np.mean((x - y) ** 2)
    return 20 * np.log10(max_val / np.sqrt(mse))

def deltae_dist (y_true, y_pred):
    """
    Calcultae DeltaE discance in the LAB color space.
    Images must numpy arrays.
    """

    gt_lab  = color.rgb2lab((y_true*255).astype('uint8'))
    out_lab = color.rgb2lab((y_pred*255).astype('uint8'))
    l2_lab  = ((gt_lab - out_lab)**2).mean()
    l2_lab  = np.sqrt(((gt_lab - out_lab)**2).sum(axis=-1)).mean()
    return l2_lab
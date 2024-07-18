import pandas as pd
from scipy import stats
import numpy as np

def plcc(x, y):
    """Pearson Linear Correlation Coefficient"""
    x, y = np.float32(x), np.float32(y)
    return stats.pearsonr(x, y)[0]

def srocc(xs, ys):
    """Spearman Rank Order Correlation Coefficient"""
    xranks = pd.Series(xs).rank()    
    yranks = pd.Series(ys).rank()    
    return plcc(xranks, yranks)

def krocc(x,y):
    """Kendall's Rank Order Correlation Coefficient"""
    x, y = np.float32(x), np.float32(y)
    return stats.kendalltau(x, y)[0]
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_absolute_error

##metric 1 mean_absolute_error
##metric 2 mean_absolute_error_with_bound MAE_with_bound -2.5 to 2.5 

def MAE_with_bound(y_true, y_pred, upper=2.5, lower=-2.5):
    y_true_bounded = y_true.copy()
    y_true_bounded[np.where(y_true>=upper)] = upper
    y_true_bounded[np.where(y_true<=lower)] = lower
    return mean_absolute_error(y_true_bounded, y_pred)

def MAE_with_log_smooth(train_y_raw, train_y_pred, range=5):
    diff = np.abs(train_y_raw - train_y_pred)/range
    diff[np.where(diff>1)] = np.log(diff[np.where(diff>1)]) + 1
    return np.mean(diff * range)

##metric 3 accuracy 
# def ratio_of_correct(y_true_descrete, y_pred_descrete):
#     return np.mean(y_pred_descrete == y_true_descrete)

##metric 4 accuracy with torlarance
def ratio_of_correct(y_true_descrete, y_pred_descrete, eps=None):
    if eps is None:
        return np.mean(y_pred_descrete == y_true_descrete)
    else:
        return np.mean(np.abs(y_pred_descrete - y_true_descrete)<= eps)

def count_of_correct_torch(y, pred):
    y_descrete = (y*2).type(torch.LongTensor)
    pred_descrete = (pred*2).type(torch.LongTensor)
    return torch.sum(y_descrete== pred_descrete)

###transformations for Y
def descrete_y_numpy(y, multiplier=2):
    return np.array(y*multiplier, dtype = 'int')

def y_transform_with_cutoff(y_df, cutoff=(-2.5, 2.5)):
    y_res = y_df.copy()
    y_res[np.where(y_res<=cutoff[0])] = cutoff[0]
    y_res[np.where(y_res>=cutoff[1])] = cutoff[1]
    return y_res

def y_transform_to_class(y_df):
    y_res = y_df.copy()
    y_res[np.where(y_res!=0)] = 1

    y_res[np.where(y_df<=-2.5)] = 0
    y_res[np.where(y_df==-2)] = 1
    y_res[np.where(y_df==-1.5)] = 2
    y_res[np.where(y_df==-1)] = 3
    y_res[np.where(y_df==-0.5)] = 4
    y_res[np.where(y_df==0)] = 5 
    y_res[np.where(y_df==0.5)] = 6
    y_res[np.where(y_df==1)] = 7
    y_res[np.where(y_df==1.5)] = 8
    y_res[np.where(y_df==2)] = 9
    y_res[np.where(y_df>=2.5)] = 10

    y_res.astype('int')
    return y_res

def map_class_to_y(y):
    y = y/2-2.5
    # y = y-1
    return y
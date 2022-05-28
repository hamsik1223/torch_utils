import numpy as np
import pandas as pd
import os

def ts_data_split(X_list,
                   y_list,
             total_days, 
             test_day = 1,
             kfold = 5,
             gap = 0
             ):
    remain_days = total_days - test_day
    cut_off_index = 1+np.array(np.quantile(np.arange(remain_days), 
                         np.arange(kfold+1)/kfold), dtype = 'int')[1:]
    dataset_list = []
    for cut_off in cut_off_index:
        # print(0, cut_off, cut_off + test_day)
        dataset_list.append((X_list[0:cut_off], y_list[0:cut_off], \
              X_list[cut_off: (cut_off+test_day)], y_list[cut_off: (cut_off+test_day)]))
        print('data cut offs are: ', 0, cut_off, cut_off+test_day)
    return dataset_list

def generate_train_dev_test(X_list, y_list, test_day=10, kfold = 1, mode = 'simple', need_test=True):
    
    train_X_list, train_y_list, test_X, test_y = \
      ts_data_split(X_list, y_list, len(X_list), test_day=test_day, kfold = 1)[0]

    test_X, test_y = pd.concat(test_X), pd.concat(test_y)
    test_y = np.array(test_y['target']).reshape([-1])
    
    if not need_test:
        return [(
          pd.concat(train_X_list),
          np.array(pd.concat(train_y_list)).reshape([-1]),
          test_X, 
          test_y
        )]

    if mode == 'simple': #强制改为1
        kfold = 1

    train_dev_dataset_tmp = \
    ts_data_split(train_X_list, train_y_list, len(train_X_list), test_day=test_day, kfold = kfold)

    train_dev_dataset = []
    for train_X_list, train_y_list, dev_X, dev_y in train_dev_dataset_tmp:
        train_X_df, train_y_df, dev_X_df, dev_y_df = \
            pd.concat(train_X_list), pd.concat(train_y_list), \
            pd.concat(dev_X), pd.concat(dev_y)
        train_y_arr = np.array(train_y_df['target']).reshape([-1])
        dev_y_arr = np.array(dev_y_df['target']).reshape([-1])
        
        train_dev_dataset.append((train_X_df, train_y_arr, dev_X_df, dev_y_arr ))
        print('train size: ', len(train_dev_dataset[-1][0]), ', dev size: ', len(train_dev_dataset[-1][2]))
    return train_dev_dataset, test_X, test_y
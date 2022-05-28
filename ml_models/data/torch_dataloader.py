import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, SequentialSampler, ##to convert numpy array to pytorch dateset..
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler

# def create_data_loaders(x, 
#                         y, batch_size= 64, shuffle= True, sampler = None):
#     x = np.array(x); y = np.array(y)
#     trainx = torch.Tensor(x)
#     trainy = torch.Tensor(y).reshape([-1, 1])
    
#     # trainy = torch.Tensor(y).type(torch.LongTensor)
#     training_data = TensorDataset(trainx, trainy)
#     if sampler is None:
#         train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle = shuffle)
#     else:
#         train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle = False, sampler = sampler)
#     return train_dataloader


# def downsample(train_dev_dataset, down_sample_factor, BSZ=64, num_B = 128):
#     train_sample_weights = np.ones([train_dev_dataset[0][1].shape[0]])
#     train_sample_weights[np.where(train_dev_dataset[0][1] == 0 )[0]] = down_sample_factor
#     weighted_sampler = WeightedRandomSampler(train_sample_weights, BSZ*num_B)
#     train_dataloader = create_data_loaders(train_dev_dataset[0][0], train_dev_dataset[0][1], 
#                                            batch_size = BSZ,
#                                            sampler = weighted_sampler)    
#     return train_dataloader

def data_loader(*array, 
                dev_mode=False,
                bsz = 64,
                down_sample_factor=0.1,
                pc=False):
    array = tuple(torch.tensor(data) for data in array)
    if not dev_mode:
        train_data = TensorDataset(*array)
        if pc:
            assert 0, 'pc not supported yet.'
        elif down_sample_factor is not None:
            labels = array[-1]
            train_sample_weights = np.ones([labels.shape[0]])
            train_sample_weights[np.where(labels==0)[0]] = down_sample_factor
            train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights), False)
        else:
            train_sampler = RandomSampler(train_data)
        dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bsz)
    else:
        val_data=TensorDataset(*array)
        val_sampler=SequentialSampler(val_data)
        dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=bsz)
    return dataloader
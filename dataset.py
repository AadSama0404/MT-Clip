'''
coding:utf-8
@Software:PyCharm
@Time:2024/8/14 11:57
@Author:tianyi.zhu
'''
from torch.utils.data import Dataset
import torch
import numpy as np


class tmb_dataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
            raw_data: raw_data: [['Study ID', 'ORR', 'PFS', 'Status', ['TMB_sum', 'AF_avg', 'CCF_clone']]]

            sorted_data: [['TMB_sum', 'AF_avg', 'CCF_clone']]
            response: ['Study ID', 'ORR', 'PFS', 'Status']
        '''
        sample = self.data[idx]

        study_id = int(sample[0])
        lable = float(sample[1])
        PFS = float(sample[2])
        Status = int(sample[3])
        response = torch.tensor([study_id, lable, PFS, Status], dtype=torch.float)

        features = torch.tensor(sample[4], dtype=torch.float)

        _, sorted_indices = torch.sort(features[:, -1], descending=True)
        sorted_features = features[sorted_indices]

        return (sorted_features, response)
import pickle

import numpy as np

from torch.utils.data import Dataset


class SampleVectorDataset(Dataset):
    """ Sample Vector Dataset - random vectors generated by the create_sample_vectors script """

    def __init__(self, data_pickle):
        self.dataframe = pickle.load(open(data_pickle, 'rb')) # dataset is small, so we can just keep this in memory

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return np.array(self.dataframe['vector'][index]), self.dataframe['class'][index]
import os
import json
from datasets.preprocessing import PoscarToGraph
import torch
import numpy as np


def load_json(file_name):
    with open(file_name, 'r') as fp:
        dataset = json.load(fp)
    return dataset


class NbMoTaW_Dataset(object):
    def __init__(self, radius=6, max_neigh=200, is_train=True):
        self.dataset_dir = self._dataset_dir()
        self.converter = PoscarToGraph(radius=radius, max_neigh=max_neigh)
        self.is_train = is_train
        self.dataset = self.load_data()

    @staticmethod
    def _dataset_dir():
        current_file_dir = os.path.dirname(__file__)
        json_dir = os.path.normpath(os.path.join(current_file_dir, 'NbMoTaW'))
        return json_dir

    def file_list(self):
        files = os.listdir(self.dataset_dir)
        train_files, test_files = [], []
        for file in files:
            if 'json' in file:
                if 'Test' in file:
                    test_files.append(file)
                else:
                    train_files.append(file)
        return train_files, test_files

    def load_data(self):
        train_files, test_files = self.file_list()
        file_list = train_files if self.is_train else test_files
        print('loading dataset from {}'.format(self.dataset_dir))
        dataset = []
        for file in file_list:
            print('loading dataset from file {}'.format(file))
            data_i = load_json(os.path.join(self.dataset_dir, file))
            # print('{} has {} different structures.'.format(file, len(data_i)))
            dataset.extend(data_i)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        mat = self.dataset[item]
        data = self.converter.to_graph_from_dict(mat.get('structure'))
        data.__setitem__('energy', mat.get('outputs').get('energy'))
        forces = mat.get('outputs').get('forces')
        forces = torch.tensor(forces, dtype=torch.float)
        data.__setitem__('force', forces)
        return data

    def to_list(self):
        data_list = []
        for i in range(self.__len__()):
            data_list.append(self.__getitem__(i))
        return data_list


class NbMoTaW_Mix(object):
    def __init__(self, radius=6, max_neigh=200, is_train=True):
        self.dataset_dir = self._dataset_dir()
        self.converter = PoscarToGraph(radius=radius, max_neigh=max_neigh)
        self.is_train = is_train
        self.dataset = self.load_data()

    @staticmethod
    def _dataset_dir():
        current_file_dir = os.path.dirname(__file__)
        json_dir = os.path.normpath(os.path.join(current_file_dir, 'NbMoTaW'))
        return json_dir

    def file_list(self):
        files = os.listdir(self.dataset_dir)
        train_files, test_files = [], []
        for file in files:
            if 'json' in file:
                if 'Test' in file:
                    test_files.append(file)
                else:
                    train_files.append(file)
        return train_files, test_files

    def load_data(self):
        train_files, test_files = self.file_list()
        file_list = train_files if self.is_train else test_files
        print('loading dataset from {}'.format(self.dataset_dir))
        dataset = []
        for file in file_list:
            print('loading dataset from file {}'.format(file))
            data_i = load_json(os.path.join(self.dataset_dir, file))
            # print('{} has {} different structures.'.format(file, len(data_i)))
            dataset.extend(data_i)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        mat = self.dataset[item]
        data = self.converter.to_graph_from_dict(mat.get('structure'))
        data.__setitem__('energy', mat.get('outputs').get('energy'))
        forces = mat.get('outputs').get('forces')
        forces = torch.tensor(forces, dtype=torch.float)
        data.__setitem__('force', forces)
        return data

    def to_list(self):
        data_list = []
        for i in range(self.__len__()):
            data_list.append(self.__getitem__(i))
        return data_list


if __name__ == '__main__':
    # a,b=NbMoTaW_Dataset().file_list()
    dataset = NbMoTaW_Dataset(is_train=False)
    print(dataset.__len__())
    data = dataset.__getitem__(0)
    ac = data.atomic_numbers
    print(ac)


    # length = len(dataset)
    # total_energy = []
    # for i in range(length):
    #     data = dataset.__getitem__(i)
    #     print(data.atomic_numbers)
    #     # print(data.num_atoms)
    #     print(data.energy)
    #     total_energy.append(data.energy)
    # mean = np.mean(total_energy)
    # std = np.std(total_energy)
    # print(mean, std)
    # print(dataset.__getitem__(650).atomic_numbers)
    # print(dataset.__getitem__(1320).atomic_numbers)

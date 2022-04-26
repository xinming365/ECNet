import os
from torch.utils.data import Dataset
from datasets.preprocessing import PoscarToGraph
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from torch_geometric.datasets import tu_dataset, qm9
from torch_geometric.data import InMemoryDataset
import torch
import torch_geometric
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from typing import List
task_dict = {'etot': 'Etot (eV/atom)',
             'etot_all': 'Etot (eV)',
             'emix': 'Emix (eV/atom)',
             'eform': 'Eform (eV/atom)',
             'ms': 'Ms (mub/atom)',
             'ms_all': 'Ms (emu/g)',
             'mb': 'mb (mub/cell)',
             'rmsd': 'rmsd (\\AA)',
             }
exclude_task_dict = {'etot': False,
             'etot_all': False,
             'emix': 'Emix (eV/atom)',
             'eform': 'Eform (eV/atom)',
             'ms': 'Ms (mub/atom)',
             'ms_all': 'Ms (emu/g)',
             'mb': 'mb (mub/cell)',
             'rmsd': 'rmsd (\\AA)',
             }


class HEADataset(Dataset):
    def __init__(self, poscar_dir, label_name, task=None, exclude_property=None):
        """

        :param poscar_dir:
        :param label_name:
        :param task:
        :param exclude_property:  If we want to exclude some properties, you can input like: ['emix']
        """
        self.dir_name = poscar_dir
        self.label_name = label_name
        self.labels = self.read_excel()
        self.task = task
        self.exclude_property = exclude_property

    def __len__(self):
        return len(self.labels)

    def read_excel(self):
        wb = pd.read_excel(io=self.label_name, sheet_name=0, engine='openpyxl')
        data = {}
        values = wb.values
        # the shape of values is (x_rows,11)
        targets = wb.columns
        # the shape of targets is (1, 11)
        targets = targets[1:]
        for value in values:
            value = value[1:]
            target = {}
            for i, t in enumerate(targets):
                if i < 1:
                    pass
                else:
                    target[t] = value[i]
            data[value[0]] = target
        return data

    def _set_attribute(self, data: Data, properties: dict, specific_task=None, exclude_property=None):
        r_etot = True
        r_etot_all = False
        r_emix = True
        r_eform = True
        r_ms_all = False
        r_ms = True
        r_mb = True
        r_rmsd = True
        r_v0 = True

        if specific_task:
            assert specific_task in task_dict.keys(), 'Please make sure the task belongs to the task_dict'
            data.__setitem__('y', properties[task_dict.get(specific_task)])
        if exclude_property:
            assert exclude_property in task_dict.keys(), 'Please make sure the task belongs to the task_dict'
            if exclude_property=='emix':
                r_emix = False
        if r_etot:
            data.__setitem__('etot', properties['Etot (eV/atom)'])
        if r_etot_all:
            data.__setitem__('etot_all', properties['Etot (eV)'])
        if r_emix:
            data.__setitem__('emix', properties['Emix (eV/atom)'])
        if r_eform:
            data.__setitem__('eform', properties['Eform (eV/atom)'])
        if r_ms:
            data.__setitem__('ms', properties['Ms (mub/atom)'])
        if r_ms_all:
            data.__setitem__('ms_all', properties['Ms (emu/g)'])
        if r_mb:
            data.__setitem__('mb', properties['mb (mub/cell)'])
        if r_rmsd:
            data.__setitem__('rmsd', properties['rmsd (\\AA)'])
        if r_v0:
            data.__setitem__('v0', properties['V0 (A3/atom)'])

    def __getitem__(self, index):
        file_list = list(self.labels.keys())
        # file_list = np.random.permutation(file_list) This is a bug, which causes some poscars not readed.
        file = file_list[index]
        poscar_path = os.path.join(self.dir_name, file)
        p2g = PoscarToGraph(radius=6, max_neigh=200)
        data = p2g.to_graph(filename=poscar_path)
        # print(file)
        properties = self.labels[file]
        self._set_attribute(data=data, properties=properties, specific_task=self.task,
                            exclude_property=self.exclude_property)
        return data


# class NbMoTaW_Dataset(object):
#     def __init__(self, radius=6, max_neigh=200, is_train=True):
#         self.dataset_dir = self._dataset_dir()
#         self.converter = PoscarToGraph(radius=radius, max_neigh=max_neigh)
#         self.is_train = is_train
#         self.dataset = self.load_data()
#
#     @staticmethod
#     def _dataset_dir():
#         current_file_dir = os.path.dirname(__file__)
#         json_dir = os.path.normpath(os.path.join(current_file_dir, 'NbMoTaW'))
#         return json_dir
#
#     def file_list(self):
#         files = os.listdir(self.dataset_dir)
#         train_files, test_files = [], []
#         for file in files:
#             if 'json' in file:
#                 if 'Test' in file:
#                     test_files.append(file)
#                 else:
#                     train_files.append(file)
#         return train_files, test_files
#
#     def load_data(self):
#         train_files, test_files = self.file_list()
#         file_list = train_files if self.is_train else test_files
#         print('loading dataset from {}'.format(self.dataset_dir))
#         dataset = []
#         for file in file_list:
#             print('loading dataset from file {}'.format(file))
#             data_i = load_json(os.path.join(self.dataset_dir, file))
#             # print('{} has {} different structures.'.format(file, len(data_i)))
#             dataset.extend(data_i)
#         return dataset
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, item):
#         mat = self.dataset[item]
#         data = self.converter.to_graph_from_dict(mat.get('structure'))
#         data.__setitem__('energy', mat.get('outputs').get('energy'))
#         forces = mat.get('outputs').get('forces')
#         forces = torch.tensor(forces, dtype=torch.float)
#         data.__setitem__('force', forces)
#         return data
#
#     def to_list(self):
#         data_list = []
#         for i in range(self.__len__()):
#             data_list.append(self.__getitem__(i))
#         return data_list


if __name__ == '__main__':
    dataset = HEADataset(poscar_dir='../HEA_Data/POSCARS', label_name='../HEA_Data/Out_labels/Database.xlsx', task='etot')

    train_loader = DataLoader(dataset=dataset, batch_size=2)
    print(train_loader.dataset[0])
    for i, b in enumerate(train_loader):
        print(b)
        print(b.etot)
        print(b.num_atoms)
    # print(a)

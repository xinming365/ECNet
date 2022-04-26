from pymatgen.core.structure import IStructure
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from copy import deepcopy
import json
from typing import List, Tuple, Dict
import time
import os
import global_var as gv
from queue import Queue
from threading import Thread
from torch_geometric.data import InMemoryDataset
import csv
from pymatgen.ext.matproj import MPRester
from tqdm import tqdm

my_api_key = 'sKlJKcjcF2RWAJ7Fy7F'


def load_dataset(file_name):
    with open(file_name, 'r') as fp:
        dataset = json.load(fp)
    return dataset


def get_len(mp_dataset):
    total_len = len(mp_dataset)
    len_ef, len_k = 0, 0
    k_dataset = []
    for i in range(total_len):
        if 'formation_energy_per_atom' in mp_dataset[i].keys():
            len_ef += 1
        if 'K' in mp_dataset[i].keys():
            len_k += 1
            k_dataset.append(mp_dataset[i])
    return len_ef, len_k, k_dataset


def convert_n_in_mp():
    dataset = []
    with open('./db.csv', 'r') as fp:
        data = csv.reader(fp)
        head = next(data)
        for item in tqdm(data):
            mat_dict = {}
            mpid = item[0]  # mp-id
            # print(mpid)
            with MPRester(my_api_key) as m:
                structure = m.get_structure_by_material_id(mpid)
                structure_str = structure.to('cif')
                mat_dict['structure'] = structure_str  # download the structure
            mat_dict['material_id'] = mpid
            mat_dict['refractive_index'] = item[2]  # refractive index_1
            mat_dict['formula'] = item[1]
            mat_dict['optical_gap'] = item[-4]  # average optical gap (eV)
            mat_dict['direct_gap'] = item[5]  # gga direct band gap (eV)
            dataset.append(mat_dict)
    with open('./refractive_index.json', 'w') as fp:
        json.dump(dataset, fp)


class MpDataset():
    """
    This is a class for converting the open MP-crystals-2018.6.1 crystal data set into our model inputs.
    The dataset is from https://figshare.com/articles/dataset/Graphs_of_materials_project/7451351.

    This is the first version of mp dataset. We found that this dataset would take up a lot of memory.
    Furthermore, it takes additional time when loading mp dataset because  it reads all the mp data at once.
    Due to the above reasons, we do not use this class. But we keep this class in order for the reference.
    """

    def __init__(self, data_set, task):
        """
        The dataset has been downloaded into local environment.
        args:
        task: str. could be ef, eg, g, k.
        """

        self.task = task
        self.dataset = data_set
        self.len_ef, self.len_k = self._get_len()
        self.task_dataset = self._get_task_dataset()

    @property
    def task_dict(self):
        return dict(ef='formation_energy_per_atom', eg='band_gap', g='G', k='K')

    def _get_task_dataset(self):
        if self.task == 'g' or self.task == 'k':
            total_len = len(self.dataset)
            k_dataset = []
            for i in range(total_len):
                if 'K' in self.dataset[i].keys():
                    k_dataset.append(self.dataset[i])
            return k_dataset
        elif self.task == 'egn':
            total_len = len(self.dataset)
            egn_dataset = []
            for i in range(total_len):
                eg = self.dataset[i].get('band_gap')
                if eg > 0:
                    egn_dataset.append(self.dataset[i])
            return egn_dataset
        else:
            return self.dataset

    def _get_len(self):
        """
        For the sake of reducing calculations, we move this function out of this class.
        And, we obtain the relevant information beforehand.
        :return:
            the size of ef dataset, the size of k/g dataset, the k/g dataset
        """
        len_ef = 69339
        len_k = 5830
        return len_ef, len_k

    def __len__(self):
        return len(self.task_dataset)

    def __getitem__(self, index):
        mat = self.task_dataset[index]
        data = self.convert(mat['structure'])
        target = self.task_dict.get(self.task)
        data.__setitem__('y', mat[target])
        return data

    @staticmethod
    def convert(structure) -> Data:
        """
        Take a pymatgen structure and convert it to a graph representation.
        This graph is a basic graph including atoms  and positions.
        :param structure: str.
        :return: Data instance.
        """
        structure = IStructure.from_str(structure, fmt='cif')
        x = torch.tensor(structure.atomic_numbers, dtype=torch.float)
        positions = torch.tensor(structure.cart_coords, dtype=torch.float)
        formula = structure.composition.formula
        cell = deepcopy(structure.lattice.matrix)
        cell = torch.Tensor(cell).view(1, 3, 3)
        data = Data(atomic_numbers=x, pos=positions, cell=cell, name=formula)
        return data

    def split(self, train_test_split: Tuple[List[int], List[int]]):
        """
        split the dataset into training or testing. If there is not the validation set, this function would split the
        original dataset into train and test. If the validation is needed, this function split the dataset into
        the train and validate set.
        :param train_test_split: a tuple containing two lists of integers, which is the index of dataset.
        :return: the training dataset and the test dataset.
        """
        train_index, test_index = train_test_split
        train_data = self.from_indices(train_index)
        test_data = self.from_indices(test_index)
        return train_data, test_data

    def from_indices(self, indices: List[int]):
        data_instance = MpDataset.__new__(MpDataset)
        attr = 'task_dataset'
        setattr(data_instance, attr, [getattr(self, attr)[index] for index in indices])
        var = 'task'
        setattr(data_instance, var, getattr(self, var))
        return data_instance


class MpGeometricDataset(InMemoryDataset):
    """
    This is a class for converting the open MP-crystals-2018.6.1 crystal data set into our model inputs.
    The dataset is from https://figshare.com/articles/dataset/Graphs_of_materials_project/7451351.
    We create a 'In memory datasets' according to the pytorch_geometric documentation.
    We process the data beforehand in order to accelerate the training speed. The processed filepaths must
    exist in order to skip processing.  In some cases, the 'root' folder is not correctly provided, and it
    will process the raw mp data again.
    """

    def __init__(self, root, task='ef', transform=None, pre_transform=None):
        """
        The dataset has been downloaded into local environment.
        args:
            root: a root folder which indicates where the dataset should be stored.;
            '': represents the current code dir. Otherwise, you should provide the processed file root path.
            transform: dynamically transforms the data object before accessing.
            pre_transofrm: applyies the transformation before saving the data objects to disk.


        """
        # extend class attributes in order to pass the variable during different files.
        self.task = task
        super(MpGeometricDataset, self).__init__(root, transform, pre_transform)
        # your processed_paths.
        # The absolute filepaths that must be present in order to skip processing.
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def task_dict(self):
        return dict(ef='formation_energy_per_atom', eg='band_gap', g='G', k='K')

    @staticmethod
    def load_mp_data():
        # This is a global variable. When run this file
        current_file_dir = os.path.dirname(__file__)
        json_file = os.path.normpath(os.path.join(current_file_dir, '../mp.2018.6.1.json'))
        print('loading raw data from {}'.format(json_file))
        try:
            mp_data = load_dataset(file_name=json_file)
            return mp_data
        except FileNotFoundError:
            print("The file 'mp.2018.6.1.json' does not exist! \nYou must download it and put it in the {} "
                  "before loading your raw data.".format(current_file_dir))

    @property
    def processed_file_names(self):
        """
        We load the k and g dataset by default.
        :return:
        """
        if self.task == 'ef':
            print('loading dataset of ef and eg.')
            return ['ef_mp_data.pt']
        elif self.task == 'egn':
            print('loading dataset of non-zero band gaps.')
            return ['egn_mp_data.pt']
        else:
            print('loading dataset of k and g.')
            return ['k_mp_data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.

        # data_list = [ ]
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]
        ef_data_list = []
        k_data_list = []
        egn_data_list = []
        mp_data = self.load_mp_data()
        for mat in mp_data:
            data = self.convert(mat['structure'])
            k_target = self.task_dict.get('k')
            if k_target in mat.keys():
                data.__setitem__('k', mat[self.task_dict.get('k')])
                data.__setitem__('g', mat[self.task_dict.get('g')])
                k_data_list.append(data)
            data.__setitem__('ef', mat[self.task_dict.get('ef')])
            band_gap = mat[self.task_dict.get('eg')]
            data.__setitem__('eg', band_gap)
            if band_gap > 0:
                data.__setitem__('egn', band_gap)
                egn_data_list.append(data)
            ef_data_list.append(data)
        if self.task == 'ef':
            data, slices = self.collate(ef_data_list)
        elif self.task == 'egn':
            data, slices = self.collate(egn_data_list)
        else:
            data, slices = self.collate(k_data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def convert(structure) -> Data:
        """
        Take a pymatgen structure and convert it to a graph representation.
        This graph is a basic graph including atoms  and positions.
        :param structure: str.
        :return: Data instance.
        """
        structure = IStructure.from_str(structure, fmt='cif')
        x = torch.tensor(structure.atomic_numbers, dtype=torch.float)
        positions = torch.tensor(structure.cart_coords, dtype=torch.float)
        formula = structure.composition.formula
        cell = deepcopy(structure.lattice.matrix)
        cell = torch.Tensor(cell).view(1, 3, 3)
        data = Data(atomic_numbers=x, pos=positions, cell=cell, name=formula)
        return data


class RIDataset(InMemoryDataset):
    """
    This is a class for converting the open MP-crystals-2018.6.1 crystal data set into our model inputs.
    The dataset is from https://figshare.com/articles/dataset/Graphs_of_materials_project/7451351.
    We create a 'In memory datasets' according to the pytorch_geometric documentation.
    We process the data beforehand in order to accelerate the training speed. The processed filepaths must
    exist in order to skip processing.  In some cases, the 'root' folder is not correctly provided, and it
    will process the raw mp data again.
    """

    def __init__(self, root, transform=None, pre_transform=None):
        """
        The dataset has been downloaded into local environment.
        args:
            root: a root folder which indicates where the dataset should be stored.;
            '': represents the current code dir. Otherwise, you should provide the processed file root path.
            transform: dynamically transforms the data object before accessing.
            pre_transofrm: applyies the transformation before saving the data objects to disk.


        """
        # extend class attributes in order to pass the variable during different files.
        super(RIDataset, self).__init__(root, transform, pre_transform)
        # your processed_paths.
        # The absolute filepaths that must be present in order to skip processing.
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def task_dict(self):
        return dict(ri='refractive_index', og='optical_gap', dg='direct_gap')

    @staticmethod
    def load_mp_data():
        # This is a global variable. When run this file
        current_file_dir = os.path.dirname(__file__)
        json_file = os.path.normpath(os.path.join(current_file_dir, './refractive_index.json'))
        print('loading raw data from {}'.format(json_file))
        mp_data = load_dataset(file_name=json_file)
        return mp_data

    @property
    def processed_file_names(self):
        """
        We load the k and g dataset by default.
        :return:
        """

        print('loading dataset of refractive index.')
        return ['ri_mp_data.pt']

    def download(self):
        pass

    def process(self):
        ri_data_list = []
        mp_data = self.load_mp_data()
        for mat in mp_data:
            data = self.convert(mat['structure'])
            # print(mat[self.task_dict.get('ri')])
            # print(type(mat[self.task_dict.get('ri')]))
            data.__setitem__('ri', float(mat[self.task_dict.get('ri')]))
            data.__setitem__('og', float(mat[self.task_dict.get('og')]))
            data.__setitem__('dg', float(mat[self.task_dict.get('dg')]))
            ri_data_list.append(data)
        data, slices = self.collate(ri_data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def convert(structure) -> Data:
        """
        Take a pymatgen structure and convert it to a graph representation.
        This graph is a basic graph including atoms  and positions.
        :param structure: str.
        :return: Data instance.
        """
        structure = IStructure.from_str(structure, fmt='cif')
        x = torch.tensor(structure.atomic_numbers, dtype=torch.float)
        positions = torch.tensor(structure.cart_coords, dtype=torch.float)
        formula = structure.composition.formula
        cell = deepcopy(structure.lattice.matrix)
        cell = torch.Tensor(cell).view(1, 3, 3)
        data = Data(atomic_numbers=x, pos=positions, cell=cell, name=formula)
        return data


if __name__ == '__main__':
    # a, b, c = get_len(mp_data)
    # print('The size of dataset are: \n Ef dataset:{}, K and G dataset: {}'.format(a, b))
    # dataset = MpDataset(data_set=mp_data, task='ef')
    # from sklearn.model_selection import train_test_split
    # splits = train_test_split(range(len(dataset)), test_size=0.2)
    # train, test = dataset.split(splits)
    # train_loader = MultiEpochsDataLoader(dataset=train, batch_size=256,num_workers=12, pin_memory=True, persistent_workers=True)
    # test_loader = DataLoader(dataset=test, batch_size=1)
    # # print(train_loader.dataset[0])
    # print(len(train_loader))
    # print(len(test_loader))
    #
    # tt = []
    # # if torch.cuda.is_available():
    # #     train_loader = CudaDataLoader(train_loader, device=0)
    # for e in range(3):
    #     start = time.time()
    #     for i, b in enumerate(train_loader):
    #         print(b.pos)
    #     end = time.time()
    #     dt = end - start
    #     tt.append(dt)
    # print(tt)

    # print(os.path.abspath(__file__))
    # mp_data = load_dataset(file_name='../mp.2018.6.1.json')
    # dataset = MpGeometricDataset(task='k', root='mp')
    # print(next(iter(dataset)))
    # from torch_geometric.loader import DataLoader
    #
    # loader = DataLoader(dataset=dataset, batch_size=2,
    #                     num_workers=4, pin_memory=True,
    #                     persistent_workers=True)
    # for batch in loader:
    #     print(batch.__getitem__('k'))

    # convert_n_in_mp()

    dataset = RIDataset(root='mp')
    loader = DataLoader(dataset=dataset, batch_size=1,
                        num_workers=4, pin_memory=True,
                        persistent_workers=True)
    for batch in loader:
        print(batch.__getitem__('ri'))

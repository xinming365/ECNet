import os
from datasets.Mp_dataset import MpGeometricDataset, RIDataset
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import SequentialSampler, Dataset, SubsetRandomSampler
from torch_geometric.data import InMemoryDataset
from typing import Tuple, Optional, Union


def assign_task_name(task):
    """
    Define the dataset name for convenience.
    It can be modified according to your own requirements.
    :param task:
    :return:
    """
    if len(task) == 1:  # single-task learning
        if 'egn' in task:
            task_name = 'egn'
        elif ('ef' in task) or ('eg' in task):
            task_name = 'ef'
        else:
            task_name = 'k'
    else:  # multi-task learning
        if 'egn' in task:
            task_name = 'egn'
        elif 'ef' in task:
            task_name = 'ef'
        else:
            task_name = 'k'
    return task_name


class DataSplit(object):
    """
    Randomly shuffle the materials and split them into training/validation/test dataset.
    """

    def __init__(self, ratios: str = "80/10/10", delim: str = "/", RANDOM_SEED=1454880):
        """
        Shuffle the materials according to the ratios.
        :param ratios: (str): ratios
        :param delim: (str): deliminators for separate ratios
        :param RANDOM_SEED:  (int) random seed
        """
        self.random_seed = RANDOM_SEED
        int_ratios = [float(i) for i in ratios.strip().split(delim)]
        self.ratios = [i / sum(int_ratios) for i in int_ratios]

    def split(self, mat_seq) -> Tuple:
        """
        Randomly split the mat_seq
        :param mat_seq: array-like
        :return:
            tuple containing train-validate-test split of inputs
        """
        seq = np.random.RandomState(seed=self.random_seed).permutation(mat_seq)
        n = len(seq)

        end_points = np.cumsum([int(n * i) for i in self.ratios[:-1]]).tolist()
        end_points = [0] + end_points + [n]
        return tuple(seq[i:j] for i, j in zip(end_points[:-1], end_points[1:]))


class LoadDataset(object):
    """
    Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    """

    def __init__(self, dataset: Union[InMemoryDataset, Dataset], batch_size: Optional[int] = 128,
                 num_workers: int = 4, pin_memory: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def load(self, fraction: float = 1, fraction_seed: int = None):
        n = len(self.dataset)
        tot_seq = range(n)

        assert 0 < fraction <= 1, 'the fraction must between 0 and 1'
        mat_size = int(n * fraction)
        if fraction_seed:
            tot_seq = np.random.RandomState(seed=fraction_seed).permutation(tot_seq)
        splitted_seq = self.split(tot_seq[:mat_size])
        train_loader, validate_loader, test_loader = [self.load_from_seq(i) for i in splitted_seq]
        return train_loader, validate_loader, test_loader

    def load_from_seq(self, mat_seq):
        data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                 sampler=SubsetRandomSampler(mat_seq),
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 persistent_workers=False)
        return data_loader

    def split(self, mat_seq, ratios: str = "80/10/10",
              delim: str = "/", random_seed: int = 1454880) -> Tuple:
        """
        Randomly split the mat_seq
        :param mat_seq: array-like
        :return:
            tuple containing train-validate-test split of inputs
        """
        seq = np.random.RandomState(seed=random_seed).permutation(mat_seq)
        n = len(seq)
        int_ratios = [float(i) for i in ratios.strip().split(delim)]
        ratios = [i / sum(int_ratios) for i in int_ratios]

        end_points = np.cumsum([int(n * i) for i in ratios[:-1]]).tolist()
        end_points = [0] + end_points + [n]
        return tuple(seq[i:j] for i, j in zip(end_points[:-1], end_points[1:]))


if __name__=='__main__':
    tasks = ['ef']
    dataset = MpGeometricDataset(task=tasks, root='./datasets/mp')
    train_loader, validate_loader, test_loader = LoadDataset(dataset=dataset).load(fraction=1)
    from datasets.HEA_dataset import HEADataset
    hea_dataset = HEADataset(poscar_dir='../HEA_Data/POSCARS', label_name='../HEA_Data/Out_labels/Database.xlsx')






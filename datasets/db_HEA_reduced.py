import os
from torch.utils.data import Dataset
from datasets.preprocessing import PoscarToGraph
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from ase.io.extxyz import read_xyz
from ase.io import read, write, iread
import torch
from ase.atoms import Atoms


class DbHEA(Dataset):
    def __init__(self, xyz_dir):
        self.file_name = xyz_dir

    def __len__(self):
        # ag = iread(self.file_name)
        # i=0
        # for a in ag:
        #     i+=1
        i = 2329
        return i

    @classmethod
    def to_graph(cls, atoms: Atoms):
        positions = torch.tensor(atoms.positions, dtype=torch.float)
        cell = torch.Tensor(atoms.cell[:]).view(1, 3, 3)
        x = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.float)
        num_atoms = x.shape[0]
        energy = atoms.get_potential_energy()
        forces = atoms.arrays['force']
        forces = torch.tensor(forces, dtype=torch.float)
        data = Data(atomic_numbers=x, pos=positions, num_atoms=num_atoms, cell=cell,
                    energy=energy, force=forces)
        return data

    def __getitem__(self, index):
        with open(self.file_name, mode='r') as fo:
            ag = read_xyz(fo, index=index)
            for atoms_object in ag:
                data = self.to_graph(atoms_object)
        return data


def get_std(array):
    m = -572.59630848
    std = 443.6452794032273
    return (array - m) / std


def inverse_std(array):
    m = -572.59630848
    std = 443.6452794032273
    return array * std + m


if __name__ == '__main__':
    import numpy as np

    data = DbHEA(xyz_dir='db_HEA_reduced.xyz')
    print(len(data))
    data_i = data.__getitem__(5)
    print(data_i.pos)
    print(data_i.force)
    print(DbHEA(xyz_dir='db_HEA_reduced.xyz').__len__())
    train_loader = DataLoader(dataset=data, batch_size=1)
    print(train_loader.dataset[0])
    total_energy = []

    for i, b in enumerate(train_loader):
        # print(b)
        print(b.energy)
        total_energy.append(b.energy.item())
        # print(b.force)
        # print(b.num_atoms)

    mean = np.mean(total_energy)
    std = np.std(total_energy)
    print(mean, std)

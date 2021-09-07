import os
from torch.utils.data import Dataset
from preprocessing import PoscarToGraph
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from torch_geometric.datasets import tu_dataset, qm9
from torch_geometric.data import InMemoryDataset
import torch
import torch_geometric


class MdDataset(Dataset):
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.file_list = os.listdir(self.dir_name)
        pass

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file = self.file_list[index]
        file_path = os.path.join(self.dir_name, file)
        POSCAR = os.path.join(file_path, 'POSCAR')
        vasprun = os.path.join(file_path, 'vasprun.xml')
        p2g = PoscarToGraph(r_forces=True, r_energy=True)
        data = p2g.to_graph(filename=POSCAR, vasprun=vasprun)
        return data


class MdDatasetGeometric(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MdDatasetGeometric, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        download_url(url, self.raw_dir)
        ...

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__=='__main__':
    md = MdDataset('../job')
    # train_loader=DataLoader(dataset=md, batch_size=2, collate_fn=None)
    train_loader=  DataLoader(dataset=md, batch_size=2)

    print(train_loader.dataset[0])
    print(train_loader.dataset[0].num_atoms)
    for i , b in enumerate(train_loader):
        print(b)
        print(b[0])
        print(b.num_atoms)
    # print(a)



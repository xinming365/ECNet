import numpy as np
from pymatgen.io.vasp import Poscar
import torch
from torch_geometric.data import Data
from lxml import etree
from ase.calculators.emt import EMT
import xlrd
import pandas as pd
import os
import shutil
from pymatgen.core import Structure
import copy

class ReadExcel:
    def __init__(self, out_file=None):
        self.out_file = out_file

    def read_excel(self, in_file):
        """
        Read the excel file. It will drop the empty values in the original excel.
        If the output filename is provided, the processed file will be saved in the out_file by the Pandas.
        :param in_file: str, the path name of the input file, such as '../HEA_Data/Quinary.xlsx'
        :return: ndarray.
        """
        sheet_nums = 2
        out = []
        for i in range(sheet_nums):
            array = self._read_sheet(in_file=in_file, sheet_name=i)
            rows, cols = array.shape
            for index in range(rows):
                row = array[index]
                if 'FCC' in row or 'BCC' in row:
                    pass
                elif '*' in row:
                    pass
                elif 'Etot (eV/atom)' in row or 'structures' in row:
                    pass
                elif 'Structure' in row or 'Etot eV' in row:
                    pass
                else:
                    row = [e for e in row if e != '']
                    if i == 0:
                        row[0] = row[0] + '_sqsfcc'
                    else:
                        row[0] = row[0] + '_sqsbcc'
                    out.append(row)
        df = pd.DataFrame(out)
        df = df.dropna(axis=0, how='any')
        col_names = ['structures', 'Etot (eV/atom)', 'Etot (eV)', 'Emix (eV/atom)', 'Eform (eV/atom)',
                     'Ms (emu/g)', 'Ms (mub/atom)', 'mb (mub/cell)', 'rmsd (\\AA)', 'V0 (A3/atom)']
        df.columns = col_names
        out_file = self.out_file
        if out_file:
            df.to_excel(excel_writer=out_file)
        a = df.to_numpy()
        return a

    def read_excel_hcp(self, in_file, sheet_nums=1):
        """
        Read the excel file. It will drop the empty values in the original excel.
        If the output filename is provided, the processed file will be saved in the out_file by the Pandas.
        :param in_file: str, the path name of the input file, such as '../HEA_Data/Quinary.xlsx'
        :return: ndarray.
        """
        out = []
        for i in range(sheet_nums):
            array = self._read_sheet(in_file=in_file, sheet_name=i)
            rows, cols = array.shape
            for index in range(rows):
                row = array[index]
                if 'HCP' in row or 'BCC' in row:
                    pass
                elif '*' in row:
                    pass
                elif 'Etot (eV/atom)' in row or 'structures' in row:
                    pass
                elif 'Structure' in row or 'Etot eV' in row:
                    pass
                else:
                    row = [e for e in row if e != '']
                    row[0] = row[0] + '_sqshcp'
                    out.append(row)
        df = pd.DataFrame(out)
        df = df.dropna(axis=0, how='any')
        col_names = ['structures', 'Etot (eV/atom)', 'Etot (eV)', 'Eform (eV/atom)',
                     'Ms (emu/g)', 'Ms (mub/atom)', 'mb (mub/cell)', 'rmsd (\\AA)', 'V0 (A3/atom)']
        df.columns = col_names
        out_file = self.out_file
        if out_file:
            df.to_excel(excel_writer=out_file)
        a = df.to_numpy()
        return a

    def read_all(self):
        """
          All the binary, ternary, quaternary, and quinary files are merged into one excel file,
          called the Database.xlsx.

          In this function, the output filename is provided. So, the processed file will be saved automatically,
          according to the 'read_excel' function.
          :return: ndarray
          """
        in_file_list = ['Binary.xlsx', 'Ternary.xlsx', 'Quaternary.xlsx', 'Quinary.xlsx']
        in_file_path = '../HEA_Data/'
        out_file = '../HEA_Data/Out_labels/Database.xlsx'
        out = []
        for name in in_file_list:
            in_fie = os.path.join(in_file_path, name)
            a = self.read_excel(in_file=in_fie)
            out.append(a)
        out = np.vstack(out)
        df = pd.DataFrame(out)
        col_names = ['structures', 'Etot (eV/atom)', 'Etot (eV)', 'Emix (eV/atom)', 'Eform (eV/atom)',
                     'Ms (emu/g)', 'Ms (mub/atom)', 'mb (mub/cell)', 'rmsd (\\AA)', 'V0 (A3/atom)']
        df.columns = col_names
        if out_file:
            df.to_excel(excel_writer=out_file)
        return df.values

    def _read_sheet(self, in_file, sheet_name):
        wb = pd.read_excel(io=in_file, sheet_name=sheet_name, engine='openpyxl')
        wb = wb.dropna(axis=0, how='all')
        wb = wb.dropna(axis=1, how='all')
        wb = wb.fillna(value='')
        array = wb.to_numpy()
        return array


class DataSegment:
    def __init__(self, base_dir):
        # For example, the base_dir is '../HEA_Data/Out_labels'.
        self.base_dir = base_dir

    def segment(self, types):
        """
        Utilize the low-component data to predict the high-component data.

        :param types: str.
        :return:
        """

        if types == 0:
            # binary + ternary
            in_file_list = ['Binary.xlsx', 'Ternary.xlsx']
            in_file_path = self.base_dir
            out_file = os.path.join(self.base_dir, 'B+T.xlsx')
        elif types == 1:
            # binary + ternary + quartenary
            in_file_list = ['Binary.xlsx', 'Ternary.xlsx', 'Quaternary.xlsx']
            in_file_path = self.base_dir
            out_file = os.path.join(self.base_dir, 'B+T+Qua.xlsx')
        elif types == 2:
            # quartenary + quinary
            in_file_list = ['Quaternary.xlsx', 'Quinary.xlsx']
            in_file_path = self.base_dir
            out_file = os.path.join(self.base_dir, 'Qua+Qui.xlsx')
        else:
            print('Not implemented!')
            return 0

        out = []
        for name in in_file_list:
            in_fie = os.path.join(in_file_path, name)
            a = self._read_excel(in_file=in_fie)
            out.append(a)
        out = np.vstack(out)
        df = pd.DataFrame(out)
        col_names = ['structures', 'Etot (eV/atom)', 'Etot (eV)', 'Emix (eV/atom)', 'Eform (eV/atom)',
                     'Ms (emu/g)', 'Ms (mub/atom)', 'mb (mub/cell)', 'rmsd (\\AA)', 'V0 (A3/atom)']
        df.columns = col_names
        df.describe()
        if out_file:
            df.to_excel(excel_writer=out_file)
        return df.values

    def train_test_split(self, train_ratio=0.8):
        """
        Split the whole dataset into the training data and testing data. The train_ratio controls
        the portion of the train data.

        :param types: float.
        :return:
        """

        filename = 'Database.xlsx'
        seed_nums = 30
        save_data = True
        in_file = os.path.join(self.base_dir, filename)
        a = self._read_excel(in_file=in_file)
        length = a.shape[0]
        train_nums = int(np.ceil(length * train_ratio))
        np.random.seed(seed_nums)
        random_a = np.random.permutation(a)
        train_data = random_a[:train_nums, :]
        test_data = random_a[train_nums:, :]
        if save_data:
            col_names = ['structures', 'Etot (eV/atom)', 'Etot (eV)', 'Emix (eV/atom)', 'Eform (eV/atom)',
                         'Ms (emu/g)', 'Ms (mub/atom)', 'mb (mub/cell)', 'rmsd (\\AA)', 'V0 (A3/atom)']

            df_train = pd.DataFrame(train_data)
            df_test = pd.DataFrame(test_data)

            df_train.columns = col_names
            df_test.columns = col_names

            df_train.to_excel(excel_writer=os.path.join(self.base_dir, 'train.xlsx'))
            df_test.to_excel(excel_writer=os.path.join(self.base_dir, 'test.xlsx'))
        return train_data, test_data

    def describe(self, filename):
        """
        Generate descriptive statistics.

        Descriptive statistics include shape of dataset.

        :param filename: str
        :return:
        """
        filename = os.path.join(self.base_dir, filename)
        a = self._read_excel(in_file=filename)
        rows, cols = a.shape
        print('Numbers of data: {}, properties:{}'.format(rows, cols - 1))
        names = a[:, 0]
        b_nums, t_nums, qua_nums, qui_nums, others = 0, 0, 0, 0, 0
        b_list, t_list, qua_list, qui_list, others_list = [], [], [], [], []
        for name in names:
            #     print(name)
            sys = name.split('_')[0]
            struct = name.split('_')[1]
            comp = mg.Composition(sys)
            chemical_systems = comp.chemical_system
            n_atoms = len(chemical_systems.split('-'))
            if n_atoms == 2:
                b_nums = b_nums + 1
                b_list.append(chemical_systems)
            elif n_atoms == 3:
                t_nums = t_nums + 1
                t_list.append(chemical_systems)
            elif n_atoms == 4:
                qua_nums = qua_nums + 1
                qua_list.append(chemical_systems)
            elif n_atoms == 5:
                qui_nums = qui_nums + 1
                qui_list.append(chemical_systems)
            else:
                others = others + 1
                others_list.append(chemical_systems)
                print(sys, struct)

        def g(list_sys):
            # count the chemical systems in various components alloys.
            eles = []
            for ele in list_sys:
                tmp = ele.split('-')
                eles.extend(tmp)
            return set(eles)

        # print(qui_list)
        b_eles, t_eles, qua_eles, qui_eles, others_eles = g(b_list), g(t_list), g(qua_list), g(qui_list), g(others_list)

        sys_dict = dict(zip(['binary', 'ternary', 'quarternary', 'quinary', 'others'],
                            [b_list, t_list, qua_list, qui_list, others_list]))
        print('The number of different systems:\nBinary:{}, Ternary:{}, Quaternary:{}, Quinary:{}, others:{}\n'.format(
            b_nums, t_nums, qua_nums, qui_nums, others
        ))
        print(
            'The specific chemical elements of different systems:\nBinary:{}, Ternary:{}, Quaternary:{}, Quinary:{}, others:{}\n'.format(
                b_eles, t_eles, qua_eles, qui_eles, others_eles
            ))
        sums = b_nums + t_nums + qua_nums + qui_nums + others
        print('The total number of the systems is {}.\n'.format(sums))
        return a

    def _read_excel(self, in_file):
        """
        Read the excel file using the Pandas.
        :param in_file: str, the path name of the input file, such as '../HEA_Data/Quinary.xlsx'
        :return: ndarray.
        """
        wb = pd.read_excel(io=in_file, sheet_name=0, engine='openpyxl', index_col=0)
        array = wb.to_numpy()
        return array

    def excel2csv(self, in_file, out_file='out.csv'):
        """
        Read the excel file using the Pandas and convert it into csv format.
        :param in_file:  str.
        :return:
        """
        wb = pd.read_excel(io=in_file, sheet_name=0, engine='openpyxl', index_col=0)
        # pd.DataFrame
        n = wb.shape[0]

        phase = []
        for i in range(n):
            name = wb['structures'][i]
            if 'fcc' in name:
                phase.append('fcc')
            elif 'bcc' in name:
                phase.append('bcc')
            else:
                phase.append('others')
        wb['phase'] = phase
        wb.to_csv(out_file)


def convert():
    """
    Convert the original labeled data into standard data format. The original data is collated by Dr. xx one by one.
    We directly convert the in_file into the out_file.
    :return:
        nothing.
    """
    in_file = '../HEA_Data/Quinary.xlsx'
    out_file = '../HEA_Data/Out_labels/Quinary.xlsx'
    re = ReadExcel(out_file=out_file)
    a = re.read_excel(in_file)
    print(a)


def collate_poscars():
    """
    We collate different excel files including binary, ternary, quaternary, and quinary files.
    Put them together to copy them into one same directory.
    :return:
        Nothing
    """
    # path_names = ['Binary_POSCAR_Files', 'Ternary_POSCAR_Files',
    #               'Quaternary_POSCAR_Files', 'Quinary_POSCAR_Files']
    path_names = ['Ternary_sqshcp_POSCAR_Files']
    file_path = '../HEA_Data/'
    out_file = os.path.join(file_path, 'POSCARS')

    for name in path_names:
        dir = os.path.join(file_path, name)
        files = os.listdir(dir)
        for file in files:
            src = os.path.join(dir, file)
            name_list = file.split('_')
            dst = name_list[1] + '_' + name_list[2]
            dst = os.path.join(out_file, dst)
            shutil.copy(src=src, dst=dst)
    print('Done')


def to_graph_from_structure(struct):
    positions = torch.tensor(struct.cart_coords, dtype=torch.float)
    x = torch.tensor(struct.atomic_numbers, dtype=torch.float)
    cell = torch.Tensor(struct.lattice.matrix).view(1, 3, 3)
    data = Data(atomic_numbers=x, pos=positions, cell=cell)
    return data


class PoscarToGraph:
    def __init__(self, radius=6, max_neigh=200):
        self.radius = radius
        self.max_neigh = max_neigh

    def _read_struct(self, struct):
        """
        Args:
            struct : Strucutre object in pymatgen.core.structure module
        Returns:
            graph_nodes, graph_edges
        """

        center_indices, points_indices, offset_vectors, distances = \
            struct.get_neighbor_list(r=self.radius, numerical_tol=1e-8, exclude_self=True)

        # Resitrict the maximum neighbors according to the distances, where the neighbors must be smaller than max_neigh
        natoms = np.max(center_indices)
        final_idx = []
        for i in range(natoms):
            idx = (i == center_indices).nonzero()[0]
            idx_sorted = np.argsort(distances[idx])
            idx_to_maxneigh = idx_sorted[:self.max_neigh]
            final_idx.append(idx[idx_to_maxneigh])
        final_idx = np.concatenate(final_idx, axis=0)

        center_indices = center_indices[final_idx]
        points_indices = points_indices[final_idx]
        offset_vectors = offset_vectors[final_idx]
        distances = distances[final_idx]

        # Remove distances smaller than a tolerance, because of the returned self atoms in certain cases.
        nonzero_idx = np.where(distances >= 1e-6)[0]
        center_indices = center_indices[nonzero_idx]
        points_indices = points_indices[nonzero_idx]
        offset_vectors = offset_vectors[nonzero_idx]
        distances = distances[nonzero_idx]

        edge_index = torch.tensor(np.vstack((center_indices, points_indices)), dtype=torch.long)
        edge_distances = torch.tensor(distances, dtype=torch.float)
        cell_offsets = torch.tensor(offset_vectors, dtype=torch.long)
        return edge_index, edge_distances, cell_offsets

    def to_graph(self, filename):
        """
        Args:
            filename (str): File name containing Poscar data.
        Returns:
            graph_nodes, graph_edges
        """
        poscar = Poscar.from_file(filename, check_for_POTCAR=False, read_velocities=True)
        struct = poscar.structure
        edge_index, edge_distances, cell_offsets = self._read_struct(struct)
        struct = poscar.structure
        positions = torch.tensor(struct.cart_coords, dtype=torch.float)
        x = torch.tensor(struct.atomic_numbers, dtype=torch.float)
        num_atoms = x.shape[0]
        cell = torch.Tensor(struct.lattice.matrix).view(1, 3, 3)

        data = Data(atomic_numbers=x, edge_index=edge_index, edge_distances=edge_distances,
                    pos=positions, num_atoms=num_atoms, cell=cell, cell_offsets=cell_offsets)

        n_index = data.edge_index[1, :]
        data.neighbors = torch.tensor(n_index.shape[0], dtype=torch.long)
        return data

    def to_graph_from_dict(self, d):
        struct = Structure.from_dict(d)
        positions = torch.tensor(struct.cart_coords, dtype=torch.float)
        x = torch.tensor(struct.atomic_numbers, dtype=torch.float)
        num_atoms = x.shape[0]
        cell = copy.copy(struct.lattice.matrix)
        cell = torch.Tensor(cell).view(1, 3, 3)
        edge_index, edge_distances, cell_offsets = self._read_struct(struct)
        n_index = edge_index[1, :]
        neighbors = torch.tensor(n_index.shape[0], dtype=torch.long)
        data = Data(atomic_numbers=x, edge_index=edge_index, edge_distances=edge_distances,
                    pos=positions, num_atoms=num_atoms, cell=cell, cell_offsets=cell_offsets,
                    neighbors=neighbors)
        return data



if __name__ == '__main__':
    # re = ReadExcel(out_file=None)
    # a = re.read_all()
    # print(a)
    ### Segment the hea data into different combinations
    ds = DataSegment('../HEA_Data/Out_labels')
    ds.excel2csv(in_file='../HEA_Data/Out_labels/Database.xlsx', out_file='../HEA_Data/Out_labels/Database.csv')

    # ds.segment(types=0)

    # ### get the processed hcp dataset
    # re = ReadExcel(out_file='../HEA_Data/Out_labels/Ternary_hcp.xlsx')
    # re.read_excel_hcp(in_file='../HEA_Data/Ternary_sqshcp.xlsx')
    ### collect the poscars and copy them into a general directory.
    # collate_poscars()



    # ####  Test a single materials.
    # pg = PoscarToGraph(radius=6, max_neigh=200)
    # # e=pg._read_energy('./CsBaCl3/job-0001/vasprun.xml')
    # # print(e)
    # data = pg.to_graph('../HEA_Data/Binary_POSCAR_Files/POSCAR_Co3Cr_sqsbcc')
    # edge_index = data.edge_index
    # positions = data.pos
    # print(positions)
    # print(torch.max(edge_index[0, :]), torch.max(edge_index[1, :]))
    # # print(edge_index[:,200:])
    # print(data)

import os
import warnings
import os.path as osp
from math import pi as PI

import ase
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList, BatchNorm1d, ELU
import numpy as np

from torch_scatter import scatter
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn import radius_graph, MessagePassing
from utils.registry import registry
from utils.utility import get_pbc_distances, radius_graph_pbc

try:
    import schnetpack as spk
except ImportError:
    spk = None

qm9_target_dict = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


@registry.register_models('heanet')
class HeaNet(torch.nn.Module):
    r"""This is the heanet designed for the high entropy alloy systems.

    """

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_interactions=6, num_gaussians=50, cutoff=10.0,
                 readout='add', dipole=False, mean=None, std=None,
                 atomref=None, num_tasks=1, tower_h1=128, tower_h2=64,
                 use_pbc=False, n_seq=None):
        super(HeaNet, self).__init__()

        assert readout in ['add', 'sum', 'mean']

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None
        self.num_tasks = num_tasks
        self.tower_h1 = tower_h1
        self.tower_h2 = tower_h2
        self.use_pbc = use_pbc
        self.n_seq = n_seq
        self.tower_layers = ModuleList()
        self.tower_heads = ModuleList()
        self.task_heads = ModuleList()
        self.tower_lin1 = Linear(self.hidden_channels, self.tower_h1)
        self.tower_lin2 = Linear(self.tower_h1, self.tower_h2)
        for _ in range(self.num_tasks):
            tower = Sequential(
                self.tower_lin1,
                ShiftedSoftplus(),
                # BatchNorm1d(self.tower_h1,affine=False),
                self.tower_lin2,
                ShiftedSoftplus(),
                # BatchNorm1d(self.tower_h2, affine=False),
                Linear(self.tower_h2, 1)
            )
            self.tower_layers.append(tower)
        if n_seq is not None:
            for _ in range(self.num_tasks):
                task = Sequential(
                    self.tower_lin1,
                    BatchNorm1d(self.tower_h1, affine=False),
                    ELU(),
                    Linear(self.tower_h1, self.tower_h1),
                    ELU(),
                    BatchNorm1d(self.tower_h1, affine=False),
                )
                self.task_heads.append(task)
            for _ in range(self.num_tasks * self.n_seq):
                head = Sequential(
                    self.tower_lin2,
                    BatchNorm1d(self.tower_h2, affine=False),
                    ELU(),
                    Linear(self.tower_h2, 1)
                )
                self.tower_heads.append(head)
        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = Embedding(100, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        # Initialize the batch tensor.
        batch_attr = torch.zeros_like(z) if batch is None else batch.batch

        # get the unique elements according to the batch.
        batch_size = batch_attr.unique().shape[0]
        # batch_c_list is the middle variable, and the batch_c is the concatenated variable.
        batch_c_list = []
        batch_c_index = []
        b_components_list = []
        for b in range(batch_size):
            b_index = torch.nonzero(batch_attr == b)
            b_z = z[b_index]
            b_components = b_z.unique()
            idx_comp = []
            for element in b_components:
                element_index = torch.nonzero(b_z == element.item(), as_tuple=True)
                idx_comp.append(b_index[element_index])
            batch_c_index.append(idx_comp)
            batch_c_list.append(torch.ones_like(input=b_components) * b)
            b_components_list.append(b_components)
        batch_c = torch.cat(tensors=batch_c_list, dim=0)
        batch_components = torch.cat(tensors=b_components_list, dim=0)
        # components = z.unique()
        # batch_c = torch.zeros_like(components)
        # idx_comp = []
        # for i in components:
        #     idx = np.where(z.cpu() == i.item())[0]
        #     idx_comp.append(idx)
        components_info = {'batch_c': batch_c, 'batch_c_index': batch_c_index, 'batch_components': batch_components}
        types = self.embedding(batch_components)
        h = self.embedding(z)

        if self.use_pbc:
            print('Using the periodic boundary condition.')
            out = get_pbc_distances(pos, batch.edge_index, batch.cell,
                                    batch.cell_offsets, batch.neighbors)
            edge_index = out['edge_index']
            edge_weight = out['distances']
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch_attr)
            row, col = edge_index
            edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            types = types + interaction(h, components_info, edge_index, edge_weight, edge_attr)

        if self.num_tasks > 1 or (not self.n_seq):  # multi-task learning
            if not self.n_seq:
                outs = []
                for i in range(self.num_tasks):
                    out_i = self.tower_layers[i](types)
                    out_i = scatter(out_i, batch_c, dim=0, reduce=self.readout)
                    outs.append(out_i)
                out = outs

            else:
                out = []
                for i in range(self.num_tasks):
                    out_i = self.task_heads[i](types)
                    outs = []
                    for seq in range(self.n_seq):
                        out_seq = self.tower_heads[i * self.n_seq + seq](out_i)
                        out_seq = scatter(out_seq, batch_c, dim=0, reduce=self.readout)
                        outs.append(out_seq.squeeze())
                    outs = torch.concat(outs, dim=0)
                    out.append(outs)
        else:
            h = self.lin1(types)
            h = self.act(h)
            h = self.lin2(h)
            out = [scatter(h, batch_c, dim=0, reduce=self.readout)]  # to keep the same data type in the output.
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, idx_comp, edge_index, edge_weight, edge_attr):
        x = self.conv(x, idx_comp, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, idx_comp, edge_index, edge_weight, edge_attr):
        """
        E is the number of edges. N is the number of nodes.

        :param x: x has shape of [N, in_channels]; where N is the number of nodes.
        :param idx_comp: list. index of the specific component.
        :param edge_index: edge_index has shape of [2, E]
        :param edge_weight:
        :param edge_attr:
        :return:
        """
        # C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        epsilon = 1e-10
        C = self.cutoff / (epsilon + edge_weight.pow(2)) - 1
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        # collate the information of the specific materials in the idx_comp.
        batch_c_index = idx_comp['batch_c_index']
        x_merge = []
        for i in batch_c_index:
            n_comps = len(i)
            # comp_x is the information for the specific materials
            comp_x = []
            for j in range(n_comps):
                comp_x.append(x[i[j]])
            comp_x = [torch.mean(input=t, dim=0, ) for t in comp_x]
            comp_x = torch.stack(tensors=comp_x, dim=0)
            x_merge.append(comp_x)
        x_merge = torch.cat(tensors=x_merge, dim=0)
        x_merge = self.lin2(x_merge)
        return x_merge

    def message(self, x_j, W):
        # x_j has shape of [E, in_channels]
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

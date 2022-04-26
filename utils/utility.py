"""
The get_pbc_distances and radius_graph_pbc come from ocp-models.
https://github.com/Open-Catalyst-Project/ocp/blob/master/ocpmodels/common/utils.py
"""

from sklearn.model_selection import StratifiedKFold as KFold
import torch
from utils.registry import registry, setup_imports

RANDOM_SEED = 12445


def kfold_splits(X, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    kf_splits = kf.split(X)
    return kf_splits


class DataTransformer(object):
    """
    Transform targets by scaling or standardization.

    When the trans_type='scaling', the transformation is given by:
        Y_scaled=Y*scaling

    When the trans_type='log', the transformation is given by:
        Y_scaled=log(Y)

    When the trans_type='normalize', the transformation is given by:
        Y_scaled= (Y-u)/s

    where 'u' is the mean of the training samples and 's' is the satndard
    deviation of the training samples.
    """

    def __init__(self):
        self.scale_parameter = 1000
        self.is_recover = True
        self.mean, self.std = 0, 1  # initialize the variables

    def linear_scaling(self, tensor, is_recover):
        if is_recover:
            return tensor / self.scale_parameter
        else:
            return tensor * self.scale_parameter

    def log_scaling(self, tensor, is_recover):
        if is_recover:
            return torch.pow(10, tensor)
        else:
            return torch.log10(tensor)

    def init_normalize(self, mean, std):
        device  = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = mean.to(device)
        self.std = std.to(device)


    def normalize(self, tensor, is_recover):
        if is_recover:
            return tensor * self.std + self.mean
        else:
            return (tensor - self.mean) / self.std

    def transform(self, trans_type, tensor, inverse_trans=False) -> object:
        """
        A general method to transform a dataset. According to the 'trans_type',
        this method will perform scaling or normalize operations.

        :param trans_type: str. Actually, we only support 'scaling', 'log10', and 'normalize' operations.
        if the a certain 'trans_type' is not implemented in this method, we will return the original tensor.
        :param tensor:
        :param inverse_trans: boolean
        :return: transformed data
        """
        if trans_type == 'log':
            return self.log_scaling(tensor, inverse_trans)
        elif trans_type == 'scaling':
            return self.linear_scaling(tensor, inverse_trans)
        elif trans_type == 'n': # normalize
            return self.normalize(tensor, inverse_trans)
        else:
            return tensor


def load_model(model_name, hidden_channels=128, n_filters=64, n_interactions=3,
               n_gaussians=50, cutoff=10, num_tasks=1, tower_h1=128, tower_h2=64,
               use_pbc=False):
    """
    load the trained ML models given the model_name.
    It should be noted that the hyper parameters are assigned according to the specific trained hyper parameters.

    args:
        model_name: str. the name of the trained model.
        For example: './saved_models/ms_type0_300.pt'
    """
    # load the ML model.
    setup_imports()
    device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
    model = registry.get_model('heanet'
                               )(hidden_channels=hidden_channels,
                                 num_filters=n_filters,
                                 num_interactions=n_interactions,
                                 num_gaussians=n_gaussians,
                                 cutoff=cutoff,
                                 readout='add',
                                 dipole=False, mean=None, std=None,
                                 atomref=None,
                                 num_tasks=num_tasks,
                                 tower_h1=tower_h1,
                                 tower_h2=tower_h2,
                                 use_pbc=use_pbc,
                                 )
    # load parameters of trained model
    model_state = torch.load(model_name, map_location=device)
    model.load_state_dict(model_state)
    return model


def get_pbc_distances(
        pos,
        edge_index,
        cell,
        cell_offsets,
        neighbors,
        return_offsets=False,
        return_distance_vec=False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances))[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


def radius_graph_pbc(data, radius, max_num_neighbors_threshold):
    device = data.pos.device
    batch_size = len(data.natoms)

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = (
            torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
            torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
            torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
                     atom_count_sqr // num_atoms_per_image_expand
             ) + index_offset_expand
    index2 = (
                     atom_count_sqr % num_atoms_per_image_expand
             ) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).
    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    rep_a1 = torch.ceil(radius * inv_min_dist_a1)

    cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    rep_a2 = torch.ceil(radius * inv_min_dist_a2)

    if radius >= 20:
        # Cutoff larger than the vacuum layer of 20A
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)
    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    unit_cell = torch.cat(torch.meshgrid(cells_per_dim), dim=-1).reshape(-1, 3)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image



from pymatgen.core import Structure
from utils.utility import load_model
import torch
from datasets.preprocessing import PoscarToGraph, to_graph_from_structure
import numpy as np
import matplotlib.pyplot as plt


def convert_data(filename):
    """
    the file name of the poscars
    args:
        filename: str. './HEA_Data/Binary_POSCAR_Files/POSCAR_Co3Cr_sqsfcc'
    """
    pg = PoscarToGraph(radius=10, max_neigh=200)
    data = pg.to_graph(filename)
    return data


# poscar_file = '../HEA_Data/Binary_POSCAR_Files/POSCAR_Co3Cr_sqsbcc'
# poscar_file = '../HEA_Data/Quaternary_POSCAR_Files/POSCAR_Fe15Ni15Co19Pd15_sqsbcc' # 测试集
poscar_file = '../HEA_Data/Quinary_POSCAR_Files/POSCAR_Fe36Ni31Co31Cr31Pd31_sqsbcc'
# model_name = '../saved_models_mtl_HEA/mtl_1_HEA_500_a0_best_ef_128.pt'
model_name = '../saved_models_mtl_HEA/mtl_1_HEA_500_a0_best_etot_128.pt'
model = load_model(model_name,
                   hidden_channels=128, n_filters=64, n_interactions=3,
                   n_gaussians=50, cutoff=10, num_tasks=1
                   )
model.eval()

output = []
volume = []
with torch.no_grad():
    for strain in np.arange(start=-0.10, stop=0.1, step=0.01):
        s = Structure.from_file(poscar_file)
        s.apply_strain(strain)
        data = to_graph_from_structure(s)
        print('The total volume is {}'.format(s.volume))
        print('Volume/atom is {}'.format(s.volume / s.num_sites))
        volume.append(s.volume / s.num_sites)
        out = model(data.atomic_numbers.long(), data.pos)
        output.append(out[0].item())
print(output)
print(volume)

plt.plot(volume, output, '--')
s0 = Structure.from_file(poscar_file)
# plt.plot(s0.volume/s0.num_sites, 0.17464, marker='o')
# plt.plot(s0.volume/s0.num_sites, -7.47937688, marker='o')
# plt.plot(s0.volume/s0.num_sites, -6.50242, marker='o')
plt.plot(s0.volume/s0.num_sites, -7.03073, marker='o')
plt.xlabel('Volume')
plt.ylabel('Etot (eV/atom)')

# plt.savefig('../fig/etot_volume.png',dpi=500)
plt.show()
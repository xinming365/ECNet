from icet import ClusterSpace
from icet.tools.structure_generation import (generate_sqs,
                                             generate_sqs_from_supercells,
                                             generate_sqs_by_enumeration,
                                             generate_target_structure)
import os
from ase.build import bulk


from icet.input_output.logging_tools import set_log_config
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
set_log_config(level='INFO')

# set the compositions.
def generate_fcc(elements=('Fe', 'Co', 'Mn')):
    # Cu element with fcc structures. lattice constant is 3.6149.
    Fcc_example = bulk('Cu', crystalstructure='fcc', a=3.6, cubic=True)
    cs = ClusterSpace(structure=Fcc_example, cutoffs=[8.0, 4.0], chemical_symbols=list(elements))
    # print(cs)
    return cs


def write_poscars(atoms):
    formula = atoms.symbols.get_chemical_formula()
    filename = formula + '_sqs' + 'fcc'
    poscar_dir = './SQS_ternary_POSCAR_Files'
    os.makedirs(poscar_dir, exist_ok=True)
    filename = os.path.join(poscar_dir, filename)
    sqs_struct = AseAtomsAdaptor.get_structure(atoms)
    sqs_poscar = Poscar(sqs_struct)
    sqs_poscar.write_file(filename=filename)


def generate_sqs_fcc(cluster_space, ternary_compositions, natoms=32, write=True,  elements=('Fe', 'Co', 'Mn')):
    x, y,  z = ternary_compositions
    target_concentrations = {elements[0]:x, elements[1]:y, elements[2]:z}
    print(target_concentrations)
    sqs = generate_sqs(cluster_space=cluster_space,
                max_size=natoms,
                target_concentrations=target_concentrations)
    # print('Cluster vector of generated structure:', cluster_space.get_cluster_vector(sqs))
    if write:
        write_poscars(sqs)
    return sqs

def get_poscars(systems):
    # 40--780; 20-190
    numbers = 20  # 80
    total = (numbers-1)*numbers/2
    elements = ['Fe', 'Co', 'Mn', 'Cr', 'Pd', 'Ni']
    x_compositions = [i / numbers for i in range(1, numbers)]
    y_compositions = [i / numbers for i in range(1, numbers)]
    cs = generate_fcc(systems)
    cnt = 0
    for x in x_compositions:
        for y in y_compositions:
            if (x + y) <= 1:
                cnt+=1
                z = 1 - x - y
                z = 0 if z<1e-5 else z
                # print((x,y,z))
                generate_sqs_fcc(cs, [x, y, z], natoms=32, write=True, elements=systems)
                print('Generating the SQS structures in {}/{}'.format(cnt, total))
    # print(cnt)


def generate_quaternary_sqs():
    systems = ''
    pass

def generate_quinary_sqs():
    systems = [('Ni', 'Fe', 'Mn', 'Pd', 'Cr'), ('Ni', 'Fe', 'Mn', 'Pd', 'Co'), ('Ni', 'Fe', 'Mn', 'Cr', 'Co'),
     ('Ni', 'Fe', 'Pd', 'Cr', 'Co'), ('Ni', 'Mn', 'Pd', 'Cr', 'Co'), ('Fe', 'Mn', 'Pd', 'Cr', 'Co')]
    a, b, c, d, e = 0, 0, 0, 0, 0
    numbers = 3
    x_compositions = [i / numbers for i in range(1, numbers)]
    y_compositions = [i / numbers for i in range(1, numbers)]
    cs = generate_fcc(systems)
    cnt = 0
    for x in x_compositions:
        for y in y_compositions:
            if (x + y) <= 1:
                cnt+=1
                z = 1 - x - y
                z = 0 if z<1e-5 else z
                # print((x,y,z))
                generate_sqs_fcc(cs, [x, y, z], natoms=32, write=True, elements=systems)
                print('Generating the SQS structures in {}/{}'.format(cnt, total))

if __name__=='__main__':
    # x = 0.25
    # y = 0.5
    # z = 1 - x - y
    get_poscars(systems=('Fe', 'Co', 'Mn'))

    # total_systems = [('Fe', 'Co', 'Mn'), ('Fe', 'Co', 'Cr'), ('Fe', 'Co', 'Pd'), ('Fe', 'Co', 'Ni'),
    #                  ('Co', 'Mn','Cr'), ('Co', 'Mn','Pd'), ('Co', 'Mn','Ni'),
    #                   ('Mn', 'Cr', 'Pd'), ('Mn', 'Cr', 'Ni'),
    #                   ('Cr', 'Pd', 'Ni') ]
    # systems = [('Fe', 'Co', 'Cr'), ('Fe', 'Co', 'Pd'), ('Fe', 'Co', 'Ni'), ('Co', 'Mn','Cr') ]
    # for sys in systems:
    #     get_poscars(systems=sys)



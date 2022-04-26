import math
import ternary
import matplotlib.pyplot as plt
from MLP_transfer import MLPTrainer, ElementFeatures
import numpy as np
from pymatgen.core import Composition
from collections import defaultdict

print("Version", ternary.__version__)


def plot_ternary_triangles_new(systems, data, figname=None, title=None):
    scale_simplex = 100
    figure, tax = ternary.figure(scale=scale_simplex)

    # figure.set_size_inches(10, 8)
    figure.set_size_inches(10, 8)

    # set Axis labels and Title
    fontsize = 25
    offset = 0.16

    # set title font size
    title_fs = 20
    tick_offset = 0.03
    tick_fs = 16
    #     title_offset = 0.3

    cb_kwargs = {"orientation": "vertical",
                 "fraction": 0.1,
                 "pad": 0.05,

                 }

    tax.right_axis_label('at% ' + systems[1], fontsize=fontsize, offset=offset)
    tax.bottom_axis_label('at% ' + systems[0], fontsize=fontsize, offset=offset)
    tax.left_axis_label('at% ' + systems[2], fontsize=fontsize, offset=offset)

    # draw gridlines
    tax.gridlines(color="k", multiple=10)

    # data[(10, 15)]=10
    # make heatmaps of data
    tax.heatmap(data=data, style="h", scale=scale_simplex, colorbar=True, cbar_tick_fs=tick_fs, cb_kwargs=cb_kwargs)
    #     cb = plt.colorbar()
    cbx = tax.get_axes()
    cbx.tick_params(labelsize=tick_fs)
    #     cbx.set_label(fontdict={'size': tick_fs})

    tax.boundary(linewidth=2.0)

    #     tax.set_title("Predictd {}".format(title), fontsize=title_fs)
    tax.ticks(axis='lbr', linewidth=1, multiple=20, tick_formats="%.1f", offset=tick_offset, fontsize=tick_fs)

    # remove the default matplotlib axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    if figname:
        plt.savefig(figname, dpi=500, bbox_inches='tight')
    tax.show()


class TernaryInference():
    def __init__(self, systems):
        self.systems = systems

    @staticmethod
    def get_ternary_compositions(elements=('Fe', 'Co', 'Mn')):
        # 40--780; 20-190
        numbers = 80  # 80
        total = (numbers - 1) * numbers / 2
        x_compositions = [i / numbers for i in range(numbers + 1)]
        y_compositions = [i / numbers for i in range(numbers + 1)]
        cnt = 0
        res = []
        scale = 100
        for x in x_compositions:
            for y in y_compositions:
                if (x + y) <= 1:
                    cnt += 1
                    z = 1 - x - y
                    z = np.around(z, 4)
                    # z = 0 if z < 1e-5 else z
                    #                     print((x,y,z))
                    if x == 1:
                        comp = elements[0]
                    elif y == 1:
                        comp = elements[1]
                    elif z == 1:
                        comp = elements[2]
                    else:
                        comp_list = []
                        for i, j in zip(elements, [x, y, z]):
                            if j != 0:
                                comp_list.append(i + str(np.round(scale * j, 1)))
                        # comp_list = [i+str(j) ]
                        comp = "".join(comp_list)
                    res.append(comp)
        #                     print('Generating the SQS structures in {}/{}'.format(cnt, total))
        return res

    @staticmethod
    def ternary_inference(formulas, inference_model='./saved_models_MLP/mlp_ms_fcc_300.pt',
                          element_features_file='./plot_figure/describers/ecmodel-ib3.json',
                          num_dim=5):
        test_target = 'ms'
        epochs = 300
        formulas = formulas
        # formulas, targets = MLPTrainer._process_hea_dataset(poscars_dir='../HEA_Data/POSCARS',
        #                                                     labels_name='../HEA_Data/Out_labels/Database.csv',
        #                                                     target=test_target,
        #                                                     phase='bcc')
        # print(formulas, targets)
        ef = ElementFeatures.from_file(filename=element_features_file)

        # where to reduce the dimension.

        ef.element_properties = ef.reduce_dimension(ef.element_properties, num_dim=num_dim)
        features = ef.convert_datasets(comps=formulas)

        trainer = MLPTrainer(input_dim=num_dim, is_embedding=True,
                             n_neurons=(128, 64))

        trainer.from_saved_model(inference_model)
        # feature = features[0]
        out = []
        for feature in features:
            out.extend(trainer.predict(feature=feature))

        # print(out)
        return out

    def generate_data(self, formulas, out):
        """
        Convert one materials into the feature vector based on the element features.
        :param mat_comp: str. Such as 'Fe2O3'
        :return:
        """
        data = {}
        systems = self.systems
        scale = 100
        element_dict = defaultdict(int)
        for idx, mat_comp in enumerate(formulas):
            comp = Composition(mat_comp)
            for i, j in comp._data.items():
                element_dict[str(i)] = j
            sum_weights = sum(element_dict.values())

            element_dict = {i: j / sum_weights for i, j in element_dict.items()}
            if not element_dict.get(systems[0]):
                x = 0
            else:
                x = scale * element_dict.get(systems[0])
            if not element_dict.get(systems[1]):
                y = 0
            else:
                y = scale * element_dict.get(systems[1])
            data[(x, y)] = out[idx]
        return data


if __name__=='__main__':
    # plot_triangles()

    systems = ('Fe', 'Co', 'Mn')
    ti = TernaryInference(systems=systems)
    formulas = ti.get_ternary_compositions(elements=systems)
    out = ti.ternary_inference(formulas=formulas,
                               # inference_model='../saved_models_MLP/exp-ib3-fcc-eform/mlp_eform_fcc_ratio1_seed5_epoch300.pt',
                               # inference_model='../saved_models_MLP/exp-ib3-bcc-eform/mlp_eform_bcc_ratio1_seed5_epoch300.pt',
                               inference_model='../saved_models_MLP/exp-ib3-fcc/mlp_ms_fcc_ratio1_seed5_epoch300.pt',
                               # inference_model='../saved_models_MLP/exp-ib3-bcc-ms/mlp_ms_bcc_ratio1_seed5_epoch300.pt',
                               # inference_model='../saved_models_MLP/mlp_ms_bcc_300.pt',
                               element_features_file='./describers/ecmodel-ib3.json',
                               num_dim=6)
    # print(formulas, out)
    data = ti.generate_data(formulas, out)
    # plot_ternary_triangles(systems= systems ,data=data, figname='./fig/ternary/FeCoMn_eform_fcc.png')
    plot_ternary_triangles_new(systems=systems, data=data, title='Eform (eV/atom)',
                               figname='../fig/ternary/FeCoMn_ms_fcc_new_test.png')

    # plot_ternary_triangles_new(systems=systems, data=data, title='Eform (eV/atom)',
    #                            figname='../fig/ternary/FeCoMn_eform_fcc_new.png')
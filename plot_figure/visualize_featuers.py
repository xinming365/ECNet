import copy
import json
import global_var as gv

gv.set_value('task', 'ef')  # loading dataset with 'ef' and 'eg'.
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
import matplotlib as mpl
from model import ECModel


def get_features(model, data):
    """
    Use the trained ML model to predict the features.

    args:
        model: Model instance. The trained model with loaded parameters.
        data: The torch_geometric.data.Data object. 

    returns:
        a dictionary containing keys and values of feature vectors (torch.Tensor).
        It should be noted that the feature vectors' shape is T*hidden_layers, 
        where T is the number of elemental species. For example, the 'POSCAR_Co3Cr_sqsfcc'
        have a feature vector of 2*128.
    """''
    activation = {}

    def get_activation(name):
        """
        save the output of the model into the dict of 'activation' when call the prediction.
        """

        def hook(module, input, output):
            #         print(output.detach().size())
            activation[name] = output.detach()

        return hook

    model.interactions[0].lin.register_forward_hook(get_activation('interactions1'))
    model.interactions[1].lin.register_forward_hook(get_activation('interactions2'))
    model.interactions[2].lin.register_forward_hook(get_activation('interactions3'))
    #     model.interactions[5].lin.register_forward_hook(get_activation('interactions6'))
    output = model(data.atomic_numbers.long(), data.pos)
    print(f'the models output is {output}')
    return activation


def visualize_features(activation1, activation2, y_axis_labels):
    feature1 = activation1['interactions1']
    feature2 = activation1['interactions2']
    feature3 = activation1['interactions3']
    feature4 = activation2['interactions3']
    # fig = plt.figure(figsize=(20, 8),dpi=300)
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 5), dpi=600)

    labels_fd = dict(fontsize=16,
                     color='r',
                     family='Times New Roman',
                     weight='light',
                     style='italic',
                     )
    labels_size = 14
    cb_dict = {}
    print(mpl.rcParams['font.size'])
    mpl.rcParams.update({'font.size': 20})
    print(mpl.rcParams['font.size'])
    ax1 = plt.subplot(4, 1, 1)
    ax1 = sns.heatmap(feature1.cpu().numpy(), cmap='vlag', yticklabels=y_axis_labels, ax=ax1, cbar=False)
    cb1 = ax1.figure.colorbar(ax1.collections[0])
    cb1.ax.tick_params(labelsize=labels_size)
    ax1.set_ylabel('IB-1', fontsize=labels_size)
    # ax1.set_axis_off()
    s1 = ' Outputs of three interaction blocks for SeO2 (mp-726)'
    ax1.text(x=-0.1, y=1.1, s='(a)',
             transform=ax1.transAxes, fontsize=labels_size)
    ax1.tick_params(axis='y', labelsize=labels_size)

    ax2 = plt.subplot(4, 1, 2)
    ax2 = sns.heatmap(feature2.cpu().numpy(), cmap='vlag', yticklabels=y_axis_labels, ax=ax2, cbar=False)
    cb2 = ax2.figure.colorbar(ax2.collections[0])
    cb2.ax.tick_params(labelsize=labels_size)
    ax2.set_ylabel('IB-2', fontsize=labels_size)
    ax2.tick_params(axis='y', labelsize=labels_size)
    # ax2.set_axis_off()

    ax3 = plt.subplot(4, 1, 3)
    ax3 = sns.heatmap(feature3.cpu().numpy(), cmap='vlag', yticklabels=y_axis_labels, ax=ax3, cbar=False)
    ax3.set_ylabel('IB-3', fontsize=labels_size)
    cb3 = ax3.figure.colorbar(ax3.collections[0])
    cb3.ax.tick_params(labelsize=labels_size)
    # ax3.set_xticks(np.linspace(start=0, stop=128, num=5), labels=[int(i) for i in np.linspace(start=0, stop=128, num=5)])
    # ax3.tick_params(axis='x', labelrotation=360, labelsize= labels_size)
    ax3.tick_params(axis='y', labelsize=labels_size)

    ax4 = plt.subplot(4, 1, 4)
    ax4 = sns.heatmap(feature4.cpu().numpy(), cmap='vlag', yticklabels=y_axis_labels, ax=ax4, cbar=False)
    ax4.set_ylabel('IB-3', fontsize=labels_size)
    cb3 = ax4.figure.colorbar(ax4.collections[0])
    cb3.ax.tick_params(labelsize=labels_size)
    ax4.set_xticks(np.linspace(start=0, stop=128, num=5),
                   labels=[int(i) for i in np.linspace(start=0, stop=128, num=5)])
    ax4.tick_params(axis='x', labelrotation=360, labelsize=labels_size)
    ax4.tick_params(axis='y', labelsize=labels_size)
    s2 = 'Outputs of 3rd interaction block for SeO2 (mp-559545)'
    ax4.text(x=-0.1, y=1.1, s='(b)',
             transform=ax4.transAxes, fontsize=labels_size)

    plt.subplots_adjust(left=0.1, bottom=None, right=0.99, top=None, wspace=None, hspace=0.25)

    # plt.savefig('./features2.png')


def SeO2_features_plot():
    """
    plot the feature vectors of some materials.
    :return:
    """
    ecmodel = ECModel(tasks=['ri', 'og', 'dg'], transform=['scaling', 'scaling', 'scaling'])
    model = ecmodel.load_model('../saved_models_mtl/mtl_3_mp_ri_og_dg_500_best.pt')
    device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')

    from pymatgen.ext.matproj import MPRester
    from datasets.Mp_dataset import RIDataset
    my_api_key = 'sKlJKcjcF2RWAJ7Fy7F'
    mpid = 'mp-726'  # SeO2
    mpid_2 = 'mp-559545'
    from torch_geometric.data import Data
    def load_mp_data(mpid) -> Data:
        with MPRester(my_api_key) as m:
            structure = m.get_structure_by_material_id(mpid)
            structure_str = structure.to('cif')
            data = RIDataset.convert(structure_str)
            data.to(device)
            print('{} materials with atomic numbers {}.'.format(data.name, data.atomic_numbers))
        return data

    data = load_mp_data(mpid=mpid)

    def compare_distance(pos1, pos2):
        delta = pos1 - pos2
        d_n = np.linalg.norm(delta, ord=2)
        pos1_n = np.linalg.norm(pos1, ord=2)
        pos2_n = np.linalg.norm(pos2, ord=2)
        print(f'the F norm of the matrix 1 is {pos1_n} and matrix 2 is {pos2_n}, \n'
              f'while the difference of the norm is {d_n} (||delta_mat||/||matrix2||={d_n / pos2_n}). ')

    data2 = load_mp_data(mpid=mpid_2)
    activation = get_features(model, data)
    activation = copy.deepcopy(activation)
    activation2 = get_features(model, data2)
    compare_distance(pos1=data.pos.cpu().numpy(), pos2=data2.pos.cpu().numpy())
    compare_distance(pos1=activation['interactions3'].cpu().numpy(),
                     pos2=activation2['interactions3'].cpu().numpy())
    compare_distance(pos1=activation['interactions1'].cpu().numpy(),
                     pos2=activation2['interactions1'].cpu().numpy())
    # plot_act_int(activation=activation,   y_axis_labels = ['Se', 'O'])
    visualize_features(activation, activation2, ['Se', 'O'])
    # plt.tight_layout()
    # plt.savefig('./features.eps', dpi=600)
    plt.savefig('./features.png')
    plt.show()
    # features = activation['interactions3'].cpu().numpy()



if __name__ == '__main__':
    # plot_features()

    SeO2_features_plot()
    # plot_act_int()

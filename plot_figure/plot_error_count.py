import copy
import json

import global_var as gv

gv.set_value('task', 'ef')  # loading dataset with 'ef' and 'eg'.
import matplotlib.pyplot as plt
import torch
from utils.registry import registry, setup_imports
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
import os
from scipy import stats
from trainer_heanet_mtl import validate_model, evaluate, load_mp_data
from trainer_heanet import transform
from trainer_heanet_mtl_HEA import load_hea_data, load_hea_data_single_file, validate_model_hea
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from datasets.preprocessing import PoscarToGraph
import matplotlib as mpl
from model import ECModel
from typing import List
from pymatgen.core.periodic_table import ElementBase
from collections import defaultdict
from pymatgen.io.vasp import Poscar

# mpl.rc('font', **{'famil'})
# fs=20
# mpl.rc('xtick', labelsize=fs)
# mpl.rc('ytick', labelsize=fs)
# mpl.rc('axes', labelsize=fs)

device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')


class Describers:
    """
    This class method is designed for the generation of describers from the trained ECNet models.
    """

    def __init__(self, tasks, transform, model_name):
        self.describer_model = self.load_ecnet(tasks, transform, model_name)
        self.elemental_properties = dict()

    def load_ecnet(self, tasks, transform, model_name):
        """
        load the model from a file.
        :param tasks: (list) Such as ['ms', 'mb']
        :param transform: (list) Such as []
        :param model_name: (str) For example. '../saved_models_mtl_HEA/mtl_2_ms_mb_HEA_500_b0_best.pt'
        :return:
        """
        ecmodel = ECModel(tasks=tasks, transform=transform)
        ecmodel.load_model(model_name=model_name)
        return ecmodel

    def load_hea_data(self, data_name, poscar_dir='./HEA_Data/POSCARS'):
        """
        load the POSCARS of the data.

        args:
            data_name: str. the dataname like './HEA_Data/Out_labels/Database.xlsx'
            Here, we use the preprocessed  'xlsx' file with one sheet, and a column named structures
        """
        wb = pd.read_excel(io=data_name, sheet_name=0, engine='openpyxl')
        poscars = []
        for struc in wb.structures:
            poscar = os.path.join(poscar_dir, struc)
            poscars.append(poscar)
        return poscars

    @staticmethod
    def one_hot_representations():
        elemental_properties = {}
        n = 100
        ele_set = [ElementBase.from_Z(i).symbol for i in range(1, n + 1)]
        for i in range(n):
            element_one_hot = [0 for i in range(n)]
            element_one_hot[i] = 1
            elemental_properties[ele_set[i]] = element_one_hot
        return elemental_properties

    def get_elemental_properties(self, poscar: str, info='interactions3'):
        elemental_properties = {}
        ecnet = self.describer_model
        model = ecnet.model
        data = ecnet.convert_data(poscar)
        an = data.atomic_numbers
        atomic_number = an.cpu().numpy().tolist()
        an_set = list(set(atomic_number))
        # keep its original order.
        an_set.sort(key=atomic_number.index)
        ele_set = [ElementBase.from_Z(i).symbol for i in an_set]
        features_dict = self.get_features(model=model, data=data)

        # v1 = features_dict['interactions1'].cpu().numpy()
        # v2 = features_dict['interactions2'].cpu().numpy()
        # v3 = features_dict['interactions3'].cpu().numpy()

        vi = features_dict.get(info).cpu().numpy()
        for i, ele in enumerate(ele_set):
            elemental_properties[ele] = vi[i][:]
        # print(features_dict)
        return elemental_properties

    def get_elemental_properties_dataset(self, poscars: List[str], info='interactions3'):
        """

        :param poscars: List[str]. in order to
        :param info:
        :return:
        """
        ep_dataset = defaultdict(int)
        element_cnt = defaultdict(int)
        elemental_properties = {}
        for poscar in poscars:
            elemental_properties = self.get_elemental_properties(poscar=poscar, info=info)
            for k, v in elemental_properties.items():
                ep_dataset[k] += v
                element_cnt[k] += 1
        for k, v in ep_dataset.items():
            elemental_properties[k] = (ep_dataset.get(k) / element_cnt.get(k)).tolist()
        return elemental_properties

    @staticmethod
    def write_describers(obj, filename):
        """

        :param obj: (dict)
        :param filename:
        :return:
        """
        save_dir = './describers'
        os.makedirs(name=save_dir, exist_ok=True)
        filename = os.path.join(save_dir, filename)

        # if  isinstance(obj.values(), float):
        with open(filename, 'w') as f:
            json.dump(obj, f)

    @staticmethod
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

        # Refer to the model itself.
        model.tower_layers[0][0].register_forward_hook(get_activation('tower_lin_1'))
        model.tower_layers[0][2].register_forward_hook(get_activation('tower_lin_2'))
        #     model.interactions[5].lin.register_forward_hook(get_activation('interactions6'))
        output = model(data.atomic_numbers.long(), data.pos)
        # print(f'the models output is {output}')
        return activation

    @staticmethod
    def get_feature_v0(model, data):
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
        output = model(data.atomic_numbers.long(), data.pos)
        print(f'the models output is {output}')
        return activation


def plot_error_count(y_true, y_pred):
    """
    plot the error count density figure.
    :param y_true: ndarray
    :param y_pred: ndarray
    :return:
        nothing
    """
    bins = 30
    save_fig = True
    fig_path = './fig'
    fig = plt.figure()
    percent = (y_true - y_pred) / y_true

    # the density is obtained from a kernel density estimatioin with Gaussian kernel
    # it do not support for non-Gaussian kernels since version 0.11.0
    fg = sns.kdeplot(data=percent, x='ef [eV/atom]', y='count density [%]')
    if save_fig:
        plt.savefig(os.path.join(fig_path, 'fig2_error.png'), format='png', bbox_inches='tight')


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


def load_data(data_name, poscar_dir='./HEA_Data/POSCARS'):
    """
    load the POSCARS and Symmetry of the data(BCC(-1)/FCC(+1)).

    args:
        data_name: str. the dataname like './HEA_Data/Out_labels/Database.xlsx'
        Here, we use the preprocessed  'xlsx' file with one sheet, and a column named structures
    """
    wb = pd.read_excel(io=data_name, sheet_name=0, engine='openpyxl')
    poscars = []
    symmetries = []
    for struc in wb.structures:
        poscar = os.path.join(poscar_dir, struc)
        if struc.split('_')[-1] == 'sqsfcc':
            symmetry = 1
        elif struc.split('_')[-1] == 'sqsbcc':
            symmetry = -1
        else:
            raise ValueError("Can not find the symmetry from the name!")
        poscars.append(poscar)
        symmetries.append(symmetry)
    return poscars, symmetries


def convert_data(filename):
    """
    the file name of the poscars
    args:
        filename: str. './HEA_Data/Binary_POSCAR_Files/POSCAR_Co3Cr_sqsfcc'
    """
    pg = PoscarToGraph(radius=10, max_neigh=200)
    data = pg.to_graph(filename)
    return data


def predict_features():
    """
    Use the trained ML model to predict the features, and use the tsne to reduce the
    dimensionality.

    returns:

    """
    # hea_mtl = ModelPrediction(model_name='../saved_models_mtl_HEA/mtl_3_etot_emix_ef_HEA_500_b0_best.pt',
    #                          tasks=['etot' ,'emix', 'eform'],
    #                          hidden_channels=128, n_filters=64, n_interactions=3,
    #                          n_gaussians=50, cutoff=10
    #                          )

    ecmodel = ECModel(tasks=['ms', 'mb'], transform=[])
    ecmodel.load_model(model_name='../saved_models_mtl_HEA/mtl_2_ms_mb_HEA_500_b0_best.pt')
    model = ecmodel.model
    # model = hea_mtl.load_trained_model()
    # poscars, symmetries = load_data(data_name='../HEA_Data/Out_labels/Database.xlsx',
    #                                 poscar_dir='../HEA_Data/POSCARS')
    poscars, symmetries = load_data(data_name='../HEA_Data/Out_labels/Qua+Qui.xlsx',
                                    poscar_dir='../HEA_Data/POSCARS')
    # poscars, symmetries = load_data(data_name='../HEA_Data/Out_labels/B+T.xlsx',
    #                                 poscar_dir='../HEA_Data/POSCARS')
    X = []
    # target = []
    for poscar in poscars:
        data = convert_data(poscar)
        data.to(device)
        activation = get_features(model, data)
        features = activation['interactions3'].cpu().numpy()
        features = features.reshape(-1)
        X.append(features)
        # target.append(symmetries)
    #     X = np.concatenate(X, axis=1)
    return X, symmetries


def plot_features():
    X, target = predict_features()
    # X = np.stack(X)
    binary, ternary, quaternay, quinary = [], [], [], []
    t2, t3, t4, t5 = [], [], [], []
    length = len(X)
    for idx, features in enumerate(X):
        if len(features) == 2 * 128:
            binary.append(features)
            t2.append(target[idx])
        elif len(features) == 3 * 128:
            ternary.append(features)
            t3.append(target[idx])
        elif len(features) == 4 * 128:
            quaternay.append(features)
            t4.append(target[idx])
        elif len(features) == 5 * 128:
            quinary.append(features)
            t5.append(target[idx])
        else:
            pass
    # binary = np.stack(binary)
    # ternary = np.stack(ternary)
    quaternay = np.stack(quaternay)
    quinary = np.stack(quinary)

    def plot_tsne(data, color):
        tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(data)
        print(tsne.shape)
        plt.scatter(tsne[:, 0], tsne[:, 1], c=color)
        plt.show()

    # for i, j in zip([binary, ternary, quaternay, quinary],[t2,t3,t4,t5]):
    #     plot_tsne(i,j)

    for i, j in zip([quaternay, quinary], [t4, t5]):
        plot_tsne(i, j)

    # for i, j in zip([ binary, ternary],[t2,t3]):
    #     plot_tsne(i,j)


def plot_act(feature1, feature2):
    # fig = plt.figure(figsize=(20, 8),dpi=300)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(20, 6), dpi=600)

    ax1 = plt.subplot(2, 1, 1)
    y_axis_labels = ['Co', 'Cr']
    ax = sns.heatmap(feature1.numpy(), cmap='vlag', yticklabels=y_axis_labels, ax=ax1)
    ax1.set_ylabel('Interaction6-fcc')

    ax2 = plt.subplot(2, 1, 2)
    ax = sns.heatmap(feature2.numpy(), cmap='vlag', yticklabels=y_axis_labels, ax=ax2)
    ax2.set_ylabel('Interaction6-bcc')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)


def plot_act_int(activation, y_axis_labels):
    feature1 = activation['interactions1']
    feature2 = activation['interactions2']
    feature3 = activation['interactions3']
    # fig = plt.figure(figsize=(20, 8),dpi=300)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(20, 6), dpi=600)

    labels_fd = dict(fontsize=16,
                     color='r',
                     family='Times New Roman',
                     weight='light',
                     style='italic',
                     )
    labels_size = 30
    cb_dict = {}

    ax1 = plt.subplot(3, 1, 1)
    ax1 = sns.heatmap(feature1.cpu().numpy(), cmap='vlag', yticklabels=y_axis_labels, ax=ax1, cbar=False)
    cb1 = ax1.figure.colorbar(ax1.collections[0])
    cb1.ax.tick_params(labelsize=labels_size)
    ax1.set_ylabel('Interaction1', fontsize=labels_size)
    # ax1.set_axis_off()
    ax1.tick_params(axis='y', labelsize=labels_size)

    ax2 = plt.subplot(3, 1, 2)
    ax2 = sns.heatmap(feature2.cpu().numpy(), cmap='vlag', yticklabels=y_axis_labels, ax=ax2, cbar=False)
    cb2 = ax2.figure.colorbar(ax2.collections[0])
    cb2.ax.tick_params(labelsize=labels_size)
    ax2.set_ylabel('Interaction2', fontsize=labels_size)
    ax2.tick_params(axis='y', labelsize=labels_size)
    # ax2.set_axis_off()

    ax3 = plt.subplot(3, 1, 3)
    ax3 = sns.heatmap(feature3.cpu().numpy(), cmap='vlag', yticklabels=y_axis_labels, ax=ax3, cbar=False)
    ax3.set_ylabel('Interaction3', fontsize=labels_size)
    cb3 = ax3.figure.colorbar(ax3.collections[0])
    cb3.ax.tick_params(labelsize=labels_size)
    ax3.set_xticks(np.linspace(start=0, stop=128, num=5),
                   labels=[int(i) for i in np.linspace(start=0, stop=128, num=5)])
    ax3.tick_params(axis='x', labelrotation=360, labelsize=labels_size)
    ax3.tick_params(axis='y', labelsize=labels_size)

    plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=0.3)


def write_describers_example():
    """
    Write the element features into json files.

    The 'info' refers the name of the ECNet structures.
    It contains 'interactions3', 'tower_lin_2'

    The final output of the json files are named according to the model's layer name.
    For example, 'ecmodel-ib3.json', 'ecmodel-tower_lin2.json'

    :return: None
    """
    des = Describers(tasks=['ms', 'mb'], transform=[],
                     model_name='../saved_models_mtl_HEA/mtl_2_ms_mb_HEA_500_b2.pt')
    # Take one poscar as an example to observe the final representations.
    # des.get_elemental_properties('../HEA_Data/Binary_POSCAR_Files/POSCAR_Co3Cr_sqsfcc')
    poscars = des.load_hea_data(data_name='../HEA_Data/Out_labels/Database.xlsx', poscar_dir='../HEA_Data/POSCARS')
    ep = des.get_elemental_properties_dataset(poscars=poscars, info='interactions3')
    des.write_describers(ep, 'ecmodel-ib3.json')


if __name__ == '__main__':
    # plot_features()
    # plot_act_int()

    # get element features extracted from specific models and write it into a json file.
    write_describers_example()

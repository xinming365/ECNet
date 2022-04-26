import copy

import global_var as gv

gv.set_value('task', 'ef')  # loading dataset with 'ef' and 'eg'.
import matplotlib.pyplot as plt
import torch
from utils.registry import registry, setup_imports
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
import os
from scipy import stats
import numpy as np
import pandas as pd
from datasets.preprocessing import PoscarToGraph
import matplotlib as mpl
from forces_trainer_heanet import load_NbMoTaW, evaluate, validate_model, normalizer
from datasets.NbMoTaW_dataset import NbMoTaW_Dataset

device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_name, hidden_channels=128, n_filters=64, n_interactions=3,
               n_gaussians=50, cutoff=10, num_tasks=2, tower_h1=128,
               tower_h2=64):
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
                                 atomref=None, num_tasks=num_tasks,
                                 tower_h1=tower_h1,
                                 tower_h2=tower_h2)
    # load parameters of trained model
    model_state = torch.load(model_name, map_location=device)
    model.load_state_dict(model_state)
    model.to(device)
    return model


def modified_validate_model(model, loader):
    elements_p, binary_p, ternary_p,  quaternary_p= [], [], [], []
    elements_t, binary_t, ternary_t, quaternary_t = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            types = set(batch.atomic_numbers.numpy())
            out = model(batch.atomic_numbers.long(), batch.pos, batch=batch)
            energy = [y.squeeze() for y in out]
            energy = energy[0]  # energy is in list type here.
            energy = normalizer.denorm(energy)
            y_energy = batch.__getitem__('energy')
            if types==1:
                elements_t.append(y_energy)
                elements_p.append(energy)
            elif types==2:
                binary_t.append(y_energy)
                binary_p.append(energy)
            elif types==3:
                ternary_t.append(y_energy)
                ternary_p.append(energy)
            elif types==4:
                quaternary_t.append(y_energy)
                quaternary_p.append(energy)
            else:
                raise Exception('Something went wrong. We do not implement n_elements > 4')
    elements_t = torch.cat(elements_t, dim=0).detach().cpu().numpy()
    elements_p = torch.cat(elements_p, dim=0).detach().cpu().numpy()

    binary_t = torch.cat(binary_t, dim=0).detach().cpu().numpy()
    binary_p = torch.cat(binary_p, dim=0).detach().cpu().numpy()

    ternary_t = torch.cat(ternary_t, dim=0).detach().cpu().numpy()
    ternary_p = torch.cat(ternary_p, dim=0).detach().cpu().numpy()

    quaternary_t = torch.cat(quaternary_t, dim=0).detach().cpu().numpy()
    quaternary_p = torch.cat(quaternary_p, dim=0).detach().cpu().numpy()
    return (elements_t, binary_t, ternary_t, quaternary_t), (elements_p, binary_p, ternary_p, quaternary_p)


class ForcesEnergyPrediction(object):
    """ Assign the parameters for model and datasets.

    This class is designed for the prediction of forces and energies of materials.
    When use the trained models, you need to provide the path of trained models <'model_name'> and the
    corresponding training hyperparameters such as hidden_channels, n_filters and so on.

    Here, the NbMoTaW datasets are  the default datasets.
    """

    def __init__(self, model_name,
                 hidden_channels=128, n_filters=64, n_interactions=3,
                 n_gaussians=50, cutoff=8):
        self.model_name = model_name
        self.hidden_channels = hidden_channels
        self.n_filters = n_filters
        self.n_interactions = n_interactions
        self.n_gaussians = n_gaussians
        self.cutoff = cutoff
        self.train_loader, self.validate_loader, self.test_loader = load_NbMoTaW(is_validate=True, cutoff=self.cutoff)

    def obtain_predictions(self):
        model = self.load_trained_model()
        out_pred, out_true = validate_model(model, self.test_loader)
        score = evaluate(out_pred, out_true)
        print('mae in the test set is {}'.format(score))
        return out_true, out_pred

    def load_trained_model(self):
        model = load_model(self.model_name, hidden_channels=self.hidden_channels,
                           n_filters=self.n_filters, n_interactions=self.n_interactions,
                           n_gaussians=self.n_gaussians, cutoff=self.cutoff,
                           num_tasks=1)
        return model

    def obtain_predictions_systems(self):
        model = self.load_trained_model()
        (elements_t, binary_t, ternary_t, quaternary_t), (elements_p, binary_p, ternary_p, quaternary_p) \
            = modified_validate_model(model, self.train_loader)
        return (elements_t, binary_t, ternary_t, quaternary_t), (elements_p, binary_p, ternary_p, quaternary_p)

    def energy_true_vs_pred(self, actual, pred, show_legend=True, show_mae=False):
        (elements_t, binary_t, ternary_t, quaternary_t), (elements_p, binary_p, ternary_p, quaternary_p) = \
            self.obtain_predictions_systems()
        marker_binary = 'o'
        marker_ternary = 'o'
        marker_quaternary = 'o'
        marker_elements = 'o'


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


def convert_data(filename):
    """
    the file name of the poscars
    args:
        filename: str. './HEA_Data/Binary_POSCAR_Files/POSCAR_Co3Cr_sqsfcc'
    """
    pg = PoscarToGraph(radius=10, max_neigh=200)
    data = pg.to_graph(filename)
    return data


def mp_data_plot():
    """
    plot the feature vectors of some materials.
    :return:
    """
    NbMoTaW_ecmtl = ForcesEnergyPrediction(model_name='../saved_models_db_HEA/NbMoTaW_epoch500_hidden256_nf128_int3_cut8.0_best.pt',
                                     hidden_channels=256, n_filters=128, n_interactions=3,
                                     n_gaussians=50, cutoff=8
                                     )
    model = NbMoTaW_ecmtl.load_trained_model()
    y_true, y_pred = NbMoTaW_ecmtl.obtain_predictions()
    mae_s = mean_absolute_error(y_true, y_pred)
    r2_s = r2_score(y_true, y_pred)
    y_pred = np.squeeze(y_pred)
    print(y_true.shape, y_pred.shape)
    from plot_figure import scatter_hist
    scatter_hist(y_true, y_pred, task='etot', fig_path='./', r2_s=r2_s, mae_s=mae_s)


def main():
    energy_predictor = ForcesEnergyPrediction(
        model_name='../saved_models_mtl_HEA/mtl_3_etot_emix_ef_HEA_500_b0_best.pt',
        hidden_channels=128, n_filters=64, n_interactions=3,
        n_gaussians=50, cutoff=8
        )
    y_true, y_pred = energy_predictor.obtain_predictions()
    mae_s = mean_absolute_error(y_true, y_pred)
    r2_s = r2_score(y_true, y_pred)
    y_pred = np.squeeze(y_pred)
    print(y_true.shape, y_pred.shape)
    from plot_figure import scatter_hist
    scatter_hist(y_true, y_pred, task='etot', fig_path='./', r2_s=r2_s, mae_s=mae_s)


if __name__ == '__main__':
    # main()
    # plot_features()
    import os
    print(os.path.abspath('./'))

    mp_data_plot()

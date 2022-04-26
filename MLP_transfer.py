import json

import pandas as pd
from pymatgen.core import Composition
from typing import Dict
from sklearn.decomposition import PCA
import numpy as np
import torch.nn as nn
from typing import Tuple, List, Optional, Sequence, Union
import torch
from utils.load_data import LoadDataset
import os
from pymatgen.io.vasp import Poscar
from datasets.HEA_dataset import task_dict
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset
import copy
from tqdm import tqdm


class My_dataset(Dataset):
    """
    This is designed for more general inputs, such as inputs of list of compounds and related property targets.
    Covert the general inputs into the dataset used in pytorch.
    """
    def __init__(self, inputs, targets):
        super().__init__()
        self.src, self.target = torch.Tensor(inputs), torch.Tensor(targets)

    def __getitem__(self, index):
        return self.src[index], self.target[index]

    def __len__(self):
        return len(self.src)


class ElementFeatures:
    """
    This class includes methods about the element features.
    It contains that converting the compound into a feature vector based on the element features,
    reading elemental features from a processed data, and reducing the feature dimension.
    """
    def __init__(self, element_properties: Dict):
        self.element_properties = element_properties

    def convert(self, mat_comp):
        """
        Convert one materials into the feature vector based on the element features.
        :param mat_comp: str. Such as 'Fe2O3'
        :return:
        """
        comp = Composition(mat_comp)
        element_dict = {str(i): j for i, j in comp._data.items()}

        featuers = []
        weights = []
        for i, j in element_dict.items():
            featuers.append(self.element_properties[i])
            weights.append(j)

        # matrix transpose. T*N_f -> N_f*T
        features = list(zip(*featuers))
        out = []
        for i in features:
            val = 0
            sum = 0
            for f, w in zip(i, weights):
                val += f * w
                sum += w
            out.append(val / sum)
        return out

    def convert_datasets(self, comps: List[str]):
        out = [self.convert(i) for i in comps]
        return out

    @classmethod
    def from_file(cls, filename: str):
        """
        Initialize this class from a json file in order to get the element feature vectors.

        :param filename: The file contains the element features.
        :return:
        """

        with open(filename, 'r') as f:
            d = json.load(f)

        return cls(element_properties=d)

    @staticmethod
    def reduce_dimension(element_properties, num_dim=None):
        """
        Reduce the feature dimension.

        :param element_properties: (dict)  dictionary of elemental/specie propeprties
        :param num_dim: (int)  number of dimension to keep
        :return:
        """

        if num_dim is None:
            return element_properties

        element_array = []
        element_keys = []
        for k, v in element_properties.items():
            element_array.append(v)
            element_keys.append(k)
        element_array = np.array(element_array)

        pca = PCA(n_components=num_dim)
        reduced_element = pca.fit_transform(element_array)

        for k, v in zip(element_keys, reduced_element):
            element_properties[k] = v.tolist()

        return element_properties


class MLP(nn.Module):
    """
    This class implements the multi-layer perceptron models
    """

    def __init__(self, input_size,
                 n_neurons=(64, 64),
                 n_targets=1,
                 is_embedding=False):
        super(MLP, self).__init__()
        self.module_list = self.construct_mlp(input_size, n_neurons, n_targets, is_embedding)
        # self.device

    def forward(self, x):
        n = len(self.module_list)
        out = self.module_list[0](x)
        for i in range(1, n):
            out = self.module_list[i](out)
        return out

    def construct_mlp(self, input_size: int, n_neurons: Tuple,
                      n_targets: int, is_embedding: bool):
        if is_embedding:
            embedding = nn.Embedding(num_embeddings=100, embedding_dim=32)
            pass
        else:
            Input_layer = nn.Linear(in_features=input_size, out_features=n_neurons[0])

        n_layers = len(n_neurons)
        linears = nn.ModuleList()
        linears.append(Input_layer)

        for i in range(n_layers - 1):
            linears.append(self.mlp_block(in_features=n_neurons[i], out_features=n_neurons[i + 1]))
        linears.append(nn.Linear(n_neurons[-1], n_targets))
        return linears

    @staticmethod
    def mlp_block(in_features, out_features):
        mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )
        return mlp


class MLPTrainer(object):
    """
    This class is designed for training the multi-layer perceptron models.

    We implement methods of how to train the model, evaluate the model, predict properties of input,
    generate the data_loader, etc.

    """
    def __init__(self, describer: Optional[ElementFeatures] = None,
                 input_dim: Optional[int] = None,
                 is_embedding: bool = True,
                 n_neurons: Sequence[int] = (128, 64),
                 n_targets: int = 1):
        self.device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MLP(input_size=input_dim, n_neurons=n_neurons,
                         n_targets=n_targets)
        self.model.to(self.device)

    def train(self, num_epochs, train_loader, val_loader=None):
        model = self.model
        model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        total_step = len(train_loader)
        best_score = float('inf')
        for epoch in range(num_epochs):
            with tqdm(
                    iterable=train_loader,
                    bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',
            ) as t:
                t.set_description_str(f"\33[36m【Epoch {epoch + 1:04d}】")
                for i, (feature, label) in enumerate(train_loader):
                    # feature = torch.tensor(data=feature)
                    feature = feature.to(self.device)
                    label = label.to(self.device)
                    out = model(feature)
                    out = out.squeeze()
                    loss = criterion(out, label)

                    # backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # if (i + 1) % 2 == 0:
                    #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

                    t.set_postfix_str(f"train_loss={loss.item():.6f}，")

                    t.update()
                if val_loader is not None:
                    if (epoch + 1) % 10 == 0:
                        out_true, out_pred = self.evaluate(val_loader)
                        epoch_score = mean_absolute_error(out_true, out_pred)
                        if epoch_score < best_score:
                            best_score = epoch_score
                            best_model_wts = copy.deepcopy(model.state_dict())
        if val_loader is not None:
            torch.save(best_model_wts, './tmp/mlp-transfer-model.pt')
            self.from_saved_model('./tmp/mlp-transfer-model.pt')
        else:
            self.model = model
        return model

    def evaluate(self, data_loader):
        out_pred = []
        out_true = []

        with torch.no_grad():
            for feature, label in data_loader:
                feature = feature.to(self.device)
                label = label.to(self.device)

                output = self.model(feature)
                output = output.squeeze()
                out_pred.extend(output.detach().cpu().numpy())
                out_true.extend(label.detach().cpu().numpy())
            mae_score = mean_absolute_error(out_true, out_pred)
            print('Average mean absolute errors of the model is : {}.'.format(mae_score))
        return out_true, out_pred

    def predict(self, feature):
        feature = torch.Tensor(feature)
        feature = feature.to(self.device)
        output = self.model(feature)
        output = output.detach().cpu().numpy()
        return output

    def save(self, model_name):
        """
        Save the trained model states.
        :param model_name: (str)
        :return:
        """
        model_states = self.model.state_dict()
        default_dir_name = './saved_models_MLP'
        os.makedirs(default_dir_name, exist_ok=True)
        model_name = os.path.join(default_dir_name, model_name)
        torch.save(model_states, model_name)

    def generate_data_loader(self, inputs: Union[List, np.ndarray], targets: Optional[Union[List, np.ndarray]] = None,
                             fraction: float = 1,
                             fraction_seed: int = None):
        """
        generate data_loader module

        :param fraction: The fraction of the total dataset.
        :param fraction_seed: random seed.
        :param targets: targets (list or np.ndarray): Numerical output target list, or
                numpy array with dim (m, ).
        :param inputs: Numerical input feature list or
                numpy array with dim (m, n) where m is the number of data and
                n is the feature dimension.
        :return:
        """
        dataset = self._create_dataset(inputs, targets)
        ld = LoadDataset(dataset=dataset, batch_size=64)
        train_loader, validate_loader, test_loader = ld.load(fraction=fraction, fraction_seed=fraction_seed)
        return train_loader, validate_loader, test_loader

    def _create_dataset(self, inputs, targets):
        return My_dataset(inputs=inputs, targets=targets)

    @staticmethod
    def _process_hea_dataset(poscars_dir, labels_name, target=None, phase='fcc'):
        """
        This is designed for our calculated CrFeCoNiMn/Pd HEA data.

        :param poscars_dir: (str)
        :param labels_name: (str)
        :param target: (str)
        :param phase: (str)
        :return:
        """
        df = pd.read_csv(labels_name)

        phase = phase.lower()
        if phase == 'fcc':
            phase_data = df[df['phase'] == 'fcc']
        elif phase == 'bcc':
            phase_data = df[df['phase'] == 'bcc']
        else:
            phase_data = df
        comps = phase_data['structures'].to_list()

        target = target.lower()  # support uppercase and lowercase of the keys.
        assert target in task_dict.keys(), 'Please refer to the supported targets in the "task_dict".'
        target = task_dict.get(target)
        targets = phase_data[target].to_list()
        formulas = []
        for c in comps:
            pos = os.path.join(poscars_dir, c)
            poscar = Poscar.from_file(pos, check_for_POTCAR=False, read_velocities=True)
            formula = poscar.structure.formula
            formulas.append(formula)
        return formulas, targets

    def from_saved_model(self, model_name):
        model_states = torch.load(model_name, map_location=self.device)
        self.model.load_state_dict(model_states)


def element_features_test():
    """
    This method is designed for the test of class method 'ElementFeatures'.
    :return:
    """
    ef = ElementFeatures.from_file('./plot_figure/describers/ecmodel-ib3.json')
    reduced = ef.reduce_dimension(ef.element_properties, num_dim=5)
    ef.element_properties = reduced
    out = ef.convert('FeCoNi')
    print(out)


def model_test():
    # test_target = 'Eform'
    test_target = 'ms'
    epochs = 300
    phase = 'fcc'
    formulas, targets = MLPTrainer._process_hea_dataset(poscars_dir='./HEA_Data/POSCARS',
                                                        labels_name='./HEA_Data/Out_labels/Database.csv',
                                                        target=test_target,
                                                        phase=phase)
    # print(formulas, targets)
    ef = ElementFeatures.from_file(filename='./plot_figure/describers/ecmodel-ib3.json')

    # where to reduce the dimension.
    num_dim = 5
    ef.element_properties = ef.reduce_dimension(ef.element_properties, num_dim=num_dim)
    features = ef.convert_datasets(comps=formulas)

    trainer = MLPTrainer(input_dim=num_dim, is_embedding=True,
                         n_neurons=(128, 64))

    # trainer.from_saved_model('./mlp-transfer-model.pt')

    train_loader, validate_loader, test_loader = trainer.generate_data_loader(inputs=features, targets=targets)
    trainer.train(num_epochs=epochs, train_loader=train_loader, val_loader=validate_loader)

    model_name = 'mlp_' + test_target + '_' + phase + '_' + str(epochs) + '.pt'
    trainer.save(model_name=model_name)
    out_true, out_pred = trainer.evaluate(data_loader=test_loader)

    from plot_figure import scatter_hist

    scatter_hist(out_true, out_pred, task=test_target)


def model_convergence_test(element_features='./plot_figure/describers/ecmodel-ib3.json',
                           exp_name='./exp-ib3-bcc.csv',
                           phase='bcc',
                           epochs=300,
                           test_target='ms'
                           ):
    formulas, targets = MLPTrainer._process_hea_dataset(poscars_dir='./HEA_Data/POSCARS',
                                                        labels_name='./HEA_Data/Out_labels/Database.csv',
                                                        target=test_target,
                                                        phase=phase)
    # print(formulas, targets)
    ef = ElementFeatures.from_file(filename=element_features)

    num_dim = 6
    ef.element_properties = ef.reduce_dimension(ef.element_properties, num_dim=num_dim)
    features = ef.convert_datasets(comps=formulas)

    ## For the one hot training.
    # num_dim = 100
    # ef.element_properties = ef.reduce_dimension(ef.element_properties, num_dim=num_dim)
    # features = ef.convert_datasets(comps=formulas)


    trainer = MLPTrainer(input_dim=num_dim, is_embedding=True,
                         n_neurons=(128, 64))

    n_exp = 5
    cols = ['exp_' + str(i + 1) for i in range(n_exp)]
    # ratios = [0.2, 0.4, 0.6, 0.8, 1]
    ratios = [1]
    # ratios = ratios[::-1]
    exps = []

    # create written files.
    df = pd.DataFrame(columns=cols)
    df.to_csv(exp_name)
    for step, ratio in enumerate(ratios):
        exp_i = []
        print('Loading ratio {} in {}-th experiment.\n'.format(ratio, step))
        for i in range(n_exp):
            train_loader, validate_loader, test_loader = trainer.generate_data_loader(inputs=features,
                                                                                      targets=targets,
                                                                                      fraction=ratio,
                                                                                      fraction_seed=(i + 1),
                                                                                      )
            trainer.train(num_epochs=epochs, train_loader=train_loader, val_loader=validate_loader)
            model_name = 'mlp_' + test_target + '_' + phase + '_ratio' + str(ratio) +'_seed'+ str(i+1) + '_epoch'+ str(epochs) + '.pt'
            trainer.save(model_name=model_name)
            out_true, out_pred = trainer.evaluate(data_loader=test_loader)
            score = mean_absolute_error(out_true, out_pred)
            exp_i.append(score)
        # Evenly round to the given number of decimals.
        exp_i = np.around(exp_i, 4).tolist()
        df = pd.DataFrame([exp_i])
        df.to_csv(exp_name, mode='a', header=False)
        # exps.append(exp_i)

    # df.to_csv(exp_name, header=cols)


def model_convergence_from_models():
    test_target = 'eform'
    phase = 'fcc'
    element_features = './plot_figure/describers/ecmodel-ib1.json'
    exp_name = 'exp-ib1-eform-fcc.csv'
    saved_models_dir = './saved_models_MLP/exp-ib1-fcc-eform'
    epochs = 300
    formulas, targets = MLPTrainer._process_hea_dataset(poscars_dir='./HEA_Data/POSCARS',
                                                        labels_name='./HEA_Data/Out_labels/Database.csv',
                                                        target=test_target,
                                                        phase=phase)
    # print(formulas, targets)
    ef = ElementFeatures.from_file(filename=element_features)

    num_dim = 6
    ef.element_properties = ef.reduce_dimension(ef.element_properties, num_dim=num_dim)
    features = ef.convert_datasets(comps=formulas)

    ## For the one hot training.
    # num_dim = 100
    # ef.element_properties = ef.reduce_dimension(ef.element_properties, num_dim=num_dim)
    # features = ef.convert_datasets(comps=formulas)

    trainer = MLPTrainer(input_dim=num_dim, is_embedding=True,
                         n_neurons=(128, 64))

    n_exp = 5
    cols = ['exp_' + str(i + 1) for i in range(n_exp)]
    ratios = [0.2, 0.4, 0.6, 0.8, 1]
    ratios = ratios[::-1]
    exps = []

    # create written files.
    df = pd.DataFrame(columns=cols)
    df.to_csv(exp_name)
    for step, ratio in enumerate(ratios):
        exp_i = []
        print('Loading ratio {} in {}-th experiment.\n'.format(ratio, step))
        for i in range(n_exp):
            train_loader, validate_loader, test_loader = trainer.generate_data_loader(inputs=features,
                                                                                      targets=targets,
                                                                                      fraction=ratio,
                                                                                      fraction_seed=(i + 1),
                                                                                      )
            model_name = 'mlp_' + test_target + '_' + phase + '_ratio' + str(ratio) + '_seed' + str(
                i + 1) + '_epoch' + str(epochs) + '.pt'
            model_name = os.path.join(saved_models_dir, model_name)
            trainer.from_saved_model(model_name)
            out_true, out_pred = trainer.evaluate(data_loader=test_loader)
            score = mean_absolute_error(out_true, out_pred)
            exp_i.append(score)
        # Evenly round to the given number of decimals.
        exp_i = np.around(exp_i, 4).tolist()
        df = pd.DataFrame([exp_i])
        df.to_csv(exp_name, mode='a', header=False)
        # exps.append(exp_i)

    # df.to_csv(exp_name, header=cols)

if __name__ == '__main__':
    # mlp = MLP(input_size=128, n_neurons=(128, 64), n_targets=1)
    # print(mlp)
    model_convergence_test(element_features='./plot_figure/describers/ecmodel-ib3.json',
                           exp_name='./saved_models_MLP/exp-ib3-ms-bcc.csv',
                           phase='bcc',
                           epochs=300,
                           test_target='ms'
                           )
    # model_test()

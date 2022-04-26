import copy
from typing import Tuple

import numpy as np
from torch_geometric.loader import DataLoader
import torch
from utils.meter import mae
import torch.nn as nn
from torch import Tensor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse
from utils.registry import registry, setup_imports
import time
from torch.utils.data import SequentialSampler, SubsetRandomSampler, RandomSampler
from datasets.db_HEA_reduced import DbHEA
from datasets.NbMoTaW_dataset import NbMoTaW_Dataset
import ase

from ase.io import read
from ase.io.trajectory import Trajectory


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, natoms: Tensor, output: Tensor):
        delta = (input - output) / natoms
        return torch.mean(torch.pow(delta, 2))


device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')


def load_db_hea(is_validate=True):
    """
    This function defines the manner of training/testing dataset.
    :param is_validate: bool.
    :param split_type: int.
        0: randomly splitting all the data into the training/testing with a specific ratio.
        1: choose the low-component alloy as the training data, and make the high-component alloy as the testing data.
    :return: the object of the Dataset.
    """
    data_file = ''
    total_dataset = DbHEA(xyz_dir='./datasets/db_HEA_reduced.xyz')
    data_size = len(total_dataset)
    seq = np.random.RandomState(seed=RANDOM_SEED).permutation(np.arange(len(total_dataset)))
    train_size = int(data_size * 0.8)
    tv_size = int(data_size * 0.9)
    test_seq = seq[tv_size:]
    validate_loader = None  # declare the validate loader
    print('The data size is {}'.format(data_size))
    print('The number of the test size is {}'.format(len(test_seq)))
    if is_validate:
        train_seq = seq[:train_size]
        v_seq = seq[train_size:tv_size]

        train_loader = DataLoader(dataset=total_dataset, batch_size=args.batch_size,
                                  sampler=RandomSampler(train_seq))
        validate_loader = DataLoader(dataset=total_dataset, batch_size=args.batch_size,
                                     sampler=RandomSampler(v_seq))
        print('The number of the train size is {}, validation size is {}'.format(len(train_seq), len(v_seq)))
    else:
        train_seq = seq[:tv_size]
        train_loader = DataLoader(dataset=total_dataset, batch_size=args.batch_size,
                                  sampler=SequentialSampler(train_seq))
    test_loader = DataLoader(dataset=total_dataset, batch_size=args.batch_size,
                             sampler=SequentialSampler(test_seq))

    if is_validate:
        return train_loader, validate_loader, test_loader
    else:
        return train_loader, test_loader


def load_NbMoTaW(is_validate=True, batch_size=128, cutoff=8, RANDOM_SEED = 1454880):
    """
    This function defines the manner of training/testing dataset.
    :param is_validate: bool.
    :param split_type: int.
        0: randomly splitting all the data into the training/testing with a specific ratio.
        1: choose the low-component alloy as the training data, and make the high-component alloy as the testing data.
    :return: the object of the Dataset.
    """
    data_file = ''
    total_dataset = NbMoTaW_Dataset(radius=cutoff, is_train=True).to_list()
    test_dataset = NbMoTaW_Dataset(radius=cutoff, is_train=False).to_list()
    data_size = len(total_dataset)
    seq = np.random.RandomState(seed=RANDOM_SEED).permutation(np.arange(len(total_dataset)))
    train_size = int(data_size * 0.95)
    validate_loader = None  # declare the validate loader
    print('The train data size is {}'.format(data_size))
    print('The number of the test size is {}'.format(len(test_dataset)))
    if is_validate:
        train_seq = seq[:train_size]
        v_seq = seq[train_size:]
        train_dataloader = DataLoader(dataset=total_dataset, batch_size=batch_size,
                                      sampler=RandomSampler(train_seq))
        validate_loader = DataLoader(dataset=total_dataset, batch_size=batch_size,
                                     sampler=RandomSampler(v_seq))
        print('The number of the train size is {}, validation size is {}'.format(len(train_seq), len(v_seq)))
    else:
        train_dataloader = DataLoader(dataset=total_dataset, batch_size=batch_size,sampler=RandomSampler(seq))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if is_validate:
        return train_dataloader, validate_loader, test_dataloader
    else:
        return train_dataloader, test_dataloader


def fine_tune(model):
    saved_model = './saved_models/g_mp_log_200.pt'
    if torch.cuda.is_available():
        # The default is to load all tensors onto GPU.
        model_state = torch.load(saved_model)
    else:
        # load all tensors onto the cpu, using a function.
        model_state = torch.load(saved_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_state)
    for param in model.parameters():
        param.requires_grad = True

    model_ft = train(model)



def evaluate(test_data, predict_data):
    """
    Evaluate the predictioins on the passed data, and return the corresponding score;
        For regression: MAE
        For classification: ROC AUC.
        averaged over the targets when multi-target

    Args:
        test_data: The true labels
        predict_data: The predicted labels by ML model

    Return:
        socre.

    """
    score = mean_absolute_error(test_data, predict_data)
    print('the score of task is {}\n'.format(score))
    return score


def validate_model(model, loader):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            out = model(batch.atomic_numbers.long(), batch.pos, batch=batch)
            energy = [y.squeeze() for y in out]
            energy = energy[0]  # energy is in list type here.
            energy = normalizer.denorm(energy)
            y_energy = batch.__getitem__('energy')
            y_true.append(y_energy)
            y_pred.append(energy)
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
    return y_true, y_pred


class Normalizer(object):
    def __init__(self, mean=None, std=None, device=None):
        self.std = std
        self.mean = mean
        if device is None:
            device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def norm_force(self, normed_tensor):
        return normed_tensor / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def denorm_force(self, normed_tensor):
        return normed_tensor * self.std


# normalizer_force = Normalizer(include_force=True, mean=-572.59630848, std=443.6452794032273)

normalizer = Normalizer(mean=-274.7560850960716, std=265.43605239791077)


def train(model):
    # initialize these variables.
    best_score = 10000
    best_model_wts = ''
    for epoch in range(num_epochs):
        print('epoch: {}'.format(epoch))
        for index, batch in enumerate(train_loader):
            model.train(mode=True)  # keep per batch send into the mode of train.
            batch.to(device, non_blocking=True)
            batch.pos.requires_grad = True
            out = model(batch.atomic_numbers.long(), batch.pos, batch=batch)
            # out = out.squeeze()
            energy = [y.squeeze() for y in out]
            energy = energy[0]  # energy is in list type here.

            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    batch.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            # y_energy = transform(args.transform, batch.__getitem__('energy'))
            # y_forces = transform(args.transform, batch.__getitem__('force'))
            y_forces = batch.__getitem__('force')
            y_energy = batch.__getitem__('energy')
            # std_y_energy = get_std(y_energy)
            std_y_energy = normalizer.norm(y_energy)
            std_y_forces = normalizer.norm_force(y_forces)
            loss_energy = criterion(energy, std_y_energy)
            loss_forces = criterion(forces, std_y_forces)
            # loss = loss_energy
            loss = 0.3*loss_forces +  0.7*loss_energy
            # loss =  loss_energy
            loss.backward()
            optim.step()
            optim.zero_grad()
            energy = normalizer.denorm(energy)
            print('Epoch[{}]({}/{}): Loss: {:.4f} Potential Energy mae:{:.4f} Force_x mae:{:.4f}'.format(epoch,
                                                                                                         index,
                                                                                                         len(train_loader),
                                                                                                         loss.item(),
                                                                                                         mae(energy.cpu(),
                                                                                                             y_energy.cpu()).view(
                                                                                                             -1)[0],
                                                                                                         mae(forces.cpu(),
                                                                                                             y_forces.cpu())[
                                                                                                             0]))

        if args.is_validate:
            y_true, y_pred = validate_model(model, loader=validate_loader)
            epoch_score = evaluate(y_true, y_pred)
            # We observe all parameters being optimized according to the loss in the validation set.
            # scheduler.step(epoch_score)
            print('The score in {} epochs is {}; The current best score is {}'.format(
                epoch, epoch_score, best_score))
            if epoch_score < best_score:
                best_score = epoch_score
                best_model_wts = copy.deepcopy(model.state_dict())

    if args.save_model:
        # model_name = args.task+'_type'+ str(args.split_type)+'_'+ str(args.epochs)
        # model_name = args.task + '_mp_' + args.transform + '_' + str(args.epochs)
        model_name = 'NbMoTaW' + args.transform + '_epoch' + str(args.epochs) + '_hidden'+str(args.hidden_channels)+'_nf'+str(args.n_filters)+\
            '_int'+str(args.n_interactions)+'_cut'+str(args.cutoff)
        saved_dir = './saved_models_db_HEA/'
        torch.save(model.state_dict(), saved_dir + model_name + '.pt')
        if args.is_validate:
            best_model_name = model_name + '_best'
            torch.save(best_model_wts, saved_dir + best_model_name + '.pt')
            print('saved in {}'.format(saved_dir + best_model_name + '.pt'))
        print('saved in {}'.format(saved_dir + model_name + '.pt'))
    return model


def test():
    # model_name = './saved_models_db_HEA/etot_db_HEA_500_128.pt'
    model_name = './saved_models_db_HEA/NbMoTaW_epoch500_hidden256_nf128_int3_cut8.0_best.pt'
    model_state = torch.load(model_name)
    model.load_state_dict(model_state)
    y_true, y_pred = validate_model(model, test_loader)
    # y_true, y_pred = validate_model(model, train_loader)
    print(y_true.shape, y_pred.shape)
    print(y_true, y_pred)
    mae_s = mean_absolute_error(y_true, y_pred)
    r2_s = r2_score(y_true, y_pred)
    print(r2_s, mae_s)
    return y_true, y_pred


def plot():
    y_true, y_pred = test()
    mae_s = mean_absolute_error(y_true, y_pred)
    r2_s = r2_score(y_true, y_pred)
    y_pred = np.squeeze(y_pred)
    print(y_true.shape, y_pred.shape)
    from plot_figure import scatter_hist
    scatter_hist(y_true, y_pred, task='etot', fig_path='./', r2_s=r2_s, mae_s=mae_s)



if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=5  python trainer_heanet.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=128, help='number of hidden channels for the embeddings')
    parser.add_argument('--n_filters', type=int, default=128, help='number of filters')
    parser.add_argument('--n_interactions', type=int, default=6, help='number of interaction blocks')
    parser.add_argument('--n_gaussian', type=int, default=50, help='number of gaussian bases to expand the distances')
    parser.add_argument('--cutoff', type=float, default=10, help='the cutoff radius to consider passing messages')
    parser.add_argument('--aggregation', type=str, default='add', help='the aggregation scheme ("add", "mean", or "max") \
                                                                       to use in the messages')
    parser.add_argument('--seed', type=int, default=30, help='random seed')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate of the algorithm')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--task', type=str, default='etot', help='the target to model in the dataset')
    parser.add_argument('--split_type', type=int, default=0, help='the splitting type of the dataset')
    parser.add_argument('--model', type=str, default='heanet', help='the name of the ML model')
    parser.add_argument('--scaling', type=float, default=1000, help='the scaling factor of the target.')
    parser.add_argument('--batch_size', type=int, default=128, help='the batch size of the data loader')
    parser.add_argument('--save_model', type=bool, default=True, help='save the trained model')
    parser.add_argument('--transform', type=str, default='',
                        help='transform the target proerty, only support log and scaling')
    parser.add_argument('--is_validate', default=False, action='store_true',
                        help='validate the model during training. when used, it will become true')
    # parser.add_argument('--saved_model', type='str', default='etot_type0_200.pt', help='the trained model')
    parser.add_argument('--use_pbc', default=False, action='store_true',
                        help='Whether to use the periodic boundary condition.')
    parser.add_argument('--train', '-t', action='store_true', help='Training the ECMTL model.')
    parser.add_argument('--predict', '-p', action='store_true', help='Applying the ECMTL to predict something.')

    args = parser.parse_args()

    setup_imports()
    RANDOM_SEED = 1454880

    print('Whether to use the validation dataset: {}\n'.format(args.is_validate))
    print('Whether to consider the periodic boundary condition: {}\n'.format(args.use_pbc))

    model = registry.get_model(args.model
                               )(hidden_channels=args.hidden_channels,
                                 num_filters=args.n_filters,
                                 num_interactions=args.n_interactions,
                                 num_gaussians=args.n_gaussian,
                                 cutoff=args.cutoff,
                                 readout=args.aggregation,
                                 dipole=False, mean=None, std=None,
                                 atomref=None, use_pbc=args.use_pbc)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=args.weight_decay, amsgrad=False)

    criterion = torch.nn.MSELoss()

    is_validate = args.is_validate
    if is_validate:
        train_loader, validate_loader, test_loader = load_NbMoTaW(is_validate=is_validate)
    else:
        train_loader, test_loader = load_NbMoTaW(is_validate=is_validate)
    num_epochs = args.epochs
    scaling = args.scaling
    model.to(device)

    if args.train:
        print('The validation is {}'.format(is_validate))
        # When train, use the following codes.
        # CUDA_VISIBLE_DEVICES=6  python trainer_heanet_mtl.py --epochs 10 --batch_size 64 --weight_decay 1e-4  --task k --transform log  --train True
        train(model=model)

    if args.predict:
        plot()


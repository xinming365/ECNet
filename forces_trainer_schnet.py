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
from torch.utils.data import SequentialSampler
from datasets.db_HEA_reduced import DbHEA

# arguments for settings.
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=128, help='number of hidden channels for the embeddings')
parser.add_argument('--n_filters', type=int, default=128, help='number of filters')
parser.add_argument('--n_interactions', type=int, default=3, help='number of interaction blocks')
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
parser.add_argument('--model', type=str, default='schnet', help='the name of the ML model')
parser.add_argument('--scaling', type=float, default=1000, help='the scaling factor of the target.')
parser.add_argument('--batch_size', type=int, default=128, help='the batch size of the data loader')
parser.add_argument('--save_model', type=bool, default=True, help='save the trained model')
parser.add_argument('--train', type=bool, default=False, help='train or test the model')
parser.add_argument('--transform', type=str, default='',
                    help='transform the target proerty, only support log and scaling')
parser.add_argument('--is_validate', type=bool, default=False,
                    help='validate the model during training the ef and eg tasks')
# parser.add_argument('--saved_model', type='str', default='etot_type0_200.pt', help='the trained model')


args = parser.parse_args()

setup_imports()
RANDOM_SEED = 1454880


class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, natoms: Tensor, output: Tensor):
        delta = (input - output) / natoms
        return torch.mean(torch.pow(delta, 2))


device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')

model = registry.get_model(args.model
                           )(hidden_channels=args.hidden_channels,
                             num_filters=args.n_filters,
                             num_interactions=args.n_interactions,
                             num_gaussians=args.n_gaussian,
                             cutoff=args.cutoff,
                             readout=args.aggregation,
                             dipole=False, mean=None, std=None,
                             atomref=None)

optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                         weight_decay=args.weight_decay, amsgrad=False)

# We reduce the learning rate when the metric has stopped improving.
# optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=10, gamma=0.1, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', factor=0.1,
#                                                         threshold=1e-4, threshold_mode='rel',
#                                                        patience=10, verbose=True, cooldown=0, min_lr=0,
#                                                        eps=1e-8)
criterion = torch.nn.L1Loss()


# criterion = torch.nn.MSELoss()
# criterion = MyLoss()


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
                                  sampler=SequentialSampler(train_seq))
        validate_loader = DataLoader(dataset=total_dataset, batch_size=args.batch_size,
                                     sampler=SequentialSampler(v_seq))
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


is_validate = args.is_validate

if is_validate:
    train_loader, validate_loader, test_loader = load_db_hea(is_validate=is_validate)
else:
    train_loader, test_loader = load_db_hea(is_validate=is_validate)
num_epochs = args.epochs
scaling = args.scaling
model.to(device)


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


def transform(trans, y, forward=True):
    """
    transform the target y.
    :param
    trans: str. only support scaling and log10.
    y: array.
    :return: transformed array
    """
    if forward == True:
        if trans == 'log':
            y = torch.log10(y)
        elif trans == 'scaling':
            y = scaling * y
        else:
            y = y
    else:  # forward=False
        if trans == 'log':
            y = torch.pow(10, y)
        elif trans == 'scaling':
            y = y / scaling
        else:
            y = y
    return y


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
            # y_energy = transform(args.transform, batch.__getitem__('energy'))
            y_energy = batch.__getitem__('energy')
            y_true.append(y_energy)
            y_pred.append(energy)
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
    score = evaluate(y_true, y_pred)
    return score



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
            out = model(batch.atomic_numbers.long(), batch.pos, batch=batch.batch)
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
            # y_forces = batch.__getitem__('force')
            y_energy = batch.__getitem__('energy')
            loss_energy = criterion(energy, y_energy)
            # loss_forces = criterion(forces, y_forces)
            loss = loss_energy
            # loss = loss_forces + 100*loss_energy
            # loss =  loss_energy
            loss.backward()
            optim.step()
            optim.zero_grad()


            # print('Epoch[{}]({}/{}): Loss: {:.4f} Potential Energy mae:{:.4f} Force_x mae:{:.4f}'.format(epoch,
            #                                                                                              index,
            #                                                                                              len(train_loader),
            #                                                                                              loss.item(),
            #                                                                                              mae(energy.cpu(),
            #                                                                                                  y_energy.cpu()).view(
            #                                                                                                  -1)[0],
            #                                                                                              mae(forces.cpu(),
            #                                                                                                  y_forces.cpu())[
            #                                                                                                  0]))
            print('Epoch[{}]({}/{}): Loss: {:.4f} Potential Energy mae:{:.4f} '.format(epoch,
                                                                                                         index,
                                                                                                         len(train_loader),
                                                                                                         loss.item(),
                                                                                                         mae(energy.cpu(),
                                                                                                             y_energy.cpu()).view(
                                                                                                             -1)[0],
                                                                                                         ))
        if args.is_validate:
            epoch_score = validate_model(model, loader=validate_loader)
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
        model_name = args.task + '_db_HEA_' + args.transform + '_' + str(args.epochs)
        saved_dir = './saved_models_db_HEA/'
        torch.save(model.state_dict(), saved_dir + model_name + '.pt')
        if args.is_validate:
            best_model_name = args.task + '_mp_' + args.transform + '_' + str(args.epochs) + '_best'
            torch.save(best_model_wts, saved_dir + best_model_name + '.pt')
            print('saved in {}'.format(saved_dir + best_model_name + '.pt'))
        print('saved in {}'.format(saved_dir + model_name + '.pt'))
    return model


def test():
    # model_state = torch.load(os.path.join('./saved_models/', args.saved_model))
    # model_name = './saved_models/k_mp_log_50.pt' # good results with mae of 0.2077
    # model_name = './saved_models_lin/g_mp_log_60.pt'
    model_name = './saved_models_lin_256/eg_mp_scaling_500_best.pt'
    # model_name = './saved_models_lin/ef_mp_scaling_50.pt'
    model_state = torch.load(model_name)
    model.load_state_dict(model_state)
    model.eval()  # freeze the dropout and BN layer
    y_pred = []
    y_true = []
    with torch.no_grad():
        end_time = 0
        # for batch in train_loader:
        for batch in test_loader:
            start_time = time.time()
            delta_t0 = end_time - start_time
            # for batch in test_loader:
            batch.to(device)
            out = model(batch.atomic_numbers.long(), batch.pos, batch=batch.batch)
            delta_t = time.time() - start_time
            # It does not change the variable at default
            # However, we directly compare the log-form data to compare with previous paper.
            if args.transform == 'log':
                y_label = transform(args.transform, batch.__getitem__(args.task), forward=True)
            else:
                y_label = batch.__getitem__(args.task)
            y_true.append(y_label)
            end_time = time.time()
            # It should scale into the raw range due to the prediction enlarge the original data in case of scaling.
            # However, we directly compare the log-form data to compare with previous paper.
            if args.transform != 'log':
                out = transform(args.transform, out, forward=False)
            y_pred.append(out)

    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()

    print(y_true.shape, y_pred.shape)
    print(y_true, y_pred)
    # is_transform = False
    # if is_transform:
    #     y_true = np.log10(y_true)
    #     y_pred = np.log10(y_pred)
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
    scatter_hist(y_true, y_pred, fig_path='./', r2_s=r2_s, mae_s=mae_s)


def main():
    import os
    is_train = True
    if is_train:
        # When train, use the following codes.
        # CUDA_VISIBLE_DEVICES=6  python trainer_heanet.py --epochs 10 --batch_size 64 --weight_decay 1e-4  --task k --transform log  --train True
        train(model=model)
    else:
        # When test, use the following codes.
        # python trainer_heanet.py --task g --transform log
        plot()


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=5  python trainer_heanet.py
    # train()
    # test()
    main()

# my own loss, epoch=100:
# 0.8536610537600162 0.22127795

# L1 loss, epoch=50:
#

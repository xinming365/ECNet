import copy
from typing import Tuple

import numpy as np
from torch_geometric.nn.models import SchNet, DimeNet
from torch_geometric.loader import DataLoader
from datasets.HEA_dataset import HEADataset
import torch
from utils.meter import mae
import torch.nn as nn
from torch import Tensor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.heanet import HeaNet
import argparse
from utils.registry import registry, setup_imports
import time
from datasets.Mp_dataset import MpDataset, load_dataset, MpGeometricDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import SequentialSampler
from utils.utility import DataTransformer
import os
from trainer_heanet_mtl import mtl_criterion, evaluate

data_transform = DataTransformer()
device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')

current_file_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_file_dir, 'HEA_Data/POSCARS')


# label_file = os.path.join(current_file_dir, 'HEA_Data/Out_labels/Database.xlsx')


def load_hea_data_single_file(split_type=2, is_validate=True, batch_size=128, RANDOM_SEED=1454880):
    """
    This function defines the manner of training/testing dataset.
    Loading one single data file  with randomly splitting into train/validate/test data set. type: b
    :param is_validate: bool.
    :param split_type: int.
        0: randomly splitting all the data into the training/testing with a specific ratio.
        1: choose the low-component alloy as the training data, and make the high-component alloy as the testing data.
    :return: the object of the Dataset.
    """
    data_file = ''
    current_file_dir = os.path.dirname(__file__)
    if split_type == 0:
        data_file = os.path.normpath(os.path.join(current_file_dir, 'HEA_Data/Out_labels/Qua+Qui.xlsx'))
        print('loading (4+5) component HEA data with randomly splitting all the data into training/testing set\t'
              'from {}.'.format(data_file))
    elif split_type == 1:
        data_file = os.path.normpath(os.path.join(current_file_dir, 'HEA_Data/Out_labels/B+T.xlsx'))
        print('loading (2+3) component HEA data with randomly splitting all the data into training/testing set\t'
              'from {}'.format(data_file))
    elif split_type == 2:
        data_file = os.path.normpath(os.path.join(current_file_dir, 'HEA_Data/Out_labels/Database.xlsx'))
        print('loading (2+3+4+5) component HEA data with randomly splitting all the data into training/testing set\t'
              'from {}'.format(data_file))
    else:
        raise AssertionError('Currently, we only supprot split=0/1 for randomly splitting and splitting based on '
                             'the number of  components')
    total_dataset = HEADataset(poscar_dir=data_dir, label_name=data_file)
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

        train_loader = DataLoader(dataset=total_dataset, batch_size=batch_size,
                                  sampler=SequentialSampler(train_seq))
        validate_loader = DataLoader(dataset=total_dataset, batch_size=batch_size,
                                     sampler=SequentialSampler(v_seq))
        print('The number of the train size is {}, validation size is {}'.format(len(train_seq), len(v_seq)))
    else:
        train_seq = seq[:tv_size]
        train_loader = DataLoader(dataset=total_dataset, batch_size=batch_size,
                                  sampler=SequentialSampler(train_seq))
    test_loader = DataLoader(dataset=total_dataset, batch_size=batch_size,
                             sampler=SequentialSampler(test_seq))

    if is_validate:
        return train_loader, validate_loader, test_loader
    else:
        return train_loader, test_loader


def load_hea_hcp(batch_size=128):
    """
    This function provide a way to load hcp data.
    :return: dataloader
    """

    print('loading the hcp data.\n')
    train_file = './HEA_Data/Out_labels/Ternary_hcp.xlsx'
    hcp_dataset = HEADataset(poscar_dir=data_dir, label_name=train_file, exclude_property='emix')
    hcp_data_loader = DataLoader(dataset=hcp_dataset, batch_size=batch_size)
    return hcp_data_loader


def load_hea_data(split_type=0, is_validate=True):
    """
    This function defines the manner of training/testing dataset.
    Loading processed dataset with training data and test data manually.  type: a
    :param is_validate: bool.
    :param split_type: int.
        0: randomly splitting all the data into the training/testing with a specific ratio.
        1: choose the low-component alloy as the training data, and make the high-component alloy as the testing data.
    :return: the object of the Dataset.
    """
    train_file, test_file = '', ''
    if split_type == 0:
        print('loading HEA data with randomly splitting all the data into training/testing set.')
        train_file = './HEA_Data/Out_labels/train.xlsx'
        test_file = './HEA_Data/Out_labels/test.xlsx'
    elif split_type == 1:
        print('loading HEA data, and make low-component (2+3+4) alloys as the training data, while the high-component '
              '(5) alloy as the testing data. ')
        train_file = './HEA_Data/Out_labels/B+T+Qua.xlsx'
        test_file = './HEA_Data/Out_labels/Quinary.xlsx'
    elif split_type == 2:
        print('loading HEA data, and make low-component (2+3) alloys as the training data, while the high-component '
              '(4+5) alloy as the testing data. ')
        train_file = './HEA_Data/Out_labels/B+T.xlsx'
        test_file = './HEA_Data/Out_labels/Qua+Qui.xlsx'
    elif split_type == 3:
        print('loading HEA data, and make low-component (4+5) alloys as the training data, while the high-component '
              '(2+3) alloy as the testing data. ')
        train_file = './HEA_Data/Out_labels/Qua+Qui.xlsx'
        test_file = './HEA_Data/Out_labels/B+T.xlsx'
    else:
        raise AssertionError('Currently, we only supprot split=0/1 for randomly splitting and splitting based on '
                             'the number of  components')
    train_dataset = HEADataset(poscar_dir=data_dir, label_name=train_file)
    test_dataset = HEADataset(poscar_dir=data_dir, label_name=test_file)
    validate_loader = None  # declare the validate loader
    if is_validate:
        train_size = len(train_dataset)
        t_seq, v_seq = train_test_split(range(train_size), test_size=0.1,
                                        random_state=RANDOM_SEED)  # 1/10 is used to validate the model.
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  sampler=SequentialSampler(t_seq))
        validate_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                     sampler=SequentialSampler(v_seq))
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    if is_validate:
        return train_loader, validate_loader, test_loader
    else:
        return train_loader, test_loader


def transfer_state_dict(pretrained_dict, model_dict):
    """
    We need to remove some unexpected parameters in order to transfer into new networks.
    :param
        pretrained_dict:  dict. The pretrained dict of parameters.
    :param
        model_dict: dict. The new model parameters.
    :return:
        dict. the new model_dict with removed parameters.
    """
    state_dict = {}  # save new model parameters.
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # remove the tower structures
            if 'tower_layers' in k:
                print('Remove the tower layer parameters of {}'.format(k))
            else:
                state_dict[k] = v
        else:
            print('Missing  keys in state_dict:{}'.format(k))

    return state_dict


def fine_tune(model, criterion):
    saved_model = './saved_models_mtl_HEA/mtl_6_HEA/mtl_6_HEA_500_b1_best.pt'

    ######## fine-tune the etot_ef DL model.
    # saved_model = './saved_models_mtl_HEA/mtl_2_etot_ef_HEA_500_b2_best.pt'
    # saved_model = './saved_models_mtl_HEA/mtl_3_etot_emix_ef_HEA_500_b1_best.pt'
    # saved_model = './saved_models_mtl_HEA/mtl_2_ms_mb_HEA_500_b1_best.pt'
    # saved_model = './saved_models_mtl_HEA/mtl_1_rmsd_HEA_500_b1_best.pt'
    pretrained_model_state = torch.load(saved_model, map_location=device)
    model_state = model.state_dict()
    ##### update the initialized parameters with pretrained model without tower layers. TL-2
    # pretrained_dict = transfer_state_dict(pretrained_model_state, model_state)

    ##### update the initialized parameters with pretrained model with tower layers. TL-1
    pretrained_dict = model_state
    model_state.update(pretrained_dict)
    model.load_state_dict(model_state)
    # for param in model.parameters():
    #     param.requires_grad = True
    model_ft = train(model, criterion)


def validate_model_hea(model, loader, tasks, transforms):
    """
    :param model:
    :param loader:
    :return:
    """
    out_pred = [[] for i in range(len(tasks))]
    out_true = [[] for i in range(len(tasks))]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            out = model(batch.atomic_numbers.long(), batch.pos, batch=batch)
            out = [y.squeeze() for y in out]
            assert isinstance(tasks, list), ' task parameter must be a list in the multi-task learning cases.'
            y_pred = [[] for i in range(len(tasks))]
            y_true = [[] for i in range(len(tasks))]
            for ix, task in enumerate(tasks):
                y = batch.__getitem__(task)  # obtain the true labels of the data set.
                yhat = out[ix]
                if transforms is not None and len(transforms):
                    yhat = data_transform.transform(transforms[ix], out[ix], inverse_trans=True)
                y_pred[ix].append(y)
                y_true[ix].append(yhat)
            for ix in range(len(tasks)):
                out_pred[ix].extend(y_pred[ix])
                out_true[ix].extend(y_true[ix])
    # for y_h, y_t in zip(y_true, y_pred)
    for i in range(len(tasks)):
        out_pred[i] = torch.cat(out_pred[i], dim=0).detach().cpu().numpy()
        out_true[i] = torch.cat(out_true[i], dim=0).detach().cpu().numpy()
    return out_pred, out_true


def train(model, criterion):
    print(model)
    # initialize these variables.
    best_score = 10000
    best_model_wts = ''
    for epoch in range(num_epochs):
        for index, batch in enumerate(train_loader):
            model.train(mode=True)  # keep per batch send into the mode of train.
            batch.to(device, non_blocking=True)
            batch.pos.requires_grad = True
            out = model(batch.atomic_numbers.long(), batch.pos, batch=batch)
            out = [y.squeeze() for y in out]
            y_list = []
            assert isinstance(args.task, list), ' task parameter must be a list in the multi-task learning cases.'
            for ix, task in enumerate(args.task):
                y = batch.__getitem__(task)
                if len(args.transform):  # transform is not empety.
                    assert len(args.transform) == len(args.task), 'the length of task and transform parameters must be ' \
                                                                  'equal. '
                    y = data_transform.transform(args.transform[ix], batch.__getitem__(task), inverse_trans=False)
                y_list.append(y)
            loss = mtl_criterion(out, y_list, criterion)
            loss.backward()
            optim.step()
            optim.zero_grad()

            print('Epoch[{}]({}/{}): Loss: {:.4f} '.format(epoch, index, len(train_loader),
                                                           loss.item()))
        # print(y, out)
        if args.is_validate:
            if epoch % 10 == 0:
                out_pred, out_true = validate_model_hea(model, loader=validate_loader, tasks=args.task,
                                                        transforms=args.transform)
                epoch_score = evaluate(out_pred, out_true)
                # We observe all parameters being optimized according to the loss in the validation set.
                # scheduler.step(epoch_score)
                print('The score in {} epochs is {}; The current best score is {}'.format(
                    epoch, epoch_score, best_score
                ))
                if epoch_score < best_score:
                    best_score = epoch_score
                    best_model_wts = copy.deepcopy(model.state_dict())

    if args.save_model:
        # model_name = args.task+'_type'+ str(args.split_type)+'_'+ str(args.epochs)
        w = 'a' if is_processed else 'b'
        model_name = 'mtl_' + str(len(args.task))
        task_name = '_'.join(args.task)
        model_name = model_name + '_' + task_name + '_HEA_' + str(args.epochs) + '_' + w + str(args.split_type)
        # saved_dir = './saved_models_mtl_HEA_transfer/'  # For the transfer learning, this is the default dir to save the model.
        saved_dir = './saved_models_mtl_HEA/'
        torch.save(model.state_dict(), saved_dir + model_name + '.pt')
        if args.is_validate:
            best_model_name = model_name + '_best'
            torch.save(best_model_wts, saved_dir + best_model_name + '.pt')
            print('saved in {}'.format(saved_dir + best_model_name + '.pt'))
        print('saved in {}'.format(saved_dir + model_name + '.pt'))
    return model


def test():
    # model_name = './saved_models_mtl_HEA/mtl_6_mp_400_best.pt'
    #### Models for the transferring learning
    model_name = './saved_models_mtl_HEA/mtl_6_HEA_500_b0_best_w_transfer_4+5.pt'
    model_name = './saved_models_mtl_HEA/mtl_6_HEA_500_b0_best_tl-1.pt'
    model_name = './saved_models_mtl_HEA/mtl_6_HEA_500_b0_tl-2.pt'

    #### Normal Models
    # model_name = './saved_models_mtl_HEA/mtl_6_HEA_500_b1_best.pt'
    # model_name = './saved_models_mtl_HEA/mtl_1_HEA_500_b1_best.pt'
    # model_name = './saved_models_mtl_HEA/mtl_3_etot_emix_ef_HEA_500_b0_best.pt'
    # model_name = './saved_models_mtl_HEA/mtl_3_etot_emix_ef_HEA_500_b2.pt'
    # model_name = './saved_models_mtl_HEA/mtl_2_etot_ef_HEA_500_b2_best.pt'
    # model_name = './saved_models_mtl_HEA/mtl_2_HEA_500_b2.pt'
    # model_name = './saved_models_mtl_HEA/mtl_2_ms_mb_HEA_500_b2.pt'
    model_name = './saved_models_mtl_HEA/mtl_1_rmsd_HEA_500_b2_best.pt'
    # model_name = './saved_models_mtl_HEA/mtl_6_HEA_500_b1.pt'
    # model_name = './saved_models_mtl_HEA/mtl_4_HEA_500_b0.pt'
    model_state = torch.load(model_name)
    model.load_state_dict(model_state)
    out_pred, out_true = validate_model_hea(model, test_loader, tasks=args.task, transforms=args.transform)

    score = evaluate(out_pred, out_true)
    print('mae in the test set is {}'.format(score))

    out_pred_train, out_true_train = validate_model_hea(model, train_loader, tasks=args.task, transforms=args.transform)
    score_train = evaluate(out_pred_train, out_true_train)
    print('mae in the train set is {}'.format(score_train))
    return out_true, out_pred, out_pred_train, out_true_train


def test_hcp():
    model_name = './saved_models_mtl_HEA/mtl_3_etot_emix_ef_HEA_500_b2.pt'
    # model_name = './saved_models_mtl_HEA/mtl_2_HEA_500_b2.pt'
    # model_name = './saved_models_mtl_HEA/mtl_2_ms_mb_HEA_500_b2.pt'
    # model_name = './saved_models_mtl_HEA/mtl_1_rmsd_HEA_500_b2_best.pt'
    model_state = torch.load(model_name)
    model.load_state_dict(model_state)
    hcp_data_loader = load_hea_hcp(batch_size=128)
    out_pred, out_true = validate_model_hea(model, hcp_data_loader, tasks=args.task, transforms=args.transform)

    score = evaluate(out_pred, out_true)
    print('mae in the hcp set is {}'.format(score))
    return out_true, out_pred


def plot():
    import matplotlib.pyplot as plt
    out_true, out_pred, out_pred_train, out_true_train = test()
    # test hcp dataset
    # out_true, out_pred = test_hcp()
    from plot_figure import scatter_hist
    for i in range(len(args.task)):
        mae_s = mean_absolute_error(out_true[i], out_pred[i])
        r2_s = r2_score(out_true[i], out_pred[i])
        out_pred[i] = np.squeeze(out_pred[i])
        out_pred_train[i] = np.squeeze(out_pred_train[i])
        scatter_hist(out_true[i], out_pred[i], out_true_train[i], out_pred_train[i], task=args.task[i], fig_path='./',
                     r2_s=r2_s, mae_s=mae_s)


if __name__ == '__main__':
    # arguments for settings.
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=128, help='number of hidden channels for the embeddings')
    parser.add_argument('--n_filters', type=int, default=64, help='number of filters')
    parser.add_argument('--n_interactions', type=int, default=3, help='number of interaction blocks')
    parser.add_argument('--n_gaussian', type=int, default=50, help='number of gaussian bases to expand the distances')
    parser.add_argument('--cutoff', type=float, default=10, help='the cutoff radius to consider passing messages')
    parser.add_argument('--aggregation', type=str, default='add', help='the aggregation scheme ("add", "mean", or "max") \
                                                                       to use in the messages')
    parser.add_argument('--seed', type=int, default=1454880, help='random seed')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate of the algorithm')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--task', type=str, default=[], nargs='+', help='the target to model in the dataset')
    parser.add_argument('--model', type=str, default='heanet', help='the name of the ML model')
    parser.add_argument('--scaling', type=float, default=1000, help='the scaling factor of the target.')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch size of the data loader')
    parser.add_argument('--save_model', type=bool, default=True, help='save the trained model')
    parser.add_argument('--transform', type=str, default=[], nargs='+',
                        help='transform the target proerty, only support log and scaling')
    parser.add_argument('--tower_h1', type=int, default=128,
                        help='validate the model during training the ef and eg tasks')
    parser.add_argument('--tower_h2', type=int, default=64,
                        help='validate the model during training the ef and eg tasks')
    parser.add_argument('--use_pbc', action='store_true', help='Whether to use the periodic boundary condition.')
    parser.add_argument('--train', '-t', action='store_true', help='Training the ECMTL model.')
    parser.add_argument('--predict', '-p', action='store_true', help='Applying the ECMTL to predict something.')
    parser.add_argument('--fine_tune', '-f', action='store_true', help='fine tune the ECMTL model to predict HEAs.')
    parser.add_argument('--is_validate', action='store_true',
                        help='validate the model during training the ef and eg tasks')
    parser.add_argument('--split_type', type=int, default=0, help='the splitting type of the dataset')
    parser.add_argument('--processed_data', action='store_true',
                        help='whether to load the preprocessed data according to number of components.')
    # parser.add_argument('--saved_model', type='str', default='etot_type0_200.pt', help='the trained model')

    args = parser.parse_args()

    setup_imports()
    RANDOM_SEED = args.seed  # 1454880
    is_validate = args.is_validate
    is_processed = args.processed_data
    num_epochs = args.epochs
    scaling = args.scaling

    # Currently, only the heanet developed by us support for the multi-class learning.
    # So, this model initialization is only for HeaNet.
    model = registry.get_model(args.model
                               )(hidden_channels=args.hidden_channels,
                                 num_filters=args.n_filters,
                                 num_interactions=args.n_interactions,
                                 num_gaussians=args.n_gaussian,
                                 cutoff=args.cutoff,
                                 readout=args.aggregation,
                                 dipole=False, mean=None, std=None,
                                 atomref=None, num_tasks=len(args.task),
                                 tower_h1=args.tower_h1,
                                 tower_h2=args.tower_h2,
                                 use_pbc=args.use_pbc,
                                 )

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=args.weight_decay, amsgrad=False)

    criterion = torch.nn.L1Loss()
    model.to(device)

    if is_validate:
        if is_processed:
            print('Loading processed dataset with training data and test data manually')
            train_loader, validate_loader, test_loader = load_hea_data(split_type=args.split_type,
                                                                       is_validate=is_validate)
        else:
            print('Loading one single data file  with randomly splitting into train/validate/test data set')
            train_loader, validate_loader, test_loader = load_hea_data_single_file(split_type=args.split_type,
                                                                                   is_validate=is_validate)
    else:
        if is_processed:
            train_loader, test_loader = load_hea_data(split_type=args.split_type, is_validate=is_validate)
        else:
            train_loader, test_loader = load_hea_data_single_file(split_type=args.split_type, is_validate=is_validate,

                                                                  batch_size=args.batch_size)

    tune_on_hcp = False
    if tune_on_hcp:
        train_loader = load_hea_hcp(batch_size=128)
    if args.train:
        print('The validation is {}'.format(is_validate))
        # When train, use the following codes.
        # CUDA_VISIBLE_DEVICES=6  python trainer_heanet_mtl.py --epochs 10 --batch_size 64 --weight_decay 1e-4  --task k --transform log  --train True
        train(model=model, criterion=criterion)

    if args.predict:
        # python trainer_heanet_mtl_HEA.py --task etot emix eform  --batch_size 128  --is_validate  --split_type 2 -p
        plot()

    if args.fine_tune:
        fine_tune(model=model, criterion=criterion)

# my own loss, epoch=100:
# 0.8536610537600162 0.22127795

# L1 loss, epoch=50:
#

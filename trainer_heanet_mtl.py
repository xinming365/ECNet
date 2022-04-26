import copy
import numpy as np
from torch_geometric.nn.models import SchNet, DimeNet
from torch_geometric.loader import DataLoader
import torch
from utils.meter import mae
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse
from utils.registry import registry, setup_imports
from sklearn.model_selection import train_test_split
from torch.utils.data import SequentialSampler
from utils.utility import DataTransformer
import os
from datasets.Mp_dataset import MpGeometricDataset, RIDataset
from utils.load_data import assign_task_name

device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
data_transform = DataTransformer()


def mtl_criterion(y_hat, y_label, criterion):
    """
    We collect all the losses of different tasks
    :param y_hat: the predicted y by ML model
    :param y_label: the true y in the dataset.
    :return:
        object of criterion
    """
    assert len(y_hat) == len(y_label), 'The lengths of two variables must be consistent!'
    total_loss = torch.zeros(1, device=device)
    nums = len(y_hat)
    for y_h, y_l in zip(y_hat, y_label):
        loss = criterion(y_h, y_l)
        total_loss = total_loss + loss
    return total_loss / nums


def load_mp_data(tasks, batch_size=128, is_validate=True, RANDOM_SEED=1454880):
    """
    Currently, we do not support the validate in the k/g dataset due to its small data size.

    :param RANDOM_SEED:
    :param is_validate:
    :return:
        If the validate set is needed, then the train_set, test_set, and validate_set are returned.
        Otherwise, the train_set and test_set are returned.

    """
    mp_data_name = [['ef'], ['eg'], ['egn'], ['k'], ['g'], ['ef', 'eg'], ['k', 'g'], ['ef', 'egn']]
    ri_data_name = [['ri'], ['og'], ['dg'], ['ri', 'og'], ['ri', 'dg'], ['og', 'dg'], ['ri', 'og', 'dg']]
    current_file_dir = os.path.dirname(__file__)
    mp_file_path = os.path.normpath(os.path.join(current_file_dir, 'datasets/mp'))
    print('loading the processed data from {}'.format(mp_file_path))
    task_name = ''
    if tasks in mp_data_name:
        task_name = assign_task_name(tasks)
        dataset = MpGeometricDataset(task=task_name, root=mp_file_path)  # according to your code position.
    elif tasks in ri_data_name:
        dataset = RIDataset(root=mp_file_path)  # according to your code position.
    else:
        raise Exception(f'Illegal task name: {tasks}, please check the supported tasks.')
    print(f'The total data size is {len(dataset)}.')
    splits = train_test_split(range(len(dataset)), test_size=0.1, random_state=RANDOM_SEED)
    train_seq, test_seq = splits
    validate_loader = None

    # We disrupt the order beforehand.
    seq = np.random.RandomState(seed=RANDOM_SEED).permutation(np.arange(len(dataset)))
    if is_validate:
        if task_name == 'ef':
            print('The training data is set 60000 manually for comparison')
            train_validate_size = 66667  # This number is set for comparison with MODNet.
            # We override the train set and test set here
            train_validate_seq = seq[:train_validate_size]
            test_seq = seq[train_validate_size:]
            train_seq, validate_seq = train_test_split(train_validate_seq, test_size=0.1,
                                                       random_state=RANDOM_SEED)  # 1/10 is used to validate
            validate_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                                         sampler=SequentialSampler(validate_seq),
                                         # num_workers=1, pin_memory=True,
                                         persistent_workers=False)  # the memory is not enough
        else:

            train_seq, validate_seq = train_test_split(train_seq, test_size=0.1,
                                                       random_state=RANDOM_SEED)  # 1/10 is used to validate
            print('The training data is {}, and the validation set is {}'.format(len(train_seq), len(validate_seq)))
            validate_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                                         sampler=SequentialSampler(validate_seq),
                                         num_workers=4, pin_memory=True,
                                         persistent_workers=True)
    else:
        # We directly split the dataset into train and test set.
        pass
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                              sampler=SequentialSampler(train_seq),
                              num_workers=8, pin_memory=True,
                              persistent_workers=True)
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             sampler=SequentialSampler(test_seq),
                             num_workers=4, pin_memory=True)

    if is_validate:
        return train_loader, validate_loader, test_loader
    else:
        return train_loader, test_loader


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
    nums = len(test_data)
    score = 0
    for i in range(nums):
        score_i = mean_absolute_error(test_data[i], predict_data[i])
        score = score + score_i
        print('the score of task {} is {}\n'.format(i, score_i))
    return score / nums


def validate_model(model, loader, tasks, transforms):
    """
    The transformation makes this function a little complicated.
     The idea is that when predicting 'ef' or 'eg',
    we need to obtain the prediction/scaling to make the prediction consistent with true label. When predicting
    'k' or 'g', we need to obtain the np.log10(true_k) to make it consistent.

    :param model: Model
    :param loader: Dataloader
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
                yhat = out[ix]  # obtain the predicted labels of the data set.
                if len(transforms):  # transform is not empety.
                    # print(f'the prediction task is {task}')
                    if task == 'k' or task == 'g':
                        # we treat 'k' or 'g' of tru labels with 'log' transformation.
                        y = data_transform.transform(transforms[ix], batch.__getitem__(task), inverse_trans=False)
                    else:
                        # we recover the predictions of 'ef' and 'eg' in order to make a direct comparison.
                        yhat = data_transform.transform(transforms[ix], out[ix], inverse_trans=True)
                y_true[ix].append(y)
                y_pred[ix].append(yhat)
            for ix in range(len(tasks)):
                out_pred[ix].extend(y_pred[ix])
                out_true[ix].extend(y_true[ix])
    for i in range(len(tasks)):
        out_pred[i] = torch.cat(out_pred[i], dim=0).detach().cpu().numpy()
        out_true[i] = torch.cat(out_true[i], dim=0).detach().cpu().numpy()
    return out_pred, out_true


def train(model, epochs, optim, train_loader, validate_loader,
          tasks, transform, criterion,
          save_model=True, device='cpu'):
    best_score = 10000  # initialize the score for saving the best model in the validation set.
    best_model_wts = ''

    for epoch in range(epochs):
        for index, batch in enumerate(train_loader):
            model.train(mode=True)  # keep per batch send into the mode of train.
            batch.to(device, non_blocking=True)
            batch.pos.requires_grad = True
            out = model(batch.atomic_numbers.long(), batch.pos, batch=batch)
            out = [y.squeeze() for y in out]
            y_list = []
            assert isinstance(tasks, list), ' task parameter must be a list in the multi-task learning cases.'
            for ix, task in enumerate(tasks):
                y = batch.__getitem__(task)
                if len(transform):  # transform is not empety.
                    assert len(transform) == len(tasks), 'the length of task and transform parameters must be ' \
                                                        'equal. '
                    # perform the data transformation according to the 'trans_type'.
                    y = data_transform.transform(transform[ix], batch.__getitem__(task), inverse_trans=False)
                y_list.append(y)
            loss = mtl_criterion(out, y_list, criterion)
            loss.backward()
            optim.step()
            optim.zero_grad()

            print('Epoch[{}]({}/{}): Loss: {:.4f} '.format(epoch, index, len(train_loader),
                                                           loss.item()))
        # print(y, out)
        # scheduler.step()
        if validate_loader is not None:
            out_pred, out_true = validate_model(model, loader=validate_loader, tasks=tasks, transforms=transform)
            epoch_score = evaluate(out_pred, out_true)
            # We observe all parameters being optimized according to the loss in the validation set.
            # scheduler.step(epoch_score)
            print('The score in {} epochs is {}; The current best score is {}'.format(
                epoch, epoch_score, best_score
            ))
            if epoch_score < best_score:
                best_score = epoch_score
                best_model_wts = copy.deepcopy(model.state_dict())

    if save_model:
        # model_name = task+'_type'+ str(split_type)+'_'+ str(epochs)
        model_name = 'mtl_' + str(len(tasks)) + '_mp_' + str(epochs)
        saved_dir = './saved_models_mtl/'
        os.makedirs(saved_dir, exist_ok=True)
        torch.save(model.state_dict(), saved_dir + model_name + '.pt')
        if validate_loader is not None:
            best_model_name = model_name + '_best'
            torch.save(best_model_wts, saved_dir + best_model_name + '.pt')
            print('saved in {}'.format(saved_dir + best_model_name + '.pt'))
        print('saved in {}'.format(saved_dir + model_name + '.pt'))
    return model


def test():
    # model_name = './saved_models_mtl/mtl_2_mp_ef_eg_128_64_400_best.pt' # multi-target model
    # model_name = './saved_models_lin_256/k_mp_log_500_best_256.pt' # single target model
    # model_name = './saved_models_mtl/mtl_1_mp_ef_500_best.pt'  # single-target model
    # model_name='./saved_models_mtl/mtl_2_mp_k_g_128_64_500_best.pt'
    model_name = './saved_models_mtl/mtl_2_mp_500_best.pt'
    model_state = torch.load(model_name)
    model.load_state_dict(model_state)
    out_pred, out_true = validate_model(model, test_loader, tasks=args.task, transforms=args.transform)
    score = evaluate(out_pred, out_true)
    print('mae in the test set is {}'.format(score))
    return out_true, out_pred


def plot():
    out_true, out_pred = test()
    from plot_figure import scatter_hist
    for i in range(len(args.task)):
        mae_s = mean_absolute_error(out_true[i], out_pred[i])
        mae_s_orig = mean_absolute_error(out_true[i], out_pred[i])
        r2_s = r2_score(out_true[i], out_pred[i])
        out_pred[i] = np.squeeze(out_pred[i])
        scatter_hist(out_true[i], out_pred[i], fig_path='./', r2_s=r2_s, mae_s=mae_s)


if __name__ == '__main__':
    # arguments for settings.
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=128, help='number of hidden channels for the embeddings')
    parser.add_argument('--n_filters', type=int, default=128, help='number of filters')
    parser.add_argument('--n_interactions', type=int, default=6, help='number of interaction blocks')
    parser.add_argument('--n_gaussian', type=int, default=50, help='number of gaussian bases to expand the distances')
    parser.add_argument('--cutoff', type=float, default=10, help='the cutoff radius to consider passing messages')
    parser.add_argument('--aggregation', type=str, default='add', help='the aggregation scheme ("add", "mean", or "max") \
                                                                       to use in the messages')
    parser.add_argument('--seed', type=int, default=1454880, help='random seed')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate of the algorithm')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--task', type=str, default=['k', 'g'], nargs='+', help='the target to model in the dataset')
    parser.add_argument('--split_type', type=int, default=0, help='the splitting type of the dataset')
    parser.add_argument('--model', type=str, default='heanet', help='the name of the ML model')
    parser.add_argument('--scaling', type=float, default=1000, help='the scaling factor of the target.')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch size of the data loader')
    parser.add_argument('--save_model', type=bool, default=True, help='save the trained model')
    parser.add_argument('--transform', type=str, default=[], nargs='+',
                        help='transform the target proerty, only support log and scaling')
    parser.add_argument('--is_validate', action='store_true',
                        help='validate the model during training the ef and eg tasks')
    parser.add_argument('--tower_h1', type=int, default=64,
                        help='validate the model during training the ef and eg tasks')
    parser.add_argument('--tower_h2', type=int, default=64,
                        help='validate the model during training the ef and eg tasks')
    parser.add_argument('--train', '-t', action='store_true', help='Training the ECMTL model.')
    parser.add_argument('--predict', '-p', action='store_true', help='Applying the ECMTL to predict something.')
    # parser.add_argument('--saved_model', type='str', default='etot_type0_200.pt', help='the trained model')

    args = parser.parse_args()
    setup_imports()
    num_epochs = args.epochs
    scaling = args.scaling
    RANDOM_SEED = args.seed  # 1454880
    is_validate = args.is_validate
    tasks = args.task
    bs = args.batch_size

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
                                 tower_h2=args.tower_h2
                                 )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay, amsgrad=False)
    criterion = torch.nn.L1Loss()

    if is_validate:
        train_loader, validate_loader, test_loader = load_mp_data(tasks=tasks, batch_size=bs, is_validate=is_validate,
                                                                  RANDOM_SEED=RANDOM_SEED)
    else:
        train_loader, test_loader = load_mp_data(tasks=tasks, batch_size=bs, is_validate=is_validate,
                                                 RANDOM_SEED=RANDOM_SEED)
        validate_loader = None

    if args.train:
        print('The validation is {}'.format(is_validate))
        # When train, use the following codes.
        # CUDA_VISIBLE_DEVICES=6  python trainer_heanet_mtl.py --epochs 10 --batch_size 64 --weight_decay 1e-4  --task k --transform log  --train True
        train(model=model, epochs=num_epochs, optim=optimizer,
              train_loader=train_loader, validate_loader=validate_loader,
              tasks=args.task, transform=args.transform,criterion=criterion,
              save_model=args.save_model, device=device)

    if args.predict:
        plot()

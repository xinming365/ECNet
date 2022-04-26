import numpy as np
from torch_geometric.nn import SchNet, DimeNet
from torch_geometric.data import DataLoader
from datasets.HEA_dataset import HEADataset
import torch
from utils.meter import mae
import torch.nn as nn
from torch import Tensor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorboardX import SummaryWriter



class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, natoms: Tensor, output: Tensor):
        delta = (input-output)/natoms
        p=50
        return p*torch.mean(torch.pow(delta, 2))

device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
model = SchNet(hidden_channels=128, num_filters=128,
               num_interactions=6, num_gaussians=50, cutoff=40.0,
               readout='add', dipole=False, mean=None, std=None,
               atomref=None)


optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False)


criterion = torch.nn.L1Loss()
# criterion = MyLoss()

data_dir = './HEA_Data/POSCARS'
label_file = './HEA_Data/Out_labels/Database.xlsx'
# dataset  = HEADataset(poscar_dir=data_dir, label_name=label_file)
# train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

def load_data(split_type=0):
    """
    This function defines the manner of training/testing dataset.
    :param split_type: int. 
    0: randomly splitting all the data into the training/testing with a specific ratio.
    1: choose the low-component alloy as the training data, and make the high-component alloy as the testing data.
    :return: the object of the Dataset.
    """
    train_file, test_file = '', ''
    if split_type==0:
        train_file = './HEA_Data/Out_labels/train.xlsx'
        test_file = './HEA_Data/Out_labels/test.xlsx'
    elif split_type==1:
        train_file = './HEA_Data/Out_labels/B+T+Qua.xlsx'
        test_file = './HEA_Data/Out_labels/Quinary.xlsx'
    else:
        pass
    train_dataset  = HEADataset(poscar_dir=data_dir, label_name=train_file, task='etot')
    test_dataset  = HEADataset(poscar_dir=data_dir, label_name=test_file, task='etot')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)
    return train_loader, test_loader

train_loader, test_loader = load_data(split_type=0)
num_epochs=300

p2 = 1000
#save_models = True

model.to(device)


def train(model_name=None):
    writer = SummaryWriter(comment='learning rate')
    threshold = 100000
    model.train(mode=True)
    print(model)
    for epoch in range(num_epochs):
    
        for index, batch in enumerate(train_loader):
            batch.to(device)
            # print(batch)
            batch.pos.requires_grad=True
            out = model(batch.atomic_numbers.long(), batch.pos, batch=batch.batch)
            # print(batch.batch)
            out = out.squeeze()

            # loss = criterion(forces, 1000* batch.forces)
            # loss = criterion(out, p2* batch.rmsd)
            # This is the setted parameters for the rmsd.
            # loss = criterion(out, batch.num_atoms, p2* batch.rmsd)

            # loss = criterion(out, p* batch.rmsd)
            # This is the setting parameters for the etot.
            # loss = criterion(out, batch.num_atoms, p2*batch.etot)

            # This is the setting parameters for the eform.
            # loss = criterion(out, batch.num_atoms, p2*batch.eform)

            # This is the setting parameters for the emix.
            loss = criterion(out, p2*batch.etot)

            # This is the setting parameters for the ms.
            # loss = criterion(out, p2*batch.ms)
            loss.backward()
            optim.step()
            optim.zero_grad()

            acc = (mae(out.cpu(), p2*batch.etot.cpu()).view(-1)[0])/p2
            print('Epoch[{}]({}/{}): Loss: {:.4f} Potential Energy mae:{:.4f} '.format(epoch,
                                                                                    index, len(train_loader), loss.item(),
                                                                                                        acc,
                                                                                                        ))
        writer.add_scalar('train/Loss', loss.item(), epoch)
        writer.add_scalar('Accuracy', acc, epoch)
        
        # print(out, batch.etot)
        # if model_name:
        #     if acc < threshold:
        #         threshold = acc
        #         print('save model at epoch {}'.format(epoch))
        #         torch.save(model.state_dict(), './saved_models/'+model_name+'.pt')


    writer.close()
    if model_name:
        torch.save(model.state_dict(), './saved_models/'+model_name+'_{}'.format(epoch)+'.pt')
        
    

def test():
    # model_state = torch.load('./saved_models/modle_emix_type1_199.pt')
    model_state = torch.load('./saved_models/modle_etot_L1_299.pt')
    model.load_state_dict(model_state)
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        # for batch in train_loader:
        for batch in test_loader:
            batch.to(device)
            out = model(batch.atomic_numbers.long(), batch.pos, batch=batch.batch)
            y_true.append(batch.etot)
            print(out)
            # print(batch.etot)
            y_pred.append(out)

    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()/p2


    print(y_true.shape, y_pred.shape)
    # print(y_true, y_pred)
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

if __name__=='__main__':
    # CUDA_VISIBLE_DEVICES=5  python trainer_schnet.py
    # train(model_name='modle_etot_L1')
    # test()
    plot()


# my own loss, epoch=100:
# 0.8536610537600162 0.22127795

# L1 loss, epoch=50:
# 
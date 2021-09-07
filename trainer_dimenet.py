import numpy as np
from torch_geometric.nn import DimeNet
from torch_geometric.data import DataLoader
from datasets.md_job_dataset import MdDataset
from utils.meter import mae
import os
import torch

device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
model = DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                num_bilinear=8, num_spherical=7, num_radial=6,
                cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                num_after_skip=2, num_output_layers=3)

model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                         weight_decay=0.01, amsgrad=False)
criterion = torch.nn.L1Loss()

data_dir = './job'
train_loader = DataLoader(dataset=MdDataset(data_dir), batch_size=128, shuffle=True)
num_epochs=50
for epoch in range(num_epochs):
    model.train(mode=True)

    for index, batch in enumerate(train_loader):
        batch.to(device)
        batch.pos.requires_grad=True
        energy = model(batch.atomic_numbers.long(), batch.pos, batch=batch.batch)
        forces =  -1 * (
            torch.autograd.grad(
                energy,
                batch.pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )
        # loss = criterion(forces, 1000* batch.forces)
        loss = criterion(energy,  batch.y)
        # loss = criterion(energy, batch.y/160) + criterion(forces, 1000*batch.forces)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # print('Epoch[{}]({}/{}): Loss: {:.4f} Potential Energy mae:{:.4f} Force_x mae:{:.4f}'.format(epoch,
        #  index, len(train_loader), loss.item(),
        #  mae(energy.cpu(), batch.y.cpu()).view(-1)[0],
        #  mae(forces.cpu(), batch.forces.cpu())[0]))
        if index%2==0:
            print('Epoch[{}]({}/{}): Loss: {:.4f} Potential Energy mae:{:.4f} Force_x mae:{:.4f}'.format(epoch,
                                                                                                         index, len(train_loader), loss.item(),
                                                                                                         mae(energy.cpu(), batch.y.cpu()).view(-1)[0],
                                                                                                         mae(forces.cpu()/1000, batch.forces.cpu())[0]))

    if epoch%3==0:
        print(energy, batch.y)
        print(forces/1000, batch.forces)

if __name__=='__main__':
    pass
import numpy as np
from torch_geometric.nn import SchNet, DimeNet
from torch_geometric.data import DataLoader
from datasets.md_job_dataset import MdDataset
import torch
from utils.meter import mae


device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
model = SchNet(hidden_channels=128, num_filters=128,
               num_interactions=6, num_gaussians=50, cutoff=10.0,
               readout='add', dipole=False, mean=None, std=None,
               atomref=None)


optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False)
criterion = torch.nn.L1Loss()

data_dir = './job'
train_loader = DataLoader(dataset=MdDataset(data_dir), batch_size=16, shuffle=True)
num_epochs=5
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
        loss = criterion(energy, batch.y) + 100*criterion(forces, batch.forces)
        loss.backward()
        optim.step()
        optim.zero_grad()

        print('Epoch[{}]({}/{}): Loss: {:.4f} Potential Energy mae:{:.4f} Force_x mae:{:.4f}'.format(epoch,
                                                                                                 index, len(train_loader), loss.item(),
                                                                                                     mae(energy.cpu(), batch.y.cpu()).view(-1)[0],
                                                                                                     mae(forces.cpu(), batch.forces.cpu())[0]))
    print(energy, batch.y)
    print(forces, batch.forces)


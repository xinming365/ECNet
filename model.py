import torch
from utils.registry import registry, setup_imports
from trainer_heanet_mtl import validate_model, evaluate, load_mp_data
from datasets.preprocessing import PoscarToGraph
from trainer_heanet_mtl import train, validate_model, evaluate


class ECModel:
    """ Assign the parameters for model and datasets.

    When tasks come from Multi-target learning model, it should be a list including str, such as ['ef', 'eg'].
    At the same time, the transforms should be like ['scaling', 'scaling']
    """

    def __init__(self, tasks, transform):
        self.device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model(num_tasks=len(tasks))
        self.tasks = tasks
        self.transform = transform
        self.criterion = torch.nn.L1Loss()

    def train(self, num_epochs, train_loader, validate_loader, save_mode=False):
        """
        
        :param num_epochs: number of epochs to train the model
        :return: 
        """
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=0, amsgrad=False)
        self.model = train(model=model, epochs=num_epochs, optim=optimizer,
                           train_loader=train_loader, validate_loader=validate_loader,
                           tasks=self.tasks, transform=self.transform, criterion=self.criterion,
                           save_model=save_mode, device=self.device)

    def test(self, test_loader):
        """

        :param test_loader: Dataloader
        :return:
        """
        model = self.model
        out_pred, out_true = validate_model(model, test_loader, tasks=self.tasks, transforms=self.transform)
        score = evaluate(out_pred, out_true)
        print('mae in the test set is {}'.format(score))
        return out_true, out_pred

    def predict(self, poscar):
        """
        Given a structure in 'POSCAR' format, it will predict its properties according to your requirements.

        """

        model = self.model
        model.eval()
        data = self.convert_data(poscar)
        out = model(data.atomic_numbers.long(), data.pos)
        return out

    def convert_data(self, filename):
        """
        the file name of the poscars
        args:
            filename: str. './HEA_Data/Binary_POSCAR_Files/POSCAR_Co3Cr_sqsfcc'
        """
        pg = PoscarToGraph(radius=10, max_neigh=200)
        data = pg.to_graph(filename)
        data.to(device=self.device)
        return data

    def init_model(self, hidden_channels=128, n_filters=64, n_interactions=3,
                   n_gaussians=50, cutoff=10, num_tasks=2, tower_h1=128,
                   tower_h2=64):
        """
        Initialize the ML models.
        It should be noted that the hyper parameters are assigned according to the specific trained hyper parameters.
        """
        setup_imports()
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
        model.to(self.device)
        self.model = model
        return model

    def load_model(self, model_name: str):
        """
        load the trained ML models given the model_name.

        args:
            model_name: str. the name of the trained model.
            For example: './saved_models/ms_type0_300.pt'
        """
        # load the ML model.
        model = self.model
        # load parameters of trained model
        model_state = torch.load(model_name, map_location=self.device)
        model.load_state_dict(model_state)
        self.model = model
        return model


if __name__ == '__main__':
    tasks = ['ef']
    transform = ['scaling']

    ecmodel = ECModel(tasks=tasks, transform=transform)

    # train and test the ECMTL model.

    # dataset = MpGeometricDataset(task=tasks, root='./datasets/mp')
    # train_loader, validate_loader, test_loader = LoadDataset(dataset=dataset).load(fraction=1)
    # ecmodel.train(num_epochs=10, train_loader=train_loader, validate_loader=validate_loader, save_mode=False)
    # out_true, out_pred = ecmodel.test(test_loader=test_loader)

    # load the trained model.
    # the initializing parameters should be consistent with the trained model.
    ecmodel.init_model(num_tasks=len(tasks))
    ecmodel.load_model(model_name='./saved_models_mtl/mtl_1_mp_ef_500_best.pt')
    train_loader, validate_loader, test_loader = load_mp_data(tasks=tasks, is_validate=True)
    ecmodel.test(test_loader=test_loader)

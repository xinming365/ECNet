from model import ECModel
from trainer_heanet_mtl import load_mp_data


def test_model(tasks, transform, model_name):
    """
    The brief function to test our models. All the other parameters are default ones.
    :param tasks: list of str
    :param transform: list of str
    :param model_name: str
    :return:
    """
    ecmodel = ECModel(tasks=tasks, transform=transform)
    ecmodel.init_model(num_tasks=len(tasks))
    ecmodel.load_model(model_name=model_name)
    train_loader, validate_loader, test_loader = load_mp_data(tasks=tasks, is_validate=True)
    out_true, out_pred = ecmodel.test(test_loader=test_loader)
    return out_true, out_pred


def test_ef():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['ef']
    transform = ['scaling']
    model_name = './saved_models_mtl/mtl_1_mp_ef_500_best.pt'
    out_true, out_pred = test_model(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred

def test_eg():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['eg']
    transform = ['scaling']
    model_name = './saved_models_mtl/mtl_1_mp_eg_500_best.pt'
    out_true, out_pred = test_model(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred

def test_egnz():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['egn']
    transform = ['scaling']
    model_name = './saved_models_mtl/mtl_1_mp_egnz_500_best.pt'
    out_true, out_pred = test_model(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred

def test_g():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['g']
    transform = ['log']
    model_name = './saved_models_mtl/mtl_1_mp_g_500_best.pt'
    out_true, out_pred = test_model(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred

def test_k():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['k']
    transform = ['log']
    model_name = './saved_models_mtl/mtl_1_mp_k_500_best.pt'
    out_true, out_pred = test_model(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred

def test_n():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['ri']
    transform = ['scaling']
    model_name = './saved_models_mtl/mtl_1_mp_ri_500_best.pt'
    out_true, out_pred = test_model(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred

# def test_ef():
#     """
#     load and test the trained model. These are trained models of my previous experiments.
#     :return:
#     """
#     tasks = ['ef']
#     transform = ['scaling']
#     model_name = './saved_models_mtl/mtl_1_mp_ef_500_best.pt'
#     out_true, out_pred = test_model(tasks=tasks, transform=transform, model_name=model_name)
#     return out_true, out_pred
#
# def test_ef():
#     """
#     load and test the trained model. These are trained models of my previous experiments.
#     :return:
#     """
#     tasks = ['ef']
#     transform = ['scaling']
#     model_name = './saved_models_mtl/mtl_1_mp_ef_500_best.pt'
#     out_true, out_pred = test_model(tasks=tasks, transform=transform, model_name=model_name)
#     return out_true, out_pred





if __name__=='__main__':

    test_ef()
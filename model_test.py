from model import ECModel
from trainer_heanet_mtl import load_mp_data
from trainer_heanet_mtl_HEA import load_hea_data_single_file


def evaluate_model_mp(tasks, transform, model_name):
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


def evaluate_ef():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['ef']
    transform = ['scaling']
    model_name = './saved_models_mtl/mtl_1_mp_ef_500_best.pt'
    out_true, out_pred = evaluate_model_mp(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred


def evaluate_eg():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['eg']
    transform = ['scaling']
    model_name = './saved_models_mtl/mtl_1_mp_eg_500_best.pt'
    out_true, out_pred = evaluate_model_mp(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred


def evaluate_egnz():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['egn']
    transform = ['scaling']
    model_name = './saved_models_mtl/mtl_1_mp_egnz_500_best.pt'
    out_true, out_pred = evaluate_model_mp(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred


def evaluate_g():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['g']
    transform = ['log']
    model_name = './saved_models_mtl/mtl_1_mp_g_500_best.pt'
    out_true, out_pred = evaluate_model_mp(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred


def evaluate_k():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['k']
    transform = ['log']
    model_name = './saved_models_mtl/mtl_1_mp_k_500_best.pt'
    out_true, out_pred = evaluate_model_mp(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred


def evaluate_n():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['ri']
    transform = ['scaling']
    model_name = './saved_models_mtl/mtl_1_mp_ri_500_best.pt'
    out_true, out_pred = evaluate_model_mp(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred


def evaluate_n_og_dg():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """

    tasks = ['ri', 'og', 'dg']
    transform = ['scaling', 'scaling', 'scaling']
    model_name = './saved_models_mtl/mtl_3_mp_ri_og_dg_500_best.pt'
    out_true, out_pred = evaluate_model_mp(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred


def evaluate_ef_eg():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['ef', 'eg']
    transform = ['scaling', 'scaling']
    model_name = './saved_models_mtl/mtl_2_mp_400_best.pt'
    # model_name = './saved_models_mtl/mtl_2_mp_ef_eg_128_64_400_best.pt'
    # model_name = './saved_models_mtl/mtl_2_mp_800_best.pt'
    out_true, out_pred = evaluate_model_mp(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred


def evaluate_ef_egn():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['ef', 'egn']
    transform = ['scaling', 'scaling']
    model_name = './saved_models_mtl/mtl_1_mp_ri_500_best.pt'
    out_true, out_pred = evaluate_model_mp(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred


def evaluate_k_g():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['k', 'g']
    transform = ['log', 'log']
    model_name = './saved_models_mtl/mtl_2_mp_k_g_128_64_500_best.pt'
    out_true, out_pred = evaluate_model_mp(tasks=tasks, transform=transform, model_name=model_name)
    return out_true, out_pred


def evaluate_model_hea(tasks, transform, model_name, data_split_type=2):
    """
    The brief function to test our models. All the other parameters are default ones.
    :param data_split_type: int
    :param tasks: list of str
    :param transform: list of str
    :param model_name: str
    :return:
    """
    ecmodel = ECModel(tasks=tasks, transform=transform)
    ecmodel.init_model(num_tasks=len(tasks))
    ecmodel.load_model(model_name=model_name)
    train_loader, validate_loader, test_loader = load_hea_data_single_file(split_type=data_split_type, is_validate=True)
    out_true, out_pred = ecmodel.test(test_loader=test_loader)
    return out_true, out_pred


def evaluate_etot_emix_ef(type='b0'):
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['etot', 'emix', 'eform']
    transform = []
    if type == 'b0':
        model_name = './saved_models_mtl_HEA/mtl_3_etot_emix_ef_HEA_500_b0_best.pt'
        data_split_type = 0
    elif type == 'b1':
        model_name = './saved_models_mtl_HEA/mtl_3_etot_emix_ef_HEA_500_b1_best.pt'
        data_split_type = 1
    elif type == 'b2':
        model_name = './saved_models_mtl_HEA/mtl_3_etot_emix_ef_HEA_500_b2.pt'
        data_split_type = 2
    else:
        return None
    out_true, out_pred = evaluate_model_hea(tasks=tasks, transform=transform,
                                            model_name=model_name, data_split_type=data_split_type)
    return out_true, out_pred


def evaluate_ms_mb(type='b0'):
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['ms', 'mb']
    transform = []
    if type == 'b0':
        model_name = './saved_models_mtl_HEA/mtl_2_ms_mb_HEA_500_b0_best.pt'
        data_split_type = 0
    elif type == 'b1':
        model_name = './saved_models_mtl_HEA/mtl_2_ms_mb_HEA_500_b1_best.pt'
        data_split_type = 1
    elif type == 'b2':
        model_name = './saved_models_mtl_HEA/mtl_2_ms_mb_HEA_500_b2.pt'
        data_split_type = 2
    else:
        return None
    out_true, out_pred = evaluate_model_hea(tasks=tasks, transform=transform,
                                            model_name=model_name, data_split_type=data_split_type)
    return out_true, out_pred


def evaluate_rmsd(type='b0'):
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['rmsd']
    transform = []
    if type == 'b0':
        model_name = './saved_models_mtl_HEA/mtl_1_rmsd_HEA_500_b0_best.pt'
        data_split_type = 0
    elif type == 'b1':
        model_name = './saved_models_mtl_HEA/mtl_1_rmsd_HEA_500_b1_best.pt'
        data_split_type = 1
    elif type == 'b2':
        # model_name = './saved_models_mtl_HEA/mtl_1_rmsd_HEA_500_b2_best.pt'
        model_name = './saved_models_mtl_HEA/mtl_1_rmsd_HEA_500_b2.pt'
        data_split_type = 2
    else:
        return None
    out_true, out_pred = evaluate_model_hea(tasks=tasks, transform=transform,
                                            model_name=model_name, data_split_type=data_split_type)
    return out_true, out_pred


def evaluate_transfer_as_unit(type='tl1'):
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['etot', 'emix', 'eform', 'ms', 'mb', 'rmsd']
    transform = []
    if type == 'tl1':
        model_name = './saved_models_mtl_HEA_transfer/mtl_6_HEA_500_b0_best.pt'
        # model_name = './saved_models_mtl_HEA_transfer/mtl_6_HEA_500_b0.pt'
    elif type == 'tl2':
        model_name = './saved_models_mtl_HEA_transfer/mtl_6_HEA_500_b0_best_tl2.pt'
        # model_name = './saved_models_mtl_HEA_transfer/mtl_6_HEA_500_b0_tl2.pt'
    else:
        return None
    out_true, out_pred = evaluate_model_hea(tasks=tasks, transform=transform,
                                            model_name=model_name, data_split_type=0)
    return out_true, out_pred


def evaluate_transfer_ms_mb():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['ms', 'mb']
    transform = []
    model_name = './saved_models_mtl_HEA_transfer/mtl_2_HEA_500_b0_best_tl1.pt'
    # model_name = './saved_models_mtl_HEA_transfer/mtl_2_HEA_500_b0_tl1.pt'

    model_name = './saved_models_mtl_HEA_transfer/mtl_2_HEA_500_b0_tl2.pt'

    out_true, out_pred = evaluate_model_hea(tasks=tasks, transform=transform,
                                            model_name=model_name, data_split_type=0)
    return out_true, out_pred


def evaluate_transfer_etot_emix_eform():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['etot', 'emix', 'eform']
    transform = []
    # TL-1
    model_name = './saved_models_mtl_HEA_transfer/mtl_3_HEA_500_b0_best_tl1.pt'
    # model_name = './saved_models_mtl_HEA_transfer/mtl_3_HEA_500_b0_tl1.pt'

    model_name = './saved_models_mtl_HEA_transfer/mtl_3_HEA_500_b0_best_tl2.pt'

    out_true, out_pred = evaluate_model_hea(tasks=tasks, transform=transform,
                                            model_name=model_name, data_split_type=0)
    return out_true, out_pred


def evaluate_transfer_rmsd():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['rmsd']
    transform = []
    model_name = './saved_models_mtl_HEA_transfer/mtl_1_HEA_500_b0_best_tl1.pt'
    model_name = './saved_models_mtl_HEA_transfer/mtl_1_HEA_500_b0_tl1.pt'

    model_name = './saved_models_mtl_HEA_transfer/mtl_1_HEA_500_b0_best_tl2.pt'

    out_true, out_pred = evaluate_model_hea(tasks=tasks, transform=transform,
                                            model_name=model_name, data_split_type=0)
    return out_true, out_pred


def evaluate_as_unit_23():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['etot', 'emix', 'eform', 'ms', 'mb', 'rmsd']
    transform = []
    model_name = './saved_models_mtl_HEA/mtl_6_HEA_500_b1.pt'
    model_name = './saved_models_mtl_HEA/mtl_6_HEA_500_b1_best.pt'

    out_true, out_pred = evaluate_model_hea(tasks=tasks, transform=transform,
                                            model_name=model_name, data_split_type=1)
    return out_true, out_pred


def evaluate_as_unit_45():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['etot', 'emix', 'eform', 'ms', 'mb', 'rmsd']
    transform = []
    model_name = './saved_models_mtl_HEA/mtl_6_HEA_500_b0.pt'
    model_name = './saved_models_mtl_HEA/mtl_6_HEA_500_b0_best.pt'

    out_true, out_pred = evaluate_model_hea(tasks=tasks, transform=transform,
                                            model_name=model_name, data_split_type=0)
    return out_true, out_pred


def evaluate_as_unit_tot():
    """
    load and test the trained model. These are trained models of my previous experiments.
    :return:
    """
    tasks = ['etot', 'emix', 'eform', 'ms', 'mb', 'rmsd']
    transform = []
    model_name = './saved_models_mtl_HEA/mtl_6_HEA_500_b2.pt'
    model_name = './saved_models_mtl_HEA/mtl_6_HEA_500_b2_best.pt'

    out_true, out_pred = evaluate_model_hea(tasks=tasks, transform=transform,
                                            model_name=model_name, data_split_type=2)
    return out_true, out_pred




if __name__ == '__main__':
    # evaluate the ECSTL

    evaluate_ef()
    # evaluate_eg()
    # evaluate_egnz()
    # evaluate_k()
    # evaluate_g()
    # evaluate_n()

    # evaluate the ECMTL
    # evaluate_k_g()
    # evaluate_n_og_dg()
    # evaluate_ef_eg()
    #
    # evaluate_etot_emix_ef(type='b2')
    # evaluate_ms_mb(type='b1')
    # evaluate_rmsd(type='b2')

    # plot_figure()
    # for target in ['b0', 'b1', 'b2']:
    #     evaluate_rmsd(type = target)
    #
    # evaluate_as_unit_23()
    # evaluate_as_unit_45()
    # evaluate_as_unit_tot()

    evaluate_transfer_as_unit(type='tl1')
    evaluate_transfer_as_unit(type='tl2')

    # evaluate_transfer_etot_emix_eform()
    # evaluate_transfer_ms_mb()
    # evaluate_transfer_rmsd()

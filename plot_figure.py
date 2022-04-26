import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Colormap
from matplotlib import colors
import pickle
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.font_manager import FontProperties
import pandas as pd

left_limit = 0
right_limit = 300
x_label = 'Calculated $E_tot$'
y_label = 'Predicted $E_tot$'

color = 'blue'
if not os.path.exists('./fig'):
    os.makedirs('./fig')
fig_path = './fig'


def krr_plot(y, y_, fig_name):
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(y, y_, s=2, c=color, alpha=0.8, label='Train')
    equal_x = np.arange(left_limit, right_limit, 0.1)
    plt.xlim(left_limit, right_limit)
    plt.ylim(left_limit, right_limit)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.plot(equal_x, equal_x, color='k', linestyle='dashed', linewidth=1, markersize=1)
    plt.savefig(os.path.join(fig_path, fig_name))


def dnn_plot(y, y_, fig_name):
    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(y, y_, s=2, c=color, alpha=0.8, label='Train')
    equal_x = np.arange(left_limit, right_limit, 0.1)
    plt.xlim(left_limit, right_limit)
    plt.ylim(left_limit, right_limit)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.plot(equal_x, equal_x, color='k', linestyle='dashed', linewidth=1, markersize=1)
    plt.savefig(os.path.join(fig_path, fig_name))

task_dict = {'etot': 'Etot (eV/atom)',
             'etot_all': 'Etot (eV)',
             'emix': 'Emix (eV/atom)',
             'eform': 'Eform (eV/atom)',
             'ms': 'Ms ($\mu_b$/atom)',
             'ms_all': 'Ms (emu/g)',
             'mb': 'mb ($\mu_b$/cell)',
             'rmsd': r'rmsd ($\AA$)',
             }

def scatter_hist(x, y, x_train=None, y_train=None, r2_s=None, mae_s=None, task=None, **kwargs ):
    # definitions for the axes
    left, width = 0.1, 0.72
    bottom, height = 0.1, 0.72
    spacing = 0.005
    property = '$E_{mix}$ (eV/atom)'
    if task:
        if task in task_dict.keys():
            property = task_dict[task]
        else:
            pass
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.13]
    rect_histy = [left + width + spacing, bottom, 0.13, height]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8), dpi=400)
    
    # Etot_all 200
    # Etot :1
    base=30
    if task=='mb':
        base=20
    xmajorLocator = MultipleLocator(base=base)

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.set_xlabel('Calculated {}'.format(property), fontdict={'family': 'Times New Roman', 'size': 20})
    ax_scatter.set_ylabel('Predicted {}'.format(property), fontdict={'family': 'Times New Roman', 'size': 20})
    ax_scatter.tick_params(direction='in', top=True, right=True, labelsize=18, size=5)
    ax_scatter.xaxis.set_major_locator(xmajorLocator)
    ax_scatter.yaxis.set_major_locator(xmajorLocator)
    labels = ax_scatter.get_xticklabels() + ax_scatter.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    if x_train and y_train:
        if x_train.any() and y_train.any():
            # #b2b2b6(灰色), #90a0c7, #A5C8E1
            s1 = ax_scatter.scatter(x_train, y_train, edgecolors='#A5C8E1', linewidths=0.1, color='#A5C8E1')
    else:
        s1=''
    # the scatter plot:
    s2 = ax_scatter.scatter(x, y, edgecolors='darkorange', linewidths=0.1, color='darkorange', alpha=0.95)
    if x_train and y_train:
        ax_scatter.legend([s1, s2], ('Train', 'Test'), loc="upper left", fontsize=14, frameon=False)
    # sandybrown
    # now determine nice limits by hand:
    binwidth = 0.25
    if task=='emix' or 'eform':
        binwidth = 0.1
    l_lim = np.floor(np.array([x, y]).min() / binwidth) * binwidth - 2 * binwidth
    r_lim = np.ceil(np.array([x, y]).max() / binwidth) * binwidth + 2 * binwidth
    ax_scatter.set_xlim((l_lim, r_lim))
    ax_scatter.set_ylim((l_lim, r_lim))
    ax_scatter.grid(linestyle='-.', alpha=.45)
    ax_scatter.set_axisbelow(True)
    # plot the ypred=ytest dashed line.
    # equal_x = np.arange(l_lim, r_lim, 0.1)
    equal_x = np.linspace(l_lim, r_lim, 50, endpoint=True)
    ax_scatter.plot(equal_x, equal_x, color='k', linestyle='dashed', linewidth=1, markersize=1)
    # ax_scatter.grid(linestyle='--')

    #
    bins = 30
    if task=='mb':
        bins = np.arange(l_lim, r_lim + binwidth, binwidth)
    sns.distplot(x, bins=bins, hist=True, kde=True, ax=ax_histx)
    if x_train:
        if x_train.any():
            sns.distplot(x_train, bins=bins, hist=True, kde=True, ax=ax_histx)
    # ax_histx.hist(x, bins=bins, histtype='stepfilled',color='#b2b2b6',alpha=0.7)
    ax_histx.axis('off')
    sns.distplot(y, bins=bins, hist=True, kde=True, ax=ax_histy, vertical=True)
    if y_train:
        if y_train.any():
            sns.distplot(y_train, bins=bins, hist=True, kde=True, ax=ax_histy, vertical=True)
    # ax_histy.hist(y, bins=bins, orientation='horizontal',alpha=0.7,color='#b2b2b6')
    ax_histy.axis('off')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    if r2_s and mae_s:
        plt.text(x=0.2, y=0.8, s='$R^2$={:.5f} \nMAE={:.5f}'.format(r2_s, mae_s), fontdict={'family': 'Times New Roman', 'size': 20}
             , transform=ax_scatter.transAxes)

    save_fig = True
    if save_fig:
        plt.savefig(os.path.join(fig_path, 'fig1.png'), format='png', bbox_inches='tight')
    plt.show()


def plot_nspecies(save=False):
    # count the number of materials of singular/binary/ternary
    n_species = [98, 2481, 2907, 0]
    N = len(n_species)
    index = np.arange(1, N + 1)
    fig, ax = plt.subplots()
    ax.bar(index, height=n_species, color='blue')
    ax.set_xticks(index)
    ax.set_xticklabels(['singular', 'binary', 'ternary', '$\geq4$'])
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.xticks(fontproperties='Times New Roman', size=13)
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.ylabel('Count', fontdict={'family': 'Times New Roman', 'size': 13})
    # plt.xlabel('Lattice System',fontdict={'family':'Times New Roman','size':12})
    for a, b in zip(index, n_species):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=9, fontproperties="Times New Roman")
    if save:
        plt.savefig('./fig/nspecies_stastics.png', dpi=600, bbox_inches='tight')
    plt.show()


def scatter_hist(x, y, x_train=None, y_train=None, r2_s=None, mae_s=None, task=None, fig_path='./', **kwargs):
    # definitions for the axes
    left, width = 0.1, 0.72
    bottom, height = 0.1, 0.72
    spacing = 0.005
    xy_label_fontsize = 25
    legend_font_size = 20
    note_size = 20
    note_x, note_y = 0.2, 0.7
    property = '$E_{mix}$ (eV/atom)'
    if task:
        if task in task_dict.keys():
            property = task_dict[task]
        else:
            pass
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.13]
    rect_histy = [left + width + spacing, bottom, 0.13, height]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8), dpi=400)

    # Etot_all 200
    # Etot :1
    base = 30
    base = num_base.get(task)
    xmajorLocator = MaxNLocator(nbins=5)
    #     xmajorLocator = AutoLocator(5)
    #     xmajorLocator = MultipleLocator(base=base)

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.set_xlabel('Calculated {}'.format(property),
                          fontdict={'family': 'Times New Roman', 'size': xy_label_fontsize})
    ax_scatter.set_ylabel('Predicted {}'.format(property),
                          fontdict={'family': 'Times New Roman', 'size': xy_label_fontsize})
    ax_scatter.tick_params(direction='in', top=True, right=True, labelsize=18, size=5)
    ax_scatter.xaxis.set_major_locator(xmajorLocator)
    ax_scatter.yaxis.set_major_locator(xmajorLocator)
    labels = ax_scatter.get_xticklabels() + ax_scatter.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # #b2b2b6(灰色), #90a0c7, #A5C8E1
    s1 = ax_scatter.scatter(x_train, y_train, edgecolors='#A5C8E1', linewidths=0.1, color='#A5C8E1')

    # the scatter plot:
    s2 = ax_scatter.scatter(x, y, edgecolors='darkorange', linewidths=0.1, color='darkorange', alpha=0.95)

    ax_scatter.legend([s1, s2], ('Train', 'Test'), loc="upper left", fontsize=legend_font_size, frameon=False)
    # sandybrown
    # now determine nice limits by hand:
    #     binwidth = 0.25
    #     if task=='emix' or 'eform':
    #         binwidth = 0.1
    binwidth = binwidth_dict.get(task)
    l_lim = np.floor(np.array([x, y]).min() / binwidth) * binwidth - 2 * binwidth
    r_lim = np.ceil(np.array([x, y]).max() / binwidth) * binwidth + 2 * binwidth
    ax_scatter.set_xlim((l_lim, r_lim))
    ax_scatter.set_ylim((l_lim, r_lim))
    ax_scatter.grid(linestyle='-.', alpha=.45)
    ax_scatter.set_axisbelow(True)
    # plot the ypred=ytest dashed line.
    # equal_x = np.arange(l_lim, r_lim, 0.1)
    equal_x = np.linspace(l_lim, r_lim, 50, endpoint=True)
    ax_scatter.plot(equal_x, equal_x, color='k', linestyle='dashed', linewidth=1, markersize=1)
    # ax_scatter.grid(linestyle='--')

    #
    bins = 30
    if task == 'mb':
        bins = np.arange(l_lim, r_lim + binwidth, binwidth)

    ## kdeplot at the upper and right side of the main figure.
    sns.kdeplot(data=x,
                fill=True, common_norm=False,
                alpha=.5, linewidth=0, ax=ax_histx
                )
    #     sns.distplot(x, bins=bins, hist=False, kde=True, ax=ax_histx)
    sns.kdeplot(data=x_train,
                fill=True, common_norm=False,
                alpha=.5, linewidth=0, ax=ax_histx
                )
    #     sns.distplot(x_train, bins=bins, hist=False, kde=True, ax=ax_histx)
    # ax_histx.hist(x, bins=bins, histtype='stepfilled',color='#b2b2b6',alpha=0.7)
    ax_histx.axis('off')

    sns.kdeplot(data=y,
                fill=True, common_norm=False,
                alpha=.5, linewidth=0, ax=ax_histy
                )
    sns.kdeplot(data=y_train,
                fill=True, common_norm=False,
                alpha=.5, linewidth=0, ax=ax_histy
                )
    #     sns.distplot(y, bins=bins, hist=True, kde=True, ax=ax_histy, vertical=True)
    #     sns.distplot(y_train, bins=bins, hist=True, kde=True, ax=ax_histy, vertical=True)
    # ax_histy.hist(y, bins=bins, orientation='horizontal',alpha=0.7,color='#b2b2b6')
    ax_histy.axis('off')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    #     if r2_s and mae_s:
    #         plt.text(x=note_x, y=note_y, s='$R^2$={:.3f} \nMAE={:.3f}'.format(r2_s, mae_s), fontdict={'family': 'Times New Roman', 'size': note_size}
    #              , transform=ax_scatter.transAxes)

    save_fig = True
    if save_fig:
        plt.savefig(os.path.join(fig_path, 'fig1.png'), format='png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    pass

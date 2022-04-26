from datasets.preprocessing import ReadExcel

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axisartist.axislines import Axes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams.update({"font.size":13})#此处必须添加此句代码方可改变标题字体大小


def read_excel(in_file):
    """
    Read the excel file using the Pandas.
    :param in_file: str, the path name of the input file, such as '../HEA_Data/Quinary.xlsx'
    :return: ndarray.
    """
    wb = pd.read_excel(io=in_file, sheet_name=0, engine='openpyxl', index_col=0)
    array = wb.to_numpy()
    return array


file_name = '../HEA_Data/Out_labels/Database.xlsx'
data = read_excel(file_name)
labels = {'etot': 1,
          'emix': 3,
          'ef': 4,
          'ms': 6,
          'mb': 7,
          'rmsd': 8}

label_name = {'etot': 'E$_{tot}$ [eV/atom]',
              'emix': 'E$_{mix}$ [eV/atom]',
              'ef': 'E$_{form}$ [eV/atom]',
              'ms': 'm$_s$ [$\mu_b$/atom] ',
              'mb': 'm$_b$ [$\mu_b$/cell]',
              'rmsd': 'RMSD [$\AA$]'

              }
print(data.shape)


def plot_cor1():
    fig = plt.figure(figsize=(15, 8))
    i = 1
    for a, b in zip(['ms', 'ms', 'ef', 'ef', 'rmsd', 'rmsd', 'rmsd', 'etot'],
                    ['ef', 'mb', 'etot', 'emix', 'ef', 'ms', 'etot', 'emix']):
        # Compute correlation
        data_a = [float(i) for i in data[:, labels[a]]]
        data_b = [float(i) for i in data[:, labels[b]]]
        rho = np.corrcoef(data_a, data_b)[0, 1]

        col = i % 4
        row = i // 4
        # Plot column wise. Positive correlation in row 0 and negative in row 1
        colors = 'green' if row == 1 else 'magenta'
        ax = fig.add_subplot(2, 4, i, axes_class=Axes)
        ax.scatter(data_a, data_b, color=colors)
        ax.title.set_text('Correlation = ' + "{:.2f}".format(rho))
        # ax[row, col].scatter(data_a, data_b, color=colors)
        # ax[row, col].title.set_text('Correlation = ' + "{:.2f}".format(rho))

        # ax[row, col].set(xlabel=label_name[a], ylabel=label_name[b])
        # ax[row, col].tick_params(labelsize=13)
        # ax[row, col].set_xlabel(xlabel=label_name[a], fontsize=13)
        # ax[row, col].set_ylabel(ylabel=label_name[b], fontsize=13)
        fs = 18
        ax.set_xlabel(xlabel=label_name[a], fontsize=fs)
        ax.set_ylabel(ylabel=label_name[b], fontsize=fs)

        ax.axis['right'].set_visible(False)
        ax.axis['top'].set_visible(False)
        # ax[row, col].set_axis_off()
        # ax[row, col].tick_params(axis='both', labelsize=16)
        i += 1

    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.tight_layout()
    plt.savefig('../fig/correlation.png', dpi=500)
    plt.show()

def plot_cor2():
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))
    i = 0
    for a, b in zip(['ms', 'ms', 'ef', 'ef', 'rmsd', 'rmsd', 'rmsd', 'etot'],
                    ['ef', 'mb', 'etot', 'emix', 'ef', 'ms', 'etot', 'emix']):
        # Compute correlation
        data_a = [float(i) for i in data[:, labels[a]]]
        data_b = [float(i) for i in data[:, labels[b]]]
        rho = np.corrcoef(data_a, data_b)[0, 1]

        col = i % 4
        row = i // 4
        # Plot column wise. Positive correlation in row 0 and negative in row 1
        colors = 'green' if row == 1 else 'magenta'
        ax[row, col].scatter(data_a, data_b, color=colors)
        ax[row, col].title.set_text('Correlation = ' + "{:.2f}".format(rho))

        # ax[row, col].set(xlabel=label_name[a], ylabel=label_name[b])
        ax[row, col].tick_params(labelsize=13)
        fs = 16
        ax[row, col].set_xlabel(xlabel=label_name[a], fontsize=fs)
        ax[row, col].set_ylabel(ylabel=label_name[b], fontsize=fs)
        #
        # ax.axis['right'].set_visible(False)
        # ax.axis['top'].set_visible(False)
        # ax[row, col].set_axis_off()
        # ax[row, col].tick_params(axis='both', labelsize=16)
        i += 1

    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.tight_layout()
    plt.savefig('../fig/correlation.png', dpi=500)
    plt.show()

plot_cor2()
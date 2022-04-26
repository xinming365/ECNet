import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

exp_dir = '../saved_models_MLP'
exp_ib1 = os.path.join(exp_dir, 'exp-ib1-fcc.csv')
exp_ib3 = os.path.join(exp_dir, 'exp-ib3-fcc.csv')

exp_ib1_bcc = os.path.join(exp_dir, 'exp-ib1-bcc.csv')
exp_ib3_bcc = os.path.join(exp_dir, 'exp-ib3-bcc.csv')

def merge_exp_csv(file_list, features=['ib1', 'ib3']):
    # The following settings are based on your dataset.
    rows = 5
    fractions = [0.2, 0.4, 0.6, 0.8, 1]
    maes = []
    feature = []
    data_size = []
    for i, f in enumerate(file_list):
        df_i = pd.read_csv(f)
        feature_i = i
        for idx in range(rows):
            a = df_i.iloc[idx]
            ncols = len(a)
            maes.extend(a)
            feature.extend([features[feature_i] for _ in range(ncols)])
            data_size.extend([fractions[idx] for _ in range(ncols)])

    merged_data = pd.DataFrame({'data_size': data_size,
                                'mae': maes,
                                'feature': feature})

    return merged_data



# df = merge_exp_csv([exp_ib1, exp_ib3], features=['ib1', 'ib3'])
df = merge_exp_csv([exp_ib1_bcc, exp_ib3_bcc], features=['ib1', 'ib3'])
# for idx, value in df.iterrows():
#     print(idx, value['exp_1'])
print(df)
sns.set_theme(style='darkgrid')
# fmri = sns.load_dataset(name='fmri')

sns.lineplot(x='data_size', y='mae', hue ='feature', data=df)
plt.show()

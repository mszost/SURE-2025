import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    # convert excel to csv
    # df = pd.read_excel('../data/xlsx/RPLControl Th17 Data Original.xlsx')
    # df.to_csv('../data/csv/RPLControl_Th17_Data.csv', index=False)

    df = pd.read_csv('../data/csv/RPLControl_Th17_Data.csv')

    new_column_names = {
        'MRN': 'mrn',
        'GA': 'ga',
        'RIF': 'rif',
        'RPL': 'rpl',
        'RPL.1': 'rpl_1',
        'RIF.1': 'rif_1',
        'RIF =1, RPL =2, Both = 3': 'target',
        'Endometriosis (0=N, 1=Y)': 'endometriosis',
        'Adenomyosis (0=N, 1=Y)': 'adenomyosis',
        'PCOS (0=N, 1=Y)': 'pcos',
        'Fibroids (0=N, 1=Y)': 'fibroids',
        'Th17 (CD4+)': 'th17_cd4',
        'Th17 (CD4+); IL17+/IFN+': 'th17_cd4_il17_ifn_pos',
        'Th17 (CD4+); IL17+/IFN-': 'th17_cd4_il17_ifn_neg',
        'Treg (CD25+CD127-)': 'treg_cd25_cd127',
        'DoublePos/Neg ratio': 'double_pos_neg_ratio',
        'Th17TregRatio': 'th17_treg_ratio'
    }
    df.rename(columns=new_column_names, inplace=True)

    # remove redundant columns 
    df = df.drop(['mrn', 'rif', 'rpl', 'rpl_1', 'rif_1'], axis=1)

    # remove features with low correlation to the target
    #df.drop(['th17_cd4_il17_ifn_pos', 'th17_cd4_il17_ifn_neg', 'double_pos_neg_ratio'], axis=1)

    # for rows where the target is empty, set it 0
    df.fillna({'target': 0}, inplace=True)
    # filter out rows for which the target is empty or 0
    df = df[df['target'] != 0.0]

    # fill empty with 0 for endometriosis, adenomyosis, pcos, and fibroids
    df.fillna({'endometriosis': 0}, inplace=True)
    df.fillna({'adenomyosis': 0}, inplace=True)
    df.fillna({'pcos': 0}, inplace=True)
    df.fillna({'fibroids': 0}, inplace=True)

    # clean gestational age column ("new" becomes 0)
    df['ga'] = pd.to_numeric(df['ga'], errors='coerce')
    df.fillna({'ga': 0}, inplace=True)

    # clean th17_cd4 column (values <1 become 0.1)
    is_lt_one = df['th17_cd4'] == '<1'
    df.loc[is_lt_one, 'th17_cd4'] = 0.1
    df['th17_cd4'] = pd.to_numeric(df['th17_cd4'])

    # interpolate missing values
    cols_to_interp = [ 'th17_cd4', 'treg_cd25_cd127', 'th17_cd4_il17_ifn_pos', 'th17_cd4_il17_ifn_neg']
    df[cols_to_interp] = df[cols_to_interp].interpolate(method='linear', limit_direction='both')

    # fill any left over empty values with their respective medians
    df[cols_to_interp] = df[cols_to_interp].fillna(df[cols_to_interp].median())

    # manually calculate ratios
    df['th17_treg_ratio'] = df['th17_cd4'] / df['treg_cd25_cd127']
    df['double_pos_neg_ratio'] = df['th17_cd4_il17_ifn_pos'] / df['th17_cd4_il17_ifn_neg']

    # normalization
    cols_to_normalize = ['ga', 'th17_cd4', 'treg_cd25_cd127','th17_cd4_il17_ifn_pos', \
        'th17_cd4_il17_ifn_neg', 'double_pos_neg_ratio', 'th17_treg_ratio']
    df[cols_to_normalize] = MinMaxScaler().fit_transform(df[cols_to_normalize])

    # export 
    df.to_csv('../data/processed/4th_jun18_sklearn_normalized_RPLControl.csv', index=False)
    
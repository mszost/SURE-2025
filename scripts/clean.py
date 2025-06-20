import pandas as pd
import numpy as np

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

    # filter out all rows of the target column which are 0 or empty
    df.fillna({'target': 0}, inplace=True)
    df = df[df['target'] != 0.0]

    # clean gestational age column ("new" becomes 0)
    df['ga'] = pd.to_numeric(df['ga'], errors='coerce')
    df.fillna({'ga': 0}, inplace=True)

    # clean th17_cd4 column (values <1 become 0.1)
    is_lt_one = df['th17_cd4'] == '<1'
    df.loc[is_lt_one, 'th17_cd4'] = 0.1
    df['th17_cd4'] = pd.to_numeric(df['th17_cd4'])

    # fill empty with 0 for endometriosis, adenomyosis, pcos, and fibroids
    df.fillna({'endometriosis': 0}, inplace=True)
    df.fillna({'adenomyosis': 0}, inplace=True)
    df.fillna({'pcos': 0}, inplace=True)
    df.fillna({'fibroids': 0}, inplace=True)

    # interpolate missing values
    cols_to_interp = ['th17_cd4', 'treg_cd25_cd127', 'th17_cd4_il17_ifn_pos', 'th17_cd4_il17_ifn_neg']
    df[cols_to_interp] = df[cols_to_interp].interpolate(method='linear', limit_direction='both')
    df[cols_to_interp] = df[cols_to_interp].fillna(df[cols_to_interp].median()) # catch any left over NaNs

    # calculate ratios 
    df['th17_treg_ratio'] = df['th17_cd4'] / df['treg_cd25_cd127'].replace(0, np.nan)
    df['double_pos_neg_ratio'] = df['th17_cd4_il17_ifn_pos'] / df['th17_cd4_il17_ifn_neg'].replace(0, np.nan)
    # handle errors caused by division by zero
    df['th17_treg_ratio'].fillna(0.05, inplace=True)  # df['th17_treg_ratio'].median()
    df['double_pos_neg_ratio'].fillna(0.05, inplace=True)  # df['double_pos_neg_ratio'].median()

    # remove features with low correlation to the target
    # df.drop(['th17_cd4_il17_ifn_pos', 'th17_cd4_il17_ifn_neg', 'double_pos_neg_ratio'], axis=1)

    # export 
    df.to_csv('../data/processed/clean_RPLControl_Th17.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

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
        'Th17 (CD4+)': 'th17',
        'Th17 (CD4+); IL17+/IFN+': 'ifn_pos',
        'Th17 (CD4+); IL17+/IFN-': 'ifn_neg',
        'Treg (CD25+CD127-)': 'treg',
        'DoublePos/Neg ratio': 'pos_neg_ratio',
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

    # clean th17 column (values <1 become 0.1)
    df['th17'] = df['th17'].replace('<1', 0.1)
    df['th17'] = pd.to_numeric(df['th17'], errors='coerce')

    # fill empty with 0 for endometriosis, adenomyosis, pcos, and fibroids
    df.fillna({'endometriosis': 0}, inplace=True)
    df.fillna({'adenomyosis': 0}, inplace=True)
    df.fillna({'pcos': 0}, inplace=True)
    df.fillna({'fibroids': 0}, inplace=True)

    # interpolate missing values
    cols_to_interp = ['th17', 'treg', 'ifn_pos', 'ifn_neg']
    df[cols_to_interp] = df[cols_to_interp].interpolate(method='linear', limit_direction='both')

    # KNN imputation
    #imputer = KNNImputer(n_neighbors=5)
    #df[cols_to_interp] = imputer.fit_transform(df[cols_to_interp])

    # outlier handling
    Q1 = df['th17'].quantile(0.25)
    Q3 = df['th17'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['th17'] < lower_bound) | (df['th17'] > upper_bound)]
    print(f"Found {len(outliers)} outliers in the 'th17' column.")
    df['th17'] = np.clip(df['th17'], lower_bound, upper_bound)

    # # calculate ratios 
    df['th17_treg_ratio'] = df['th17'] / df['treg'].replace(0, np.nan)
    df['pos_neg_ratio'] = df['ifn_pos'] / df['ifn_neg'].replace(0, np.nan)

    # export 
    df.to_csv('../data/processed/fa_RPLControl.csv', index=False)

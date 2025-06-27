import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer.rotator import Rotator

from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours, NearMiss, InstanceHardnessThreshold
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    ULINE = '\033[4m'


def factor_analysis(X_train, X_test):
    feats_for_fa = ['ga', 'th17', 'ifn_pos', 'pos_neg_ratio', 'th17_treg_ratio'] #'ifn_neg', 'treg']

    n_factors=3
    fa = FactorAnalyzer(n_factors=n_factors, rotation='oblimin', method='minres', use_smc=True)
    fa.fit(X_train[feats_for_fa])

    #kmo_all, kmo_per_variable = calculate_kmo(X_train[feats_for_fa])
    #bartlett_chi_square, bartlett_p_value = calculate_bartlett_sphericity(X_train[feats_for_fa])
    #print('KMO', kmo_all)

    #scree plot
    # ev, v = fa.get_eigenvalues()
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(ev) + 1), ev, marker='o')
    # plt.title('Scree Plot')
    # plt.xlabel('Number of Factors')
    # plt.ylabel('Eigenvalue')
    # plt.grid(True)
    # plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (Eigenvalue > 1)')
    # plt.legend()
    # plt.xticks(range(1, len(ev) + 1))
    # plt.show()

    # get factor loadings
    loadings = pd.DataFrame(fa.loadings_, index=X_train[feats_for_fa].columns,
                      columns=[f'Factor {i+1}' for i in range(n_factors)])
    print("\nFactor loadings:\n", loadings.round(3))

    # get factor scores
    fa_transform = lambda data : pd.DataFrame(fa.transform(data[feats_for_fa]),
                                            columns=[f'factor_{i+1}' for i in range(n_factors)],
                                            index=data.index)

    X_train_fa_scores = fa_transform(X_train)
    X_test_fa_scores = fa_transform(X_test)

    print(X_train_fa_scores)

    # remove source features
    X_train = X_train.drop(columns=feats_for_fa)
    X_test = X_test.drop(columns=feats_for_fa)

    # combine factor scores back into the data as new features
    X_train = pd.concat([X_train, X_train_fa_scores], axis=1)
    X_test = pd.concat([X_test, X_test_fa_scores], axis=1)

    #print(f'X_train: \n{X_train}')
    #print(f'X_test: \n{X_test}')
    return X_train, X_test


def grid_search(model_name, model_instance, X_train, X_test, y_train):
    print(f'\n{C.BOLD+C.BLUE}[{model_name}]{C.END} Starting GridSearchCV ...')

    match model_name:
        case 'Random Forest':
            param_grid = {
               'max_features': ['sqrt', 'log2', 0.8, 1, None],
               'max_depth': [30, None],
               'min_samples_split': [2, 5],
               'min_samples_leaf': [1, 2],
               'criterion': ['gini', 'entropy'],
               'class_weight': ['balanced', None]
            }
        case 'Gradient Boost':
            param_grid = {
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 7],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }
        case 'SVC':
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'degree': [2, 3, 4],
                'shrinking': [True, False]
            }
        case 'LinearSVC':
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'loss': ['squared_hinge'],
                'dual': [True, False],
                'max_iter': [1000, 5000, 10000]
            }
        case 'MLPClassifier':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 50, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': [0.001, 0.01],
                'max_iter': [200, 500]
            }

    gs = GridSearchCV(model_instance, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print(f"\nBest parameters found for {model_name}: {gs.best_params_}")
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    return (best_model, y_pred)


def main(datasets, models, iterations=1, gscv=False, clf_report=False):
    performance_summary = {}
    for csv_path in datasets:
        print(f'Using data file: {csv_path}')
        performance_summary[csv_path] = {}

        # load data
        df = pd.read_csv(csv_path)
        X = df.drop('target', axis=1)
        y = df['target']

        # plot correlation heatmap
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        # plt.title('Correlation Heatmap')
        # plt.show()

        # evaluate models
        for name, model_instance in list(models.items()):
            accuracies = []

            for i in range(iterations):
                if clf_report: 
                    print(f'\n{'_'*60}')

                print(f'\r{C.BOLD+C.BLUE}[{name}]{C.END} Running model evaluation ({C.BOLD+C.GREEN}{i+1}/{iterations}{C.END}) ', end='', flush=True)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

                # apply normalization
                scaler = StandardScaler()
                cols_to_normalize = ['ga', 'th17', 'ifn_pos', 'ifn_neg', 'treg', 'pos_neg_ratio', 'th17_treg_ratio']
                X_train[cols_to_normalize] = scaler.fit_transform(X_train[cols_to_normalize])
                X_test[cols_to_normalize] = scaler.transform(X_test[cols_to_normalize])

                #factor_analysis(X_train, X_test)

                # apply oversampling and undersampling
                X_train, y_train  = SMOTENC(random_state=0, k_neighbors=5, categorical_features=['endometriosis', 'adenomyosis', 'pcos', 'fibroids']).fit_resample(X_train, y_train)
                X_train, y_train = EditedNearestNeighbours(sampling_strategy='majority', kind_sel='mode').fit_resample(X_train, y_train)

                # apply grid search
                if gscv:
                    model_instance, y_pred = grid_search(name, model_instance, X_train, X_test, y_train)
                else:
                    model_instance.fit(X_train, y_train)
                    y_pred = model_instance.predict(X_test)

                if clf_report:
                    print(f'\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['1 (RIF)', '2 (RPL)', '3 (Both)'])}')
                    print(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}')

                accuracy = accuracy_score(y_test, y_pred)
                if accuracy is None:
                    break
                accuracies.append(accuracy)

            if accuracies:
                average_accuracy = np.mean(accuracies)
                performance_summary[csv_path][name] = average_accuracy
                print(f'\n{C.BOLD+C.BLUE}[{name}]{C.END} Average accuracy: {C.BOLD+C.ULINE+C.RED}{(average_accuracy * 100):.2f}%{C.END}\n')

            models.pop(name)

            if clf_report and len(models) > 0:
                print(f'\n{'#'*90}\n')


if __name__ == '__main__':
    # df = pd.read_excel('../data/processed/fa3_RPLControl.xlsx')
    # df.to_csv('../data/processed/fa3_RPLControl.csv', index=False)

    datasets = ['../data/processed/fa_RPLControl.csv']
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=1000, criterion='entropy', max_depth=30, max_features='sqrt',
            min_samples_leaf=1, min_samples_split=2, class_weight=None),

        # 'Gradient Boost': GradientBoostingClassifier(
        #     n_estimators=800, subsample=0.7, max_depth=4, max_features='sqrt',
        #     min_samples_leaf=2, min_samples_split=5, learning_rate=0.2),

        # 'MLPClassifier': MLPClassifier(
        #    activation='tanh', alpha=0.001, hidden_layer_sizes=(100,50), learning_rate='constant',
        #    learning_rate_init=0.01, max_iter=500, solver='adam'),

        # 'KNeighbors': KNeighborsClassifier(,
        # 'SVC': SVC(C=100, degree=2, gamma=1, kernel='rbf', shrinking=True),
        # 'LinearSVC': LinearSVC(C=10, dual=True, loss='squared_hinge', max_iter=5000, penalty='l2'),
    }

    main(datasets, models, iterations=1, gscv=False, clf_report=True)
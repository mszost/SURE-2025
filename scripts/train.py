import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

from imblearn.over_sampling import SMOTE
from collections import Counter

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
    feats_for_fa = ['ga', 'th17_cd4', 'th17_cd4_il17_ifn_pos', 'th17_cd4_il17_ifn_neg', 'treg_cd25_cd127',
        'double_pos_neg_ratio', 'th17_treg_ratio']

    n_factors=4
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='principal')
    fa.fit(X_train[feats_for_fa])

    # get FA scores for both training and test sets
    fa_transform = lambda data : pd.DataFrame(fa.transform(data[feats_for_fa]),
                                            columns=[f'factor_score_{i+1}' for i in range(n_factors)],
                                            index=data.index)

    X_train_fa_scores = fa_transform(X_train)
    X_test_fa_scores = fa_transform(X_test)

    # remove orignal (factorized) features 
    X_train = X_train.drop(columns=feats_for_fa)
    X_test = X_test.drop(columns=feats_for_fa)

    # combine factor scores back into the data as new features
    X_train = pd.concat([X_train, X_train_fa_scores], axis=1)
    X_test = pd.concat([X_test, X_test_fa_scores], axis=1)

    #print(X_train)
    return X_train, X_test


def gridsearch(model_name, model_instance, X_train, X_test, y_train):
    print(f'\n{C.BOLD+C.BLUE}[{model_name}]{C.END} Starting GridSearchCV ...')

    match model_name:
        case 'Random Forest':
            param_grid = {
               'n_estimators': [100],
               'max_features': ['sqrt', 'log2', 0.8, 1, None],
               'max_depth': [30, None],
               'min_samples_split': [2, 5],
               'min_samples_leaf': [1, 2],
               'criterion': ['gini', 'entropy'],
               'class_weight': ['balanced', None]
            }
        case 'Gradient Boost':
            param_grid = {
                'n_estimators': [100],
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

        for name, model_instance in list(models.items()):
            accuracies = []
            for i in range(iterations):
                if clf_report: 
                    print(f'\n{'_'*60}')
                print(f'\r{C.BOLD+C.BLUE}[{name}]{C.END} Running model evaluation ({C.BOLD+C.GREEN}{i+1}/{iterations}{C.END}) ', end='', flush=True)

                # evaluate model
                df = pd.read_csv(csv_path)
                X = df.drop('target', axis=1)
                y = df['target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)  # stratify=y to ensure train/test sets have similar class proportions to the original data

                #apply normalization
                scaler = MinMaxScaler()
                cols_to_normalize = ['ga', 'th17_cd4', 'treg_cd25_cd127', 'th17_cd4_il17_ifn_pos', 'th17_cd4_il17_ifn_neg', 'double_pos_neg_ratio', 'th17_treg_ratio',]
                X_train[cols_to_normalize] = scaler.fit_transform(X_train[cols_to_normalize])
                X_test[cols_to_normalize] = scaler.transform(X_test[cols_to_normalize])

                # apply factor analysis
                X_train, X_test = factor_analysis(X_train, X_test)

                # apply SMOTE
                X_train, y_train = SMOTE(sampling_strategy='minority', k_neighbors=5).fit_resample(X_train, y_train)

                # apply GridSearchCV
                if gscv:
                    model_instance, y_pred = gridsearch(name, model_instance, X_train, X_test, y_train)
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
        # 'Random Forest': RandomForestClassifier(
        #     n_estimators=800, criterion='entropy', max_depth=30, max_features='sqrt',
        #     min_samples_leaf=1, min_samples_split=2, class_weight='balanced'),

        'Gradient Boost': GradientBoostingClassifier(
                    n_estimators=800, subsample=0.7, max_depth=4, max_features='sqrt',
                    min_samples_leaf=1, min_samples_split=5, learning_rate=0.1),

        #'MLPClassifier': MLPClassifier(
            #activation='relu', alpha=0.0001, hidden_layer_sizes=(100,50), learning_rate='constant',
            #learning_rate_init=0.01, max_iter=500, solver='adam'),

        #'KNeighbors': KNeighborsClassifier(),
        #'SVC': SVC(C=100, degree=2, gamma=1, kernel='rbf', shrinking=True),
        #'LinearSVC': LinearSVC(C=10, dual=True, loss='squared_hinge', max_iter=5000, penalty='l2'),
    }

    main(datasets, models, iterations=4, gscv=True, clf_report=False)

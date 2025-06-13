import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
#import seaborn as sns


df = pd.read_csv('../data/processed/WIP2_RPLControl_Th17.csv')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    #'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    #'K-Nearest Neighbors': KNeighborsClassifier(),
    #'Naive Bayes': GaussianNB()
}

for i in models.keys():
    models[i].fit(X_train, y_train)
    y_pred = models[i].predict(X_test)

    #print(f'\nModel Accuracy: {(accuracy_score(y_test, y_pred) * 100):.2f}%')
    print(f'{i}\nClassification report:\n {classification_report(y_test, y_pred)}')
    print(f'Confusion matrix:\n {confusion_matrix(y_test, y_pred)}\n\n{"-"*60}')

    # plot confusion matrix
    #plt.figure(figsize=(8, 6))
    #sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
    #plt.xlabel('Predicted')
    #plt.ylabel('Actual')
    #plt.title('Confusion Matrix')
    #plt.show()

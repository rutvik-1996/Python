# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
plt.style.use('ggplot')
!pip install plotly
mall = pd.read_excel("/Users/suryateja/Documents/Data Management and Big data/ospi.xlsx")
mall
mall.info()
mall.describe()
mall.isna().sum()
mall.shape
mall.columns

plt.figure(figsize=(15,15))
plt.subplot(221)
plt.ylabel('Count')
sns.distplot(mall.ProductRelated_Duration,color='blue')
plt.subplot(222)
sns.distplot(mall.BounceRates,color='Red')
plt.ylabel('Count');
plt.figure(figsize=(15,5))
plt.subplot(121)
sns.distplot(mall.ExitRates,color='Green')
plt.figure(figsize=(30,30))
sns.heatmap(mall.corr(),annot=True,fmt='g',cbar=True,cmap='viridis');
plt.title("Heat map for Mall Dataset")

plt.rcParams['figure.figsize']=(30,15)
sns.violinplot(mall['OperatingSystems'],mall['Revenue'],palette='rainbow')
plt.rcParams['figure.figsize']=(30,15)
sns.violinplot(mall['TrafficType'],mall['Revenue'],palette='rainbow')

ind_vars = ['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
       'OperatingSystems', 'Browser', 'Region', 'TrafficType',
       'Weekend']
x = mall[ind_vars]
y = mall['Revenue']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(x,y,test_size =0.25,random_state=101)
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,Y_train)
Y_predict = rf.predict(X_test)
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,f1_score
metrics.accuracy_score(Y_test,Y_predict)
from sklearn.model_selection import StratifiedKFold,GridSearchCV

rf_clf = RandomForestClassifier()
parameter_grid ={'n_estimators':[5,10,25,50,100],
                'criterion':['gini','entropy'],
                'max_features':[1,2,3,4],
                'warm_start':[True,False]}#hyper parameters
cross_validation = StratifiedKFold(n_splits=10,shuffle=True,random_state=101)

grid_search = GridSearchCV(rf_clf,param_grid=parameter_grid,cv=cross_validation)

grid_search.fit(X_train.values,Y_train.values)
print('Best score:{}'.format(grid_search.best_score_))
print('Best parameters:{}'.format(grid_search.best_params_))

print(metrics.accuracy_score(Y_test,grid_search.predict(X_test)))
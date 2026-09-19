# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:20:24 2026

@author: 10191
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:56:45 2023

@author: qiubinxu
"""

import streamlit as st
import pandas as pd #处理数据所用库
import numpy as np
import requests #工具，访问服务器
import numpy as np#加载数据所用库
import pandas as pd #处理数据所用库
import xgboost
import xgboost as xgb#极限梯度提升机所用库
from xgboost import XGBClassifier#分类算法#加载极限梯度提升机中分类算法
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree #导入需要的模块
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split#模型选择将数据集分为测试集和训练集
from sklearn.metrics import accuracy_score#模型最终的预测准确度分数
import matplotlib#加载绘图工具
from xgboost import plot_importance#加载极限梯度提升机中重要性排序函数
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score   # 准确率
import scipy.stats as stats
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score as CVS
from xgboost import plot_importance
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from IPython.display import display
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import confusion_matrix
import eli5
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from eli5.sklearn import PermutationImportance
from IPython.display import display, Image
from sklearn.model_selection import cross_val_score,StratifiedKFold
st.title('Application of Machine Learning Methods to Predict Bone Metastases in Bladder Cancer Patients'.center(33, '-'))
classes = {0:'NBM',1:'BM'}
st.sidebar.expander('')     
st.sidebar.subheader('Variable')       
Age=st.sidebar.selectbox('Age', ['<50','≥50'])
Age_map = {'<50':0,'≥50':1}
Radiation=st.sidebar.selectbox("Radiation",['No','Yes'])
Radiation_map={'No':0,'Yes':1}
Surgery=st.sidebar.selectbox("Surgery",['No','Yes'])
Surgery_map={'No':0,'Yes':1}
Tumorsize=st.sidebar.selectbox("Tumorsize",['<5cm',"≥5cm"])
Tumorsize_map={'<5cm':0,'≥5cm':1}
T_stage=st.sidebar.selectbox("T_stage",['T1','T2','T3','T4'])
T_stage_map = {'T1':0,'T2':1,'T3':2,"T4":3} 
N_stage=st.sidebar.selectbox("N_stage",['N0','N1','N2','N3'])
N_stage_map = {'N0':0,'N1':1,'N2':2,"N3":3}
filename = 'modelbladder.txt'
x = []
x.extend([Age_map[Age],Radiation_map[Radiation],Surgery_map[Surgery],
          Tumorsize_map[Tumorsize],T_stage_map[T_stage],N_stage_map[N_stage]])
x=np.array(x).reshape(1,6)
import pickle
if st.button("Predict"):
    #  predict_class()
    import os
    if os.path.exists(filename):
        with open(filename, 'rb') as fq:
            modelXGB = pickle.load(fq, encoding='bytes')
            y_pred = modelXGB.predict_proba(x)
            print(max(y_pred[:,1]))
            st.header('Probability of bone metastases: %.2f %%' % (max(y_pred[:,1])* 100))
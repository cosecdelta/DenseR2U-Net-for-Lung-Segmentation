# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:15:00 2020

@author: Kaushik Dutta
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv ('C:/Users/Kaushik Dutta/Box/codebase/data/T2/With Collowet/Final Features/T2_prediction.csv')
df2 = pd.read_csv('C:/Users/Kaushik Dutta/Box/codebase/data/T2/With Collowet/Final Features/T2_kaushik.csv')
#df3 = pd.read_csv('C:/Users/Kaushik Dutta/Box/codebase/data/T2/With Collowet/Final Features/T2_sudipta.csv')
df3 = pd.read_csv('C:/Users/Kaushik Dutta/Box/codebase/data/T2/With Collowet/Final Features/T2_tim.csv')
df4 = pd.read_csv('C:/Users/Kaushik Dutta/Box/codebase/data/T2/With Collowet/Final Features/T2_xia.csv')
df5 = pd.read_csv('C:/Users/Kaushik Dutta/Box/codebase/data/T2/With Collowet/Final Features/T2_zezhong.csv')

df = [df2,df3,df4,df5]
radiomics_data_algo = df1.to_numpy()

aug_rad = []
for dfs in df:
    
    radiomics_data_expert = dfs.to_numpy()
    radiomics_data = radiomics_data_algo - radiomics_data_expert
    perc_change = (radiomics_data/radiomics_data_expert)*100
    aug_rad.append(perc_change)
    
aug_rad = np.array(aug_rad)   
aug_rad_new = abs(aug_rad) 
avg_apc = np.mean(aug_rad_new, axis = 0)

#plt.imshow(avg_apc.T, cmap='viridis', interpolation='nearest')
#ax = sns.heatmap(avg_apc.T, linewidth=0.5, vmin=0, vmax=100)




# perc_change = avg_apc

r_scores = []
vif_scores = []
coeff = []


## Calculation of Lasso for the difference in feature values to that of difference in Volume

for i in range(1,radiomics_data.shape[1]):
    clf = linear_model.LinearRegression()
    y = np.reshape(perc_change[:,0],(perc_change.shape[0],1))
    x = np.reshape(perc_change[:,i],(perc_change.shape[0],1))
    params = clf.fit(x,y)
    coeff.append(params.coef_)
    r2 = clf.score(x,y)
    vif = 1/(1-r2)
    r_scores.append(r2)
    vif_scores.append(vif)
    
# apc = np.mean(perc_change, axis = 0)
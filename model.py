import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import StandardScaler
import pickle
import scipy.sparse.linalg

df=pd.read_csv("fetal_health.csv")
## conversion of the data type of target variable to int
df["fetal_health"]=df["fetal_health"].astype(int)
sc = StandardScaler()

Feature=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
scaled= sc.fit_transform(Feature)
Scaled=pd.DataFrame(scaled)
Scaled.head(10)

F1=Scaled.iloc[:,[9,7,6,17,8,18,1,3,16,0]]
F1.columns=['percentage_of_time_with_abnormal_long_term_variability','abnormal_short_term_variability','prolongued_decelerations','histogram_mean',
          'mean_value_of_short_term_variability','histogram_median','accelerations','uterine_contractions','histogram_mode','baseline value']
F1.head(10)

Features=F1.copy()

Targets=df.iloc[:,-1]

X_train, X_test, Y_train, Y_test =split(Features, Targets, test_size = 0.4, random_state = 12)

M1=SVC(C=100,gamma=0.1,kernel='rbf',random_state=0)

M1.fit(X_train,Y_train)

pickle.dump(M1,open("model.pkl",mode="wb"))

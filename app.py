import streamlit as sm
import pandas as pd
import numpy as np
import seaborn as sea
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plot3d
import pickle
sm.title("Fetal Health System")
sc=StandardScaler()
model=pickle.load(open("model.pkl",mode="rb"))
@sm.cache()
def fetal_classify(feature):
    Fetal_pred=model.predict(feature)
    return Fetal_pred
def app():
    title= html_temp = """ 
    <div style ="background-color:green;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Fetal Health Detection system-SVM</h1> 
    </div> 
    """
    sm.markdown(title,unsafe_allow_html=True)
    f1= sm.slider("Abnormal long term availability %",0.0,100.0)
    f2=sm.slider("Abnromal short time availability",10.0,90.0)
    f3=sm.slider("Short term availability mean",0.1,8.0)
    f4=sm.slider("Histogram mean",70.0,190.0)
    f5=sm.slider("Histogram median",70.0,190.0)
    f6=sm.slider("Histogram Mode",50.0,190.0)
    f7=sm.slider("Baseline value",100.0,180.0)
    f8=sm.slider("Uterine contractions",0.0,0.02)
    f9=sm.slider("accelerations",0.0,0.025)
    f10=sm.slider("prolonged decelerations",0.0,0.008)
    feature=sc.fit_transform([[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]])
    if sm.button("Classify"):
        Feature=fetal_classify(feature)
        if Feature==1:
            sm.success("The Baby is normal")
        elif Feature==2:
            sm.success("Suspect")
        else:
            sm.success("Pathological")

if __name__ == '__main__':
    app()

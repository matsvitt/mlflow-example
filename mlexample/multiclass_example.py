from keras import layers
from keras import models
from keras.layers import Dense, Dropout
from keras.layers import Dropout
from keras.models import Sequential
from matplotlib.ticker import MaxNLocator, FuncFormatter
from os.path import expanduser
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotx
import numpy as np
import os
import pandas as pd


HOME=expanduser("~")

def read_data(filename):
    df=pd.read_csv(filename, sep=",",encoding="latin1")

    df["purchase_date"]=pd.to_datetime(df["purchase_date"],format="%Y-%m-%d")
    df["selling_date"]=pd.to_datetime(df["selling_date"],format="%Y-%m-%d")
    df['owning_age'] = (df['selling_date'] - df['purchase_date']) / np.timedelta64(1, 'D')

    # df["color_type"]="OTHER"
    # df.loc[df["ftype"]=="METALLIC","farbtyp"]="METALLIC"
    # df.loc[df["ftype"]=="PERL","farbtyp"]="PERL"


    dfx=df[['selling_price', 'purchase_price', 'selling_date',
        'purchase_date', 'color_type', 'owning_age', 'color', 'margin', 'age',
        'maker', 'owning_age_z', 'age_z', 'margin_z']].copy()
    mean_standzeit = dfx["owning_age"].mean()
    mean_alter=dfx["age"].mean()

    std_standzeit = dfx["owning_age"].std()
    std_alter=dfx["age"].std()

    mean_marge = dfx["margin"].mean()
    std_marge = dfx["margin"].std()


    dfx["owning_age_z"]=(dfx["owning_age"]-mean_standzeit)/std_standzeit
    dfx["age_z"]=(dfx["age"]-mean_alter)/std_alter
    dfx["margin_z"]=(dfx["margin"]-mean_marge)/std_marge
    return dfx

def baseline_model(X,y):
    # Create model here
    #X.shape[1] == number of features in X
    #y.shape[1] == number of classes in y 
    model = Sequential()
    model.add(Dense(16, input_shape = (X.shape[1],), activation = 'relu')) # Rectified Linear Unit Activation Function
    model.add(Dropout(0.2))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation = 'softmax')) # Softmax for multi-class classification
    # Compile model here
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    return model


def pre_process_data(dfx):
    dfx1=dfx.dropna().copy()
    dfx2=dfx1.copy()
    dfx2["margin"]=(dfx2["selling_price"]-dfx2["purchase_price"])/dfx2["purchase_price"]
    #dfx2=dfx2[np.abs(dfx2["margin"])<1].copy()
    margin_mean = dfx2["margin"].mean()
    margin_std = dfx2["margin"].std()
    dfx2["margin_z"]=(dfx2["margin"]-margin_mean)/margin_std
    
    
    dfi0=dfx2[["margin_z","owning_age_z","age_z","maker","color","color_type"]].copy()
    dfi0.rename(columns={"margin_z":"target"},inplace=True)
    dfi1=pd.get_dummies(dfi0,columns=["maker","color","color_type"])
    #dfi1=dfi0.copy()

    for col in dfi1.columns:
        if "maker" in col or "color_type" in col or "color" in col:
            dfi1[col]=dfi1[col].astype(np.int8)
    return dfi1.copy()


def generate_classes(dfi1,ranges=[0,33,66,100]):
    percentiles=np.percentile(dfi1["target"],ranges)

    for i in range(0,len(percentiles)):
        if i == 0:
            lo=-100
            hi=percentiles[i]
            print(f"lo={lo:.4f}...hi={hi:.4f}")
        else:
            lo=percentiles[i-1]
            hi=percentiles[i]
            print(f"lo={lo:.4f}...hi={hi:.4f}")
    
    dfi1["class"]=0

    for i in range(0,len(percentiles)):
        if i == 0:
            lo=-100
            hi=percentiles[i]
            
            dfi1.loc[(dfi1["target"] >= lo) & (dfi1["target"] < hi),"class"]=i
            
            #print(f"lo={lo:.4f}...hi={hi:.4f}")
        else:
            lo=percentiles[i-1]
            hi=percentiles[i]
            dfi1.loc[(dfi1["target"] >= lo) & (dfi1["target"] < hi),"class"]=i
            #print(f"lo={lo:.4f}...hi={hi:.4f}")
            
def create_data(dfi1):
    df_model=dfi1.drop(columns=["target"]).copy()
    X = df_model.drop(columns=["class"]).values
    y = df_model["class"].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)
    y = pd.get_dummies(y).astype(np.int8)
    return X, y

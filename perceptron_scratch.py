import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("diabetes.csv")
x=df.drop(columns="Outcome",axis=1)
y=df["Outcome"]
x_train,x_test,y_train,y_train=train_test_split(x,y,test_size=0.2,random_state=2)
import tensorflow
import keras
from sklearn.model_selection import perceptron


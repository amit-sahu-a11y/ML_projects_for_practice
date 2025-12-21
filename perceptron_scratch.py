import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("diabetes.csv")
x=df.drop(columns="Outcome",axis=1)
y=df["Outcome"]
x_train,x_test,y_train,y_train=train_test_split(x,y,test_size=0.2,random_state=2)
# import tensorflow
# import keras
# from sklearn.model_selection import perceptron
# print(x.shape)
x=np.insert(x,0,1,axis=1)
print(x)
def perceptron(x,y):
    
    w=np.ones(x.shape[1])
    for i in range(1000):
        lr=0.02
        j=np.random.randint(0,768)
        y_hat=np.dot(x[j],w)
        y_hat= 1 if y_hat>0 else 0
        w=w+lr*(y[j]-y_hat)*x[j]
    return w[0],w[1:]

# print(perceptron(x,y))


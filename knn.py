import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
class knn():
    def __init__(self,learning_rate,no_of_iteration):
        self.learning_rate=learning_rate
        self.no_of_iteration=no_of_iteration

    def fit(self,x,y):
        pass

    def update(self):
        pass
    def predict(self,x):
        pass
df=pd.read_csv(r"C:\Users\AMIT SAHU\Desktop\December Challenge\diabetes.csv")
x=df.drop(columns="Outcome",axis=1)
y=df["Outcome"]
model=knn(0.02,1000)
model.fit(x,y)
y_pre=model.predict(x)
acc=accuracy_score(y,y_pre)
print(acc)
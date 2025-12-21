import numpy as np
class lasso_regression():
    def __init__(self,learning_rate,no_of_iteration,lamda):
        self.learning_rate=learning_rate
        self.no_of_iteration=no_of_iteration
        self.lamda=lamda
    def fit(self,x,y):
        self.x=x
        self.y=y
        self.m,self.n=x.shape
        self.w=np.zeros(self.n)
        self.b=0

        for i in range(self.no_of_iteration):
            self.update

    def update(self):
        y=self.predict(self.x)
        for i in range(self.n):
            if self.w>0:
                dw[i]=-(2*self.x[:,i]).dot(self.y-y)+self.lamda/self.m
            else:
                dw[i]=-(2*self.x[:,i]).dot(self.y-y)-self.lamda/self.m
        db=-2*np.sum(self.y-y)/self.m
            
        self.w=self.w-self.learning_rate*dw
        self.b=self.b-self.learning_rate*db

    def predict(self,x):
        return x.dot(self.w)+self.b
import pandas as pd
df=pd.read_csv(r"C:\Users\AMIT SAHU\Desktop\December Challenge\diabetes.csv")
x=df.drop(columns="Outcome",axis=1)
y=df["Outcome"]
model=lasso_regression(0.2,1000,0.02)
model.fit(x,y)
y_pre=model.predict(x)
from sklearn.metrics import mean_absolute_error
mse=mean_absolute_error(y,y_pre)
print(mse)

import numpy as np
import pandas as pd
class logistic_regression():
    def __init__(self,learning_rate,no_of_iteration):
        self.learing_rate=learning_rate
        self.no_of_iteration=no_of_iteration

    def fit(self,x,y):
        self.m,self.n=x.shape
        self.x=x
        self.y=y
        self.w=np.zeros(self.n)
        self.b=0
        for i in range(self.no_of_iteration):
            self.update()

    def update(self):
        z=self.x.dot(self.w)+self.b
        y_pred=1/(1+np.exp(-z))
        dw=(1/self.m)*(self.x.T).dot(y_pred-self.y)
        db=(1/self.m)*np.sum(y_pred-self.y)
        self.w=self.w-self.learing_rate*(dw)
        self.b=self.b-self.learing_rate*(db)

    def predict(self,x):
        z=self.x.dot(self.w)+self.b
        y_pred=1/(1+np.exp(-z))
        y_pred=np.where(y_pred>0.5,1,0)
        return y_pred
    
df=pd.read_csv("diabetes.csv")
from sklearn.model_selection import train_test_split
x=df.drop(columns="Outcome",axis=1)
y=df["Outcome"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

model=logistic_regression(0.02,100)
model.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
val=model.predict(x_train)
acc=accuracy_score(y_train,val)
print(acc)
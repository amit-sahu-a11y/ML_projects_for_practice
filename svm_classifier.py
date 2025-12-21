import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class svm_classifier():
    def __init__(self,learning_rate,no_of_iteration,lamda_function):
        self.learning_rate=learning_rate
        self.no_of_iteration=no_of_iteration
        self.lamda_function=lamda_function

    def fit(self,x,y):
        self.m,self.n=x.shape
        self.x=x
        self.y=y
        self.w=np.zeros(self.n)
        self.b=0
        for i in range(self.no_of_iteration):
            self.update()

    def update(self):
        y_label=np.where(self.y<=0,-1,1)
        for index,x_i in enumerate(self.x):
            condition=y_label[index] *(np.dot(x_i,self.w)-self.b)>=1
            if condition==True:
                dw=2*self.lamda_function*self.w
                db=0
            else:
                dw=2*self.lamda_function*self.w-np.dot(x_i,y_label[index])
                db=y_label[index]
            self.w=self.w-self.learning_rate*dw
            self.b=self.b-self.learning_rate*db

    def predict(self,x):
        output=np.dot(x,self.w)-self.b
        predicted_labels=np.sign(output)
        y_hat=np.where(predicted_labels>-1,1,0)
        return y_hat

df=pd.read_csv("diabetes.csv")
x=df.drop(columns="Outcome",axis=1)
y=df["Outcome"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
model=svm_classifier(0.02,1000,0.2)
model.fit(x_train,y_train)
predict_acc=model.predict(x_train)
acc=accuracy_score(y_train,predict_acc)
print(acc)



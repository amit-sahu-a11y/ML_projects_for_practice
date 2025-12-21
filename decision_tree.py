import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
class decision_tree():
    def __init__(self,learning_rate,no_of_iteration):
        self.learning_rate=learning_rate
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
        pass

    def predict(self,x):
        return x.dot(self.w)+self.b
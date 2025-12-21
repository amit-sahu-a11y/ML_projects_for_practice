import numpy as np
class linear_regression():
    def __init__(self,learning_rate,no_of_epoch):
        self.learning_rate=learning_rate
        self.no_of_epoch=no_of_epoch
    
    def fit(self,x,y):
        self.m,self.n=x.shape
        self.x=x
        self.y=y
        self.w=np.zeros(self.n)
        self.b=0

        for i in range(self.no_of_epoch):
            self.update()
    
    def update(self):
        pred_y=self.prediction(self.x)
        dw=(-2*(self.x.t).dot(self.y-pred_y))/self.m
        db=(-2*np.sum(self.y-pred_y))/self.m
        self.w=self.w-self.learning_rate.dot(dw)
        self.b=self.b-self.learning_rate.dot(db)

    def prediction(self,x):
        return (x.dot(self.w)+ self.b)
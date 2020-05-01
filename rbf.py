import numpy as np
import math

class RBF():
    def __init__(self, x, theta):
        """
        @param x: the center point, 
        """
        self.center = x
        self.theta = theta
        if theta==0:
            raise ValueError
    def evaluate(self, x_i):
        diff = np.linalg.norm(np.subtract(self.center,x_i)) # l2 (Frobenius) norm
        return np.exp(np.divide(-1.0*np.power(diff,2),np.array([self.theta])))
    def __str__(self):
        print("the rbf function centered at {} with shape parameter {}".format(self.center,self.theta))
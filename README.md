# sotfmax
import numpy as np


class Solfmax:

    def __init__(self):
        self.softmax = None
        self.grad = None
        self.dnx = None

    def __call__(self, nx):
        shifted_x = nx - np.max(nx)
        ex = np.exp(shifted_x)
        sum_ex = np.sum(ex)
        self.solfmax = ex / sum_ex
        return self.solfmax
        
    def cross_entropy(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    def get_grad(self):
        self.grad = self.solfmax[:, np.newaxis] * self.solfmax[np.newaxis, :]
        for i in range(len(self.grad)):
            self.grad[i, i] -= self.solfmax[i]
        self.grad = - self.grad
        return self.grad

    def backward(self, dl):
        self.get_grad()
        self.dnx = np.sum(self.grad * dl, axis=1)
        return self.dnx
        

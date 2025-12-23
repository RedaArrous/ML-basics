import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import seaborn as sns
from scipy import stats


def gradient_calc(x, y, w, b):
    m = x.shape[0]
    f_wb = x @ w + b
    error = f_wb - y
    dj_db = np.mean(error)
    dj_dw = (x.T @ error) / m
    return dj_dw, dj_db

class LinearRegressionModel():
   
    def __init__(self, x_train=None, y_train=None, weight=None, bias=None):
        self.x = x_train
        self.y = y_train
        
        self.w = weight if weight is not None else np.zeros((x_train.shape[1],))
        self.b = bias if bias is not None else 0.0

        self.x_mean = np.mean(self.x, axis=0)
        self.x_std = np.std(self.x, axis=0)
        self.x_std[self.x_std == 0] = 1
        self.x_scaled = (self.x - self.x_mean) / self.x_std

        self.MSE = self.cost()
        self.R2 = None

    def show_params(self):
        # Convert scaled weights back to original scale
        w_orig = self.w / self.x_std
        # Correct bias transformation
        b_orig = self.b - np.sum(self.w * self.x_mean / self.x_std)
        
        print(f"Unscaled w: {w_orig} --- Unscaled b: {b_orig}")
        return w_orig, b_orig
        
    def cost(self, x=None, y=None):
        x = x if x is not None else self.x_scaled
        y = y if y is not None else self.y
        return np.mean(np.square((x @ self.w + self.b) - y)) / 2

    def fit(self, alpha=0.00001, iters=30000, show_process=True):
            print_count = max(1, iters // 10)
            cost_history = list()
            for i in range(iters):
                dj_dw, dj_db = gradient_calc(self.x_scaled, self.y, self.w, self.b)
                c = self.cost(self.x_scaled, self.y)
                cost_history.append(c)
                
                self.w -= alpha * dj_dw
                self.b -= alpha * dj_db

                if i % print_count == 0 and show_process: 
                    print(f"cost at iteration {i}: {c} --- w = {self.w} --- b = {self.b}")
            self.R2 = self.score()  
            self.MSE = c
            return self.w, self.b, cost_history

    def predict(self, x):
        x = (x - self.x_mean) / self.x_std
        return x @ self.w + self.b

    def score(self, x=None, y=None):
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        y_pred = self.predict(x)
        mean_y = np.mean(y)
        SSres = np.sum(np.square(y - y_pred))
        SStot = np.sum(np.square(y - mean_y))
        self.R2 = 1 - (SSres / SStot)
        return self.R2

        
        
        
        




















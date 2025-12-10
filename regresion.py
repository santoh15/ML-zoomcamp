import numpy as np
import pandas as pd

def dot(xi,w):
    n=len(xi)
    res=0.0
    for j in range(n):
        res=res+xi[j]*w[j]
    return res 

def train_linear_regression(X,y):
    ones= np.ones(X.shape[0])
    X=np.column_stack([ones,X])
    XTX=X.T.dot(X)
    XTX_inv=np.linalg.inv(XTX)
    w_full=XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]

def rmse(y,y_pred):
    error=y-y_pred
    se=error**2
    mse=se.mean()
    return np.sqrt(mse)
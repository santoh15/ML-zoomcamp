import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def linear_regression(xi):
    result=0.0
    for j in range(len(w)):
        result += xi[j]*w[j]
    return result

def logistic_regression(xi):
    score=w0
    for j in range(len(w)):
        score += xi[j]*w[j]
    result=sigmoid(score)
    return result
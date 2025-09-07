
import numpy as np

def rmse(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat)**2)))

def smape(y, yhat, eps=1e-8):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(100.0 * np.mean(2.0*np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + eps)))

def r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
    return float(1.0 - ss_res/ss_tot)

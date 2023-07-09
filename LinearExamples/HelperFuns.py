import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
import math


####### 1D Linear Least Square Taks
####### 

## loss w.r.t. original parameter W
def minimumCost(W_hat, G, X_1d, r_hat):
    m = len(X_1d)
    A_hat = inv(inv(G)- W_hat)
    return np.sum((A_hat@X_1d.T-r_hat)**2)/m 

## loss w.r.t. re-parameter A
def minimumCostA(A, X_1d, r_hat):
    m = len(X_1d)
    return np.sum((A*X_1d.T-r_hat)**2)/m 







## this is for nD cost function
def W_nd_Costfun(Weight, G_nd, X_nd, r_hat_nd):
  A = inv(inv(G_nd)- Weight)
  return np.sum((A@X_nd-r_hat_nd)**2)/X_1d.shape[1]

def A_nd_Costfun(A, X_nd, r_hat_nd):
  return np.sum((A@X_nd-r_hat_nd)**2)/X_1d.shape[1]

# loss function w.r.t. A
def Squared_norm_loss_A(U,A,X_input, Y_output):
    return np.sum((U@A@X_input-Y_output)**2)/Y_output.shape[1]

# ideal min. for W
def Squared_norm_loss_W(U, G, W, X_input, Y_output):
    A = pinv(pinv(G) - W)
    return np.sum((U@A@X_input-Y_output)**2)/Y_output.shape[1]
    #return np.sum((U@np.linalg.solve(einvG - W,X_input)-y_hat.T)**2)/numsamples


def CostValue_nd(num_paris, W_hat_nd, W1_set, W2_set, G_nd):
  x = y = np.arange(-1.01, 1 + 0.01, 0.01)
  CostW_nd_land = np.zeros((num_paris, len(x), len(y)))
  CostA_nd_land = np.zeros((num_paris, len(x), len(y)))
  for k in range(num_paris):
    # for k in range(1):
    W0 = W_hat_nd
    W1 = W1_set[k]
    W2 = W2_set[k]

    A0 = A_hat_nd
    A1 = inv(inv(G_nd) - W1)
    A2 = inv(inv(G_nd) - W2)
    for i in range(len(x)):
      for j in range((len(y))):
        CostW_nd_land[k, i, j] = W_nd_Costfun((x[i] * W1 + y[j] * W2) + W0, G_nd, X_nd, r_hat_nd)
        CostA_nd_land[k, i, j] = A_nd_Costfun((x[i] * A1 + y[j] * A2) + A0, X_nd, r_hat_nd)

  return (CostW_nd_land, CostA_nd_land)
import numpy as np

def prox_l2_norm_w_regularization_batch(lamb, rho, c, v):
  lamb_tilde = lamb/(1+lamb*rho)
  return prox_l2_norm_batch(lamb_tilde, lamb_tilde/lamb * v + rho*lamb_tilde*c)

def prox_l1_norm_c(x,kappa):
  xnormalized = np.divide(x, np.abs(x), out=np.zeros_like(x), where=np.abs(x)!=0)
  return xnormalized*np.fmax(np.abs(x) - kappa,0)

def prox_l2_norm_batch(lamb, v):
  return np.fmax(0, 1-lamb/np.linalg.norm(v,axis=1))[:,None] * v
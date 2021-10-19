import numpy as np
import numpy.linalg as la

def prox_l2_norm_w_regularization_batch(lamb, rho, c, v):
  lamb_tilde = lamb/(1+lamb*rho)
  return prox_l2_norm_batch(lamb_tilde, lamb_tilde/lamb * v + rho*lamb_tilde*c)

def prox_l1_norm_c(x,kappa):
  xnormalized = np.divide(x, np.abs(x), out=np.zeros_like(x), where=np.abs(x)!=0)
  return xnormalized*np.fmax(np.abs(x) - kappa,0)

def prox_l2_norm_batch(lamb, v):
  return np.fmax(0, 1-lamb/np.linalg.norm(v,axis=1))[:,None] * v

def admm_problem3(Y, p):
  A = p.A
  Phi = p.Phi
  N, M, Ng, sigma, mu, rho, taux, tauz, = p.N, p.M, p.Ng, p.sigma, p.mu, p.rho, p.taux, p.tauz
  sigma = np.abs(sigma)
  mu = np.abs(mu)
  rho = np.abs(rho)
  taux = np.abs(taux)
  tauz = np.abs(tauz)
  X = np.zeros((N,M), dtype=complex)
  Z = np.zeros((N,Ng),dtype=complex)
  U = np.zeros_like(X)
  Xhat = np.zeros_like(X)
  AtA = np.matmul(A.T.conj(),A)
  AtY = np.matmul(np.conj(A.T),Y)
  for t in range(p.maxiter):
    Xhatprev = Xhat
    # X-update
    D = Xhat - 1/rho * U
    G = 2*(np.matmul(AtA, X) - AtY)
    C = X - taux/2 * G
    X = prox_l2_norm_w_regularization_batch(mu, rho, D, C)

    # Z update
    G = - 2 * np.matmul((X - Xhat + 1/rho * U), np.conj(Phi))
    Z = prox_l1_norm_c(Z - tauz/2 * G, sigma)
    
    # U update
    Xhat = np.matmul(Z,Phi.T)
    U = U + rho*(X - Xhat)

    if t > 10:
      if la.norm(Xhatprev-Xhat) <= p.tol*la.norm(Xhatprev):
        break

  return Xhat

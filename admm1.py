import numpy as np
import numpy.linalg as la
from proximal_operators import prox_l2_norm_batch

def admm_problem1(Y, p):
  N, L, M, mu, beta, taux, gamma = p.N, p.L, p.M, p.mu, p.beta, p.taux, p.gamma
  X = np.zeros((N,M),dtype=complex)
  E = np.zeros_like(Y)
  T = np.zeros_like(Y)
  A = p.A

  AtA = np.matmul(A.T.conj(),A)
  AtY = np.matmul(np.conj(A.T),Y)

  for t in range(p.maxiter):
    Xprev = X

    E = mu*beta/(1+mu*beta) * (-np.matmul(A,X) + Y - 1/beta * T)
    
    G = 2*(np.matmul(AtA, X) + np.matmul(np.conj(A.T), E + (1/beta)*T) - AtY)
    
    D = X - taux/2 * G
    X = prox_l2_norm_batch(taux/beta, D)

    T = T + gamma*beta*(np.matmul(A, X) + E - Y)

    if t > 10:
      if np.linalg.norm(X-Xprev) <= p.tol*np.linalg.norm(Xprev):
        break
  return X

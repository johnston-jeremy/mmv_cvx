import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

def prox_l1_norm_c(x,kappa):
  # xnormalized = np.divide(x, np.abs(x), out=np.zeros_like(x), where=np.abs(x)!=0)
  if np.any(np.abs(x)) < 1:
    return 0 * np.fmax(np.abs(x) - kappa,0)
  else:
    xnormalized = np.divide(x, np.abs(x))
    # xnormalized = x/ np.abs(x)
    return xnormalized * np.fmax(np.abs(x) - kappa,0)

def ista(X, p):
  Z = p.Z0
  for _ in range(p.MAXITER):
    temp1 = np.matmul(p.M1, Z)
    temp2 = np.matmul(p.M2,X.T)
    Z = prox_l1_norm_c(temp1 + temp2, p.alpha*p.lam)

  if p.istawarmstart == True:
    p.Z0 = Z 

  return Z

def ista2(X, p, Z0):
  # used for MC Jacobian computation
  Z = Z0

  for _ in range(p.MAXITER):
    temp1 = np.matmul(p.M1, Z)
    temp2 = np.matmul(p.M2,X.T)
    Z = prox_l1_norm_c(temp1 + temp2, p.alpha*p.lam)
  return Z

def ista_grad_mc(X,Xtilde,p):
  N,M = Xtilde.shape
  J = np.zeros((N,M,M), dtype=complex)

  for n in range(N):
    # S = np.random.randn(M) + 1j*np.random.randn(M)
    for i in range(M):
      epsilon = 1e-8 * np.exp(1j*2*np.pi*np.random.rand())

      Xp = np.copy(Xtilde[n])
      Xp[i] += epsilon#*S[i]

      Zeps = ista2(Xp, p, p.Z0prev[:,n])
      Xeps = np.matmul(p.Phi, Zeps).T
      J[n,:,i] = (Xeps - X[n])/(epsilon)

  return np.sum(J, axis=0)

def vamp(Y, p):
  Phi = p.Phi
  A = p.A
  damp, damp2 = p.damp
  L = p.L
  M = p.M
  N = p.N
  R = Y
  X = np.zeros((N,M), dtype=complex)
  Xtilde = np.zeros((N,M), dtype=complex)
  p.M1 = np.eye(Phi.shape[1]) - p.alpha*np.matmul(Phi.T.conj(),Phi)
  p.M2 = p.alpha*Phi.T.conj()

  if p.denoiser == 'mmse':
    sum_gain = 0
    for m in range(M):
      sum_gain = sum_gain+(np.linalg.norm(Y[:,m]))**2
    tau = np.sqrt(sum_gain)*np.sqrt(1/(M*L))
  
  p.Z0 = np.zeros((N,Phi.shape[1]), dtype=complex).T
                                                      
  for t in range(p.maxiter):
    Xprev = X
    Xtilde = np.matmul(A.conj().T, R) + X
    if p.denoiser == 'ista':
      if not p.onsager:
        Z = ista(Xtilde, p)
        X = np.matmul(p.Phi, Z).T
        Rnew = Y - np.matmul(A, X)
        R = (1-damp) * Rnew + damp * R
      elif p.onsager:
        p.Z0prev = np.copy(p.Z0)
        Z = ista(Xtilde, p)
        X = np.matmul(p.Phi, Z).T
        if p.istagrad == 'mc':
          J = ista_grad_mc(X,Xtilde,p)
          Rnew = Y - np.matmul(A, X) + onsager * (1-damp2) * (1/L) * np.matmul(R, J)
        R = (1-damp) * Rnew + damp * R

    if p.denoiser == 'mmse':
      X, div = denoise_MMSE(Xtilde, tau, p)
      R = Y - np.matmul(A, X) + (N/L) * np.matmul(R, div)

      sum_gain = 0
      for m in range(M):
          sum_gain = sum_gain + np.linalg.norm(R[:,m])**2
      tau = np.sqrt(sum_gain)*np.sqrt(1/(M*L))

    if t > 0:
      if np.linalg.norm(X-Xprev)/np.linalg.norm(Xprev) <= 1e-6:
        break

  return X

def grad_denoise_MMSE(x, epsilon, beta, M, tau):
  g1 = -phi(x, epsilon, beta, M, tau)**2
  g2 = -M * (1-epsilon)/epsilon * np.exp(-M * (pi(x, tau, beta, M) - psi(beta,tau)))
  g3 = (1/tau - 1/(tau + beta)) * 2 * x / M
  return g1*g2*g3

def denoise_MMSE(Xtilde, tau, p):
  omega, epsilon, beta, _, ksi, _, _ = p.params
  beta = p.beta
  N,M = Xtilde.shape 
  X = np.zeros_like(Xtilde)
    
  a = beta/(beta + tau**2)
  b = (1 - epsilon)/epsilon*((beta+tau**2)/tau**2)**M
  c = beta/tau**2/(beta+tau**2)
  coeff0 = a**2/tau**2

  inners = np.sum(Xtilde.real**2 + Xtilde.imag**2, axis=1)
  t0 = b*np.exp(-c*np.abs(inners))
  t = 1 + t0
  coeff1 = np.diag(a/t)
  coeff2 = np.diag(t0/t**2)
  X = np.matmul(coeff1, Xtilde)
  outer = coeff0 * np.matmul(np.matmul(Xtilde.T.conj(), coeff2), Xtilde)
  d = np.eye(M)*np.sum(a/t)

  D = (outer + d)/N
  
  return X, D

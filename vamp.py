import numpy as np
# import autograd as ag
# import autograd.numpy as np
# import autogradista as ista
import matplotlib.pyplot as plt
from time import time
from pdb import set_trace

def CN(d1,d2,variance):

  return np.sqrt(variance/2) * (np.random.randn(d1,d2) + 1j*np.random.randn(d1,d2))

def RN(d1,d2,variance):

  return np.sqrt(variance) * np.random.randn(d1,d2)

def prox_l1_norm_c(x,kappa):
  # xnormalized = np.divide(x, np.abs(x), out=np.zeros_like(x), where=np.abs(x)!=0)
  if np.any(np.abs(x)) < 1:
    return 0 * np.fmax(np.abs(x) - kappa,0)
  else:
    xnormalized = np.divide(x, np.abs(x))
    # xnormalized = x/ np.abs(x)
    return xnormalized * np.fmax(np.abs(x) - kappa,0)

def ista_distr(X, Z0, p):
  # import pdb; pdb.set_trace()
  # n = p.Phi.shape[1]
  Z = np.copy(Z0)
  set_trace()
  # print(p.MAXITER)
  for i in range(p.MAXITER):
    # xold = x
    temp1 = np.matmul(p.M1, Z)
    temp2 = np.matmul(p.M2,X.T)
    # print('ista')
    # print(temp1.real)
    # print(temp1.imag)
    # print(temp2.real)
    # print(temp2.imag)
    # print('ista')
    Z = prox_l1_norm_c(temp1 + temp2, p.alpha*p.lam)
    # Z = soft_threshold(Z - p.alpha*np.matmul(np.conj(p.Phi.T), np.matmul(p.Phi, Z)) + p.alpha*np.matmul(np.conj(p.Phi.T), y), p.alpha*p.lam)
    # if np.linalg.norm(x-xold) < 1e-6:
    #   break
    Xhat = np.matmul(Phi, Z).T

    errs.append(np.sum(np.abs(Xtrue-Xhat)**2)/np.sum(np.abs(Xtrue)**2))

  # if p.istawarmstart == True:
  #   p.Z0 = Z 

  # return x
  return Z


def ista(X, p, Xtrue):
  # import pdb; pdb.set_trace()
  # n = p.Phi.shape[1]
  Z = p.Z0
  # set_trace()
  errs = []
  # print(p.MAXITER)
  for i in range(p.MAXITER):
    # xold = x
    temp1 = np.matmul(p.M1, Z)
    temp2 = np.matmul(p.M2,X.T)
    # print('ista')
    # print(temp1.real)
    # print(temp1.imag)
    # print(temp2.real)
    # print(temp2.imag)
    # print('ista')
    Z = prox_l1_norm_c(temp1 + temp2, p.alpha*p.lam)
    # Z = soft_threshold(Z - p.alpha*np.matmul(np.conj(p.Phi.T), np.matmul(p.Phi, Z)) + p.alpha*np.matmul(np.conj(p.Phi.T), y), p.alpha*p.lam)
    # if np.linalg.norm(x-xold) < 1e-6:
    #   break
    Xhat = np.matmul(p.Phi, Z).T

    errs.append(np.sum(np.abs(Xtrue-Xhat)**2)/np.sum(np.abs(Xtrue)**2))

  if p.istawarmstart == True:
    p.Z0 = Z 

  # return x
  return Z, errs

def ista2(X, p, Xtrue, z0):
  # used for MC Jacobian computation
  Z = z0

  # print(p.MAXITER)
  for i in range(p.MAXITER):
    # xold = x
    temp1 = np.matmul(p.M1, Z)
    temp2 = np.matmul(p.M2,X.T)
    # print('ista')
    # print(temp1.real)
    # print(temp1.imag)
    # print(temp2.real)
    # print(temp2.imag)
    # print('ista')
    Z = prox_l1_norm_c(temp1 + temp2, p.alpha*p.lam)
    # Z = soft_threshold(Z - p.alpha*np.matmul(np.conj(p.Phi.T), np.matmul(p.Phi, Z)) + p.alpha*np.matmul(np.conj(p.Phi.T), y), p.alpha*p.lam)
    # if np.linalg.norm(x-xold) < 1e-6:
    #   break
    Xhat = np.matmul(p.Phi, Z).T

    # errs.append(np.sum(np.abs(Xtrue-Xhat)**2)/np.sum(np.abs(Xtrue)**2))

  # if p.istawarmstart == True:
  #   p.Z0 = Z 

  # return x
  return Z

def ista_ag(p, x, z):
  # used for autograd Jacobian computation
  p.zzz = z
  def ista_real(X):
    # print(p.MAXITER)
    Z = p.zzz
    for i in range(p.MAXITER):

      temp1 = np.matmul(np.real(p.M1), np.real(Z)) - np.matmul(np.imag(p.M1), np.imag(Z))
      temp2 = np.matmul(np.real(p.M2),np.real(X.T)) + np.matmul(np.imag(p.M2),np.imag(X.T))

      Z = prox_l1_norm_c(temp1 + temp2, p.alpha*p.lam)

      Xhat = np.matmul(p.Phi, Z).T

    # return x
    return np.real(Z)

  def ista_imag(X):
    # print(p.MAXITER)
    Z = p.zzz
    for i in range(p.MAXITER):

      temp1 = np.matmul(np.imag(p.M1), np.real(Z)) + np.matmul(np.real(p.M1), np.imag(Z))
      temp2 = np.matmul(np.imag(p.M2),np.real(X.T)) + np.matmul(np.real(p.M2),np.imag(X.T))

      Z = prox_l1_norm_c(temp1 + temp2, p.alpha*p.lam)

      Xhat = np.matmul(p.Phi, Z).T

    return np.imag(Z)

  gr= ag.jacobian(ista_real)
  gi = ag.jacobian(ista_imag)
  
  return (gr(x.real) + gi(x.imag))/2

def ista_grad_mc(X,Xtilde,Xtrue,p):
  N,M = Xtilde.shape
  J = np.zeros((N,M,M), dtype=complex)
  # epsilon = np.max(np.abs(Xtilde))/1000
  

  for n in range(N):

    # S = np.random.randn(M) + 1j*np.random.randn(M)
    for i in range(M):
      # set_trace()
      # epsilon = (np.random.randn() + 1j*np.random.randn()) 
      # epsilon = 1e-8 * epsilon/np.abs(epsilon)
      epsilon = 1e-8 * np.exp(1j*2*np.pi*np.random.rand())

      Xp = np.copy(Xtilde[n])
      Xp[i] += epsilon#*S[i]

      Zeps = ista2(Xp, p, Xtrue[n], p.Z0prev[:,n])
      Xeps = np.matmul(p.Phi, Zeps).T
      J[n,:,i] = (Xeps - X[n])/(epsilon)


  # J = (J.real + J.imag)/2
      # temp = np.mean(S.conj()*(Xeps - X[n]))/epsilon
  # print(np.mean(J))
  return np.sum(J,axis=0)



def vamp_AP(Y, Xtrue, p, onsager, ind, X0=None, Z0=None, R0=None):
  if ind is not None:
    A = p.A[:,ind]
  else:
    A = p.A
  # A = p.A
  damp, damp2 = p.damp
  L = p.L
  # M = p.M
  # N = p.N
  N,M = Xtrue.shape
  R = Y
  if X0 is not None:
    X = np.copy(X0)
  else:
    X = np.zeros((N,M), dtype=complex)
  Xtilde = np.zeros((N,M), dtype=complex)
  omega = p.N/p.L
  epsilon = p.K/p.N
  omega, epsilon, beta, sigma, ksi, maxiter, alpha = p.params
  # p.M1 = np.eye(Phi.shape[1]) - p.alpha*np.matmul(Phi.T.conj(),Phi)
  # p.M2 = p.alpha*Phi.T.conj()

  if p.denoiser == 'mmse':
    sum_gain = 0;
    for m in range(M):
      sum_gain = sum_gain+(np.linalg.norm(Y[:,m]))**2
    tau = np.sqrt(sum_gain)*np.sqrt(1/(M*L))
  
  # if Z0 is None:
  #   p.Z0 = np.zeros((N,Phi.shape[1]), dtype=complex).T
  # else:
  #   p.Z0 = Z0
  
  if R0 is not None:
    R = R0

  p1 = []
  p2 = []
  tt = time()
  err_ista = []

  for t in range(p.maxiter):
    grad = []
    Xtilde = np.matmul(A.conj().T, R) + X
    if p.denoiser == 'ista':

      if not onsager:
        Z, err = ista_distr(Xtilde, p, Xtrue, Phi)
        err_ista.append(err)
        X = np.matmul(Phi, Z).T
        Rnew = Y - np.matmul(A, X)
        R = (1-damp) * Rnew + damp * R

      elif onsager:
        p.Z0prev = np.copy(p.Z0)
        Z, err = ista(Xtilde, p, Xtrue)
        X = np.matmul(p.Phi, Z).T

        if p.istagrad == 'mc':
          J = ista_grad_mc(X,Xtilde,Xtrue,p)
          Rnew = Y - np.matmul(A, X) + onsager * (1-damp2) * (1/L) * np.matmul(R, J)

        elif p.istagrad == 'regular':
          grad = []
          for n in range(N):
            g, z = ista_grad_complex_2(Phi, p.M1, p.M2, Xtilde[n], p.Z0[:,n], p.MAXITER, p.alpha*p.lam)
            X[n] = np.matmul(Phi, z)
            grad.append(g)
          Rnew = Y - np.matmul(A, X) + onsager * (1-damp2) * (1/L) * np.matmul(R, sum(grad))
        
        elif p.istagrad == 'ag':
          grad = []
          for n in range(N):
            grad.append(ista_ag(p, Xtilde[n], p.Z0prev[:,n]))
          J = np.ones((M,M))*np.sum(grad)
          Rnew = Y - np.matmul(A, X) + onsager * (1-damp2) * (1/L) * np.matmul(R,J)
          print(np.sum(J))
          # set_trace()
        
        R = (1-damp) * Rnew + damp * R
      

    if p.denoiser == 'mmse':

      X, div = denoise_MMSE_2(Xtilde, tau, p)

      R = Y - np.matmul(A, X) + onsager * (N/L) * np.matmul(R, div)

      sum_gain = 0
      for m in range(M):
          sum_gain = sum_gain + np.linalg.norm(R[:,m])**2
      tau = np.sqrt(sum_gain)*np.sqrt(1/(M*L))

    p2.append(np.sum(grad))
    p1.append(np.sum(np.abs(X-Xtrue)**2)/np.sum(np.abs(Xtrue)**2))
  # print()

    # print(10*'-')
    # print(np.linalg.norm(Y - np.matmul(A, X)))
  # plt.subplot(121)
  # plt.plot(p1)
  # plt.title('NMSE')
  # plt.subplot(122)
  # plt.plot(p2)
  # plt.title('Onsager')
  # plt.show()
  # if p1[-1]>100:
  #   print('Diverged')
  #   return np.array([0]),np.array([0]),1
  # print(time() - T)
  return p1, detect(X,Xtrue), np.array(err_ista), X, R

def vamp(Y, p, onsager):
  Phi = p.Phi
  A = p.A
  damp, damp2 = p.damp
  L = p.L
  M = p.M
  N = p.N
  R = Y
  X = np.zeros((N,M), dtype=complex)
  Xtilde = np.zeros((N,M), dtype=complex)
  omega = p.N/p.L
  epsilon = p.K/p.N
  omega, epsilon, beta, sigma, ksi, maxiter, alpha = p.params
  p.M1 = np.eye(Phi.shape[1]) - p.alpha*np.matmul(Phi.T.conj(),Phi)
  p.M2 = p.alpha*Phi.T.conj()

  if p.denoiser == 'mmse':
    sum_gain = 0;
    for m in range(M):
      sum_gain = sum_gain+(np.linalg.norm(Y[:,m]))**2
    tau = np.sqrt(sum_gain)*np.sqrt(1/(M*L))
  
  p.Z0 = np.zeros((N,Phi.shape[1]), dtype=complex).T
                                                      
  p1 = []
  p2 = []
  tt = time()
  err_ista = []

  for t in range(p.maxiter):
    Xprev = X
    grad = []
    Xtilde = np.matmul(A.conj().T, R) + X
    if p.denoiser == 'ista':

      if not onsager:
        Z, err = ista(Xtilde, p, Xtrue)
        err_ista.append(err)
        X = np.matmul(p.Phi, Z).T
        Rnew = Y - np.matmul(A, X)
        R = (1-damp) * Rnew + damp * R
        # plt.plot(err)
        # plt.show()


      elif onsager:
        p.Z0prev = np.copy(p.Z0)
        Z, err = ista(Xtilde, p, Xtrue)
        X = np.matmul(p.Phi, Z).T
        # plt.plot(err)
        # plt.show()

        if p.istagrad == 'mc':
          J = ista_grad_mc(X,Xtilde,Xtrue,p)
          Rnew = Y - np.matmul(A, X) + onsager * (1-damp2) * (1/L) * np.matmul(R, J)

        elif p.istagrad == 'regular':
          grad = []
          for n in range(N):
            g, z = ista_grad_complex_2(Phi, p.M1, p.M2, Xtilde[n], p.Z0[:,n], p.MAXITER, p.alpha*p.lam)
            X[n] = np.matmul(Phi, z)
            grad.append(g)
          Rnew = Y - np.matmul(A, X) + onsager * (1-damp2) * (1/L) * np.matmul(R, sum(grad))
        
        elif p.istagrad == 'ag':
          grad = []
          for n in range(N):
            grad.append(ista_ag(p, Xtilde[n], p.Z0prev[:,n]))
          J = np.ones((M,M))*np.sum(grad)
          Rnew = Y - np.matmul(A, X) + onsager * (1-damp2) * (1/L) * np.matmul(R,J)
          print(np.sum(J))
          # set_trace()
        
        # R = (1-damp) * Rnew + damp * R
        R = (1-damp) * Rnew
      

    if p.denoiser == 'mmse':
      X, div = denoise_MMSE_2(Xtilde, tau, p)

      R = Y - np.matmul(A, X) + (N/L) * np.matmul(R, div)

      sum_gain = 0
      for m in range(M):
          sum_gain = sum_gain + np.linalg.norm(R[:,m])**2
      tau = np.sqrt(sum_gain)*np.sqrt(1/(M*L))

    if t > 0:
      if np.linalg.norm(X-Xprev)/np.linalg.norm(Xprev) <= 0.001:
        break
    # p2.append(np.sum(grad))
    # p1.append(np.sum(np.abs(X-Xtrue)**2)/np.sum(np.abs(Xtrue)**2))

  return X

def vamp_nmse(Y, Xtrue, p, onsager):
  Phi = p.Phi
  A = p.A
  damp, damp2 = p.damp
  L = p.L
  M = p.M
  N = p.N
  R = Y
  X = np.zeros((N,M), dtype=complex)
  Xtilde = np.zeros((N,M), dtype=complex)
  omega = p.N/p.L
  epsilon = p.K/p.N
  omega, epsilon, beta, sigma, ksi, maxiter, alpha = p.params
  p.M1 = np.eye(Phi.shape[1]) - p.alpha*np.matmul(Phi.T.conj(),Phi)
  p.M2 = p.alpha*Phi.T.conj()

  if p.denoiser == 'mmse':
    sum_gain = 0;
    for m in range(M):
      sum_gain = sum_gain+(np.linalg.norm(Y[:,m]))**2
    tau = np.sqrt(sum_gain)*np.sqrt(1/(M*L))
  
  p.Z0 = np.zeros((N,Phi.shape[1]), dtype=complex).T
                                                      
  p1 = []
  p2 = []
  tt = time()
  err_ista = []

  for t in range(p.maxiter):
    grad = []
    Xtilde = np.matmul(A.conj().T, R) + X
    if p.denoiser == 'ista':

      if not onsager:
        Z, err = ista(Xtilde, p, Xtrue)
        err_ista.append(err)
        X = np.matmul(p.Phi, Z).T
        Rnew = Y - np.matmul(A, X)
        R = (1-damp) * Rnew + damp * R
        # plt.plot(err)
        # plt.show()


      elif onsager:
        p.Z0prev = np.copy(p.Z0)
        Z, err = ista(Xtilde, p, Xtrue)
        X = np.matmul(p.Phi, Z).T
        # plt.plot(err)
        # plt.show()

        if p.istagrad == 'mc':
          J = ista_grad_mc(X,Xtilde,Xtrue,p)
          Rnew = Y - np.matmul(A, X) + onsager * (1-damp2) * (1/L) * np.matmul(R, J)

        elif p.istagrad == 'regular':
          grad = []
          for n in range(N):
            g, z = ista_grad_complex_2(Phi, p.M1, p.M2, Xtilde[n], p.Z0[:,n], p.MAXITER, p.alpha*p.lam)
            X[n] = np.matmul(Phi, z)
            grad.append(g)
          Rnew = Y - np.matmul(A, X) + onsager * (1-damp2) * (1/L) * np.matmul(R, sum(grad))
        
        elif p.istagrad == 'ag':
          grad = []
          for n in range(N):
            grad.append(ista_ag(p, Xtilde[n], p.Z0prev[:,n]))
          J = np.ones((M,M))*np.sum(grad)
          Rnew = Y - np.matmul(A, X) + onsager * (1-damp2) * (1/L) * np.matmul(R,J)
          print(np.sum(J))
          # set_trace()
        
        # R = (1-damp) * Rnew + damp * R
        R = (1-damp) * Rnew
      

    if p.denoiser == 'mmse':
      X, div = denoise_MMSE_2(Xtilde, tau, p)

      R = Y - np.matmul(A, X) + onsager * (N/L) * np.matmul(R, div)

      sum_gain = 0
      for m in range(M):
          sum_gain = sum_gain + np.linalg.norm(R[:,m])**2
      tau = np.sqrt(sum_gain)*np.sqrt(1/(M*L))

    p2.append(np.sum(grad))
    p1.append(np.sum(np.abs(X-Xtrue)**2)/np.sum(np.abs(Xtrue)**2))
  # print()

    # print(10*'-')
    # print(np.linalg.norm(Y - np.matmul(A, X)))
  # plt.subplot(121)
  # plt.plot(p1)
  # plt.title('NMSE')
  # plt.subplot(122)
  # plt.plot(p2)
  # plt.title('Onsager')
  # plt.show()
  # if p1[-1]>100:
  #   print('Diverged')
  #   return np.array([0]),np.array([0]),1
  # print(time() - T)
  return np.array(p1), np.array(err_ista)

def vamp_ista(Y, p):
  Phi = p.Phi
  A = p.A
  damp, damp2 = p.damp
  L = p.L
  M = p.M
  N = p.N
  R = Y
  X = np.zeros((N,M), dtype=complex)
  Xtilde = np.zeros((N,M), dtype=complex)
  omega, epsilon, beta, sigma, ksi, maxiter, alpha = p.params
  p.M1 = np.eye(Phi.shape[1]) - p.alpha*np.matmul(Phi.T.conj(),Phi)
  p.M2 = p.alpha*Phi.T.conj()
  
  p.Z0 = np.zeros((N,Phi.shape[1]), dtype=complex).T
                                   

  for t in range(p.maxiter):
    grad = []
    Xtilde = np.matmul(A.conj().T, R) + X
    Z, err = ista(Xtilde, p, 1)
    X = np.matmul(p.Phi, Z).T
    Rnew = Y - np.matmul(A, X)
    R = (1-damp) * Rnew + damp * R
  
  return X


def phi(x, epsilon, beta, M, tau):
  return 1/(1 + (1-epsilon)/epsilon * np.exp(-M*(pi(x, tau, beta, M) - psi(beta,tau))))

def psi(beta,tau):
  return np.log(1 + beta/tau)

def pi(x, tau, beta, M):
  return  (1/tau - 1/(tau + beta)) * np.dot(np.conj(x), x) / M

def grad_denoise_MMSE(x, epsilon, beta, M, tau):
  g1 = -phi(x, epsilon, beta, M, tau)**2
  g2 = -M * (1-epsilon)/epsilon * np.exp(-M * (pi(x, tau, beta, M) - psi(beta,tau)))
  g3 = (1/tau - 1/(tau + beta)) * 2 * x / M
  return g1*g2*g3

# def grad_mc(Xtilde, t, tau_t, phi_t, params)
# denoise_MMSE(Xtilde, t, tau_t, phi_t, params)
def denoise_MMSE(Xtilde, t, tau_t, phi_t, p):
  omega, epsilon, beta, _, ksi, _, _ = p.params
  beta = p.beta
  sigma = p.sigma_noise
  N,M = Xtilde.shape 
  X = np.zeros_like(Xtilde)

  phi = np.zeros(N, dtype=complex)
  grad = np.zeros(N, dtype=complex)
  if t == 0:
    tau = sigma**2/ksi + omega*epsilon*beta
  else:
    theta = 0
    for n in range(N):
      theta = theta + phi_t[n] * (1 - phi_t[n]) * beta**2 / (beta + tau_t)**2 * np.dot(np.conj(Xtilde[n]), Xtilde[n])
    theta = 1/M * 1/N * theta

    tau = sigma**2/ksi + omega*epsilon*beta*tau_t/(beta + tau_t) + omega*theta

  for n in range(N):
    
    pi = (1/tau - 1/(tau + beta)) * np.dot(np.conj(Xtilde[n]), Xtilde[n]) / M
    psi = np.log(1 + beta/tau)
    
    phi[n] = 1/(1 + (1-epsilon)/epsilon * np.exp(-M*(pi - psi)) )
    
    X[n] = phi[n]*beta/(beta + tau) * Xtilde[n]

  return X, phi, tau

def denoise_MMSE_2(Xtilde, tau, p):
  omega, epsilon, beta, _, ksi, _, _ = p.params
  beta = p.beta
  sigma = p.sigma_noise
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

def khatri_rao(a, b):
  c = a[...,:,np.newaxis,:] * b[...,np.newaxis,:,:]
  # collapse the first two axes
  return c.reshape((-1,) + c.shape[2:])

def a(M,Ngk):
  Nrx = int(np.sqrt(M))
  Ntx = int(np.sqrt(M))

  Ngk = 2*M;

  ngk = np.arange(Ngk)
  nrx = np.arange(Nrx)
  ntx = np.arange(Ntx)

  hrx = np.exp(1j*2*np.pi*np.outer(nrx,ngk)/Ngk)
  htx = np.exp(1j*2*np.pi*np.outer(ntx,ngk)/Ngk)

  return khatri_rao(hrx,htx)

def scaled_noise(A,X,SNR):
  AX = np.matmul(A,X)
  norm_AX = np.linalg.norm(AX)
  
  noise = RN(L,M,1);
  noise = 10**(-SNR/20) * norm_AX * noise/np.linalg.norm(noise)
  SNR_emp = 20*np.log10(norm_AX/np.linalg.norm(noise))
  # print(SNR_emp)
  stdev = np.sqrt(np.mean((noise - np.mean(noise))**2))
  return noise,stdev

if __name__ == '__main__':
  Nsamp = 50
  maxiter = 20
  beta = 1 # channel variance
  # sigma = .01 # noise stdev
  
  ksi = 1
  N = 10 # number of potential users
  L = 5 # pilot sequence length
  M = 10 # number of antennas
  Ng = 2*M
  K = 2 # number of active users
  epsilon = K/N
  A = RN(L, N, 1/L)
  A = np.matmul(A,np.diag(1/np.sqrt(np.sum(A**2,axis=0))))
  # Phi = array(M,Ng)
  # Phi = np.cos(np.pi*np.arange(M)[:,None]*np.arange(Ng)/(Ng*M))
  Phi = RN(M,Ng,1/M)
  Phi = np.matmul(Phi,np.diag(1/np.sqrt(np.sum(Phi**2,axis=0))))
  w,v = np.linalg.eig( np.matmul( np.conj(Phi.T), Phi ) )
  alpha = 1/max(np.abs(w))  

  nmse0 = 0
  nmse1 = 0
  grad = 0
  
  SNR = 15
  divcount = 0
  import time

  p = ista.problem()
  p.alpha = alpha
  p.Phi = Phi
  p.A = A
  
  for j in range(Nsamp):
    print('sample', j)
    channel_sparsity = 2
    s = np.zeros((Ng,N))
    for i in range(N):
      s[np.random.permutation(Ng)[:channel_sparsity],i] = np.random.normal(size=(channel_sparsity,))
    H = np.matmul(Phi, s).T
    # H = RN(N, M, beta)
    # Z = RN(L, M, sigma**2)
    mask = np.zeros(N)
    
    ind = np.random.permutation(N)[:int(epsilon*N)]
    mask[ind] = 1
    mask = np.diag(mask)
    X = np.matmul(mask, H)
    Z, sigma = scaled_noise(A,X,SNR)
    Y = np.matmul(A, X) + Z

    omega = N/L
    # sigma = sigma
    p.params = omega, epsilon, beta, sigma, ksi, maxiter, alpha
    # print(sigma)
    # print(10*np.log10(np.linalg.norm(np.matmul(A, X))**2/np.linalg.norm(Z)**2))
    # import pdb; pdb.set_trace() 
    t = time.time()

    p.MAXITER = 20
    damp = 0.8
    damp2 = 0
    p.lam = 1e-1
    p.damp = damp, damp2
    p.denoiser = 'mmse'
    p10, p20, divflag = vamp(Y, X, p, onsager=0)
    nmse0 = nmse0 + p10[-1]
    # grad = grad + p20
    # print(p.denoiser, 10*np.log10(p10[-1]))

    p.MAXITER = 20
    damp = 0.8
    damp2 = 0
    p.lam = 1e-1
    p.damp = damp, damp2
    p.denoiser = 'ista'
    p11, p21, divflag = vamp(Y, X, p, onsager=0)
    nmse1 = nmse1 + p11[-1]
    grad = grad + p21
    # print(p.denoiser, 10*np.log10(p11[-1]))
    if 1: #(j+1)%50==0:
      print('j =', j)
      print('mmse:', 10*np.log10(p10[-1]))
      print('ista:', 10*np.log10(p11[-1]))
    divcount = divcount + divflag

  print('nmse0:', 10*np.log10(nmse0/Nsamp))
  print('nmse1:', 10*np.log10(nmse1/Nsamp))
    
  stop_tol = 1e-5
  for i in range(10,maxiter):
    if abs((nmse0[i]-nmse0[i-1])/nmse0[i-1]) < stop_tol:
      if nmse0[i] < 1:
          break

  print(i)
  for i in range(10,maxiter):
    if abs((nmse1[i]-nmse1[i-1])/nmse1[i-1]) < stop_tol:
      if nmse1[i] < 1:
          break
  print(i)
  
    
    #time.time() - t

  # plt.subplot(121)
  plt.plot(nmse0/Nsamp)
  plt.title('NMSE0')
  # plt.subplot(122)
  # plt.plot(grad/Nsamp)
  # plt.title('Onsager')

  plt.figure()
  # plt.subplot(121)
  plt.plot(nmse1/Nsamp)
  plt.title('NMSE1')
  # plt.subplot(122)
  # plt.plot(grad/Nsamp)
  # plt.title('Onsager')

  plt.show()
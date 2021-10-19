import numpy as np
from pdb import set_trace

def CN(d1,d2,variance):

  return np.sqrt(variance/2) * (np.random.randn(d1,d2) + 1j*np.random.randn(d1,d2))

def RN(d1,d2,variance):

  return np.sqrt(variance) * np.random.randn(d1,d2)

def scaled_noise(A,X,SNR):
  AX = np.matmul(A,X)
  norm_AX = np.linalg.norm(AX)
  L,M = AX.shape
  noise = RN(L,M,1);
  noise = 10**(-SNR/20) * norm_AX * noise/np.linalg.norm(noise)
  SNR_emp = 20*np.log10(norm_AX/np.linalg.norm(noise))
  # print('SNR_emp=',SNR_emp)
  stdev = np.sqrt(np.mean((noise - np.mean(noise))**2))
  return noise,stdev

def scaled_noise_c(A,X,SNR):
  AX = np.matmul(A,X)
  norm_AX = np.linalg.norm(AX)
  L,M = AX.shape
  noise = CN(L,M,1);
  noise = 10**(-SNR/20) * norm_AX * noise/np.linalg.norm(noise)
  SNR_emp = 20*np.log10(norm_AX/np.linalg.norm(noise))
  # print('SNR_emp=',SNR_emp)
  stdev = np.sqrt(np.mean(np.abs(noise - np.mean(noise))**2))
  return noise,stdev

def gen_c(p,Nsamp,mode):
  J,N,L,M,Ng,K,SNR = p.J, p.N, p.L, p.M, p.Ng, p.K, p.SNR
  Phi = p.Phi
  A = p.A

  Y = np.zeros((Nsamp,L,M),dtype=complex)
  X = np.zeros((Nsamp,N,M),dtype=complex)
  Z = np.zeros((Nsamp,N,Ng),dtype=complex)
  # set_trace()
  for j in range(Nsamp):
    ind = np.random.permutation(N)[:K]

    if mode == 'mmwave':
      for i in ind:
        Z[j,i,np.random.permutation(Ng)[:J]] \
          = np.random.normal(size=(J,)) \
          + 1j*np.random.normal(size=(J,))
      X[j] = np.matmul(Z[j], Phi.T)
    elif mode =='gaussian':
      X[j,ind] = CN(K,M,1)

    noise, sigma = scaled_noise_c(A,X[j],SNR)

    Y[j] = np.matmul(A, X[j]) + noise

  return Y, X, Z, sigma
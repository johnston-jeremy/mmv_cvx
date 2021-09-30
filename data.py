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


def gen_r(p,Nsamp,channel_sparsity,N,L,M,Ng,K,SNR):
  Phi = p.Phi
  A = p.A
  epsilon = K/N

  Y = np.zeros((Nsamp,L,M),dtype=complex)
  X = np.zeros((Nsamp,N,M),dtype=complex)

  for j in range(Nsamp):
    s = np.zeros((Ng,N), dtype=complex)
    for i in range(N):
      s[np.random.permutation(Ng)[:channel_sparsity],i] = np.random.normal(size=(channel_sparsity,))
    H = np.matmul(Phi, s).T
    mask = np.zeros(N)
    ind = np.random.permutation(N)[:int(epsilon*N)]
    mask[ind] = 1
    mask = np.diag(mask)
    X[j] = np.matmul(mask, H)

    Z, sigma = scaled_noise(A,X[j],SNR)

    Y[j] = np.matmul(A, X[j]) + Z

  return Y, X, sigma


def gen_c(A,Nsamp,N,L,M,K,SNR):

  Y = np.zeros((Nsamp,L,M),dtype=complex)
  X = np.zeros((Nsamp,N,M),dtype=complex)

  for j in range(Nsamp):
    ind = np.random.permutation(N)[:K]
    X[j,ind] = CN(K,M,1)
    Z, sigma = scaled_noise_c(A,X[j],SNR)
    Y[j] = np.matmul(A, X[j]) + Z

  return Y, X, sigma

def gen_c_2(p,Nsamp,channel_sparsity,N,L,M,Ng,K,SNR):
  Phi = p.Phi
  A = p.A
  epsilon = K/N

  Y = np.zeros((Nsamp,L,M),dtype=complex)
  X = np.zeros((Nsamp,N,M),dtype=complex)
  Z = np.zeros((Nsamp,N,Ng),dtype=complex)

  for j in range(Nsamp):
    # set_trace()
    # z = Z[j]
    rows = np.random.permutation(N)[:2]
    for r in rows:
      row = [r]*channel_sparsity
      col = np.random.permutation(Ng)[:channel_sparsity]
      Z[j][row,col] = np.random.normal(size=(channel_sparsity,)) \
                 + 1j*np.random.normal(size=(channel_sparsity,))
      # indr = np.random.permutation(M)[:channel_sparsity]
      # indc = np.random.permutation(Ng)[:K]
      # ind = np.meshgrid(indr,indc)
      # z[ind[0],ind[1]] = \
      #     np.random.normal(size=(channel_sparsity,K)) \
      #   + 1j*np.random.normal(size=(channel_sparsity,K))

    # h = np.matmul(Phi, s).T
    # h = CN(K,M,1)
    # ind = np.random.permutation(N)[:K]
    # X[j,ind] = h
    
    X[j] = np.matmul(Z[j],Phi.T)
    noise, sigma = scaled_noise_c(A,X[j],SNR)

    Y[j] = np.matmul(A, X[j]) + noise
    # set_trace()
    
    
  return Y, X, Z, sigma
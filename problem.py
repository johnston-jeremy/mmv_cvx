import numpy as np
def RN(d1,d2,variance):

  return np.sqrt(variance) * np.random.randn(d1,d2)

def CN(d1,d2,variance):
  if d2 is None:
    return np.sqrt(variance/2) * (np.random.randn(d1) + 1j*np.random.randn(d1))
  return np.sqrt(variance/2) * (np.random.randn(d1,d2) + 1j*np.random.randn(d1,d2))

def khatri_rao(a, b):
    c = a[...,:,np.newaxis,:] * b[...,np.newaxis,:,:]
    # collapse the first two axes
    return c.reshape((-1,) + c.shape[2:])

def Phi_mimo(Ntxrx):
  # Ntx = 5;
  # Nrx = 8;
  Ntx, Nrx = Ntxrx
  # Ntx = 5;
  # Nrx = 2;

  # Ntx = 2
  # Nrx = 1

  Ngk = 2*Ntx*Nrx;

  ngk = np.arange(Ngk)
  nrx = np.arange(Nrx)
  ntx = np.arange(Ntx)

  hrx = np.exp(1j*2*np.pi*np.outer(nrx,ngk)/Ngk)
  htx = np.exp(1j*2*np.pi*np.outer(ntx,ngk)/Ngk)

  return khatri_rao(hrx,htx)

def Phi_siso(M,N):
  m = np.arange(M)[:,None]
  n = np.arange(N)
  return np.exp(1j*2*np.pi*m*n/N)/np.sqrt(M)

class problem():
  def __init__(self,N,L,M,Ng,K,Ntxrx,J):

    self.N = N
    self.M = np.prod(Ntxrx)
    self.L = L
    self.K = K
    self.Ng = Ng
    self.J = J

    # Real A
    # A = RN(L, N, 1/L)
    # A = np.matmul(A,np.diag(1/np.sqrt(np.sum(A**2,axis=0))))

    # Complex A
    A = CN(L, N, 1/L)
    # A = np.matmul(A,np.diag(1/np.sqrt(np.sum(np.abs(A)**2,axis=0))))

    # Phi = array(M,Ng)
    # Phi = np.cos(np.pi*np.arange(M)[:,None]*np.arange(Ng)/(Ng*M))
    # Phi = RN(M,Ng,1/M)
    # Phi = np.matmul(Phi,np.diag(1/np.sqrt(np.sum(Phi**2,axis=0))))

    # Phi = Phi_mimo(Ntxrx)
    Phi = Phi_siso(M,Ng)
    
    # Phi = CN(M,Ng,1/M)
    # Phi = np.matmul(Phi,np.diag(1/np.sqrt(np.sum(np.abs(Phi)**2,axis=0))))

    w,v = np.linalg.eig( np.matmul( np.conj(Phi.T), Phi ) )
    alpha = 1/max(np.abs(w))

    # import pdb; pdb.set_trace()
    self.alpha = np.float32(alpha)
    self.alpha0 = np.float32(alpha)
    # self.Phi = np.float32(Phi)
    # self.A = np.float32(A)
    self.Phi = np.complex64(Phi)
    self.A = A

    self.M2 = np.matmul(np.conj(A.T), A)
    self.M3 = np.conj(A.T)

    
import tqdm
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from data import gen_c, gen_c_2
from problem import problem
from pdb import set_trace
from multiprocessing import Pool, Manager


def f1():
  M = 8
  N = 50
  L = 12
  K = 3
  A = np.random.normal(size=(L,N)) + 1j*np.random.normal(size=(L,N))
  # X = np.zeros((N,M))
  # X[np.random.permutation(N)[:K]] = np.random.normal(size=(K,M))
  # noise = np.random.normal(size=(L,M))/5
  # Y = A@X + noise
  SNRs = [0,5,10,15]
  # SNR = 10
  Nsamp = 100
  
  # print('SNR=', 10*np.log10(np.linalg.norm(A@X)**2/np.linalg.norm(noise)**2))
  res = []
  # lams = np.logspace(0,1,5)
  lam = 3.1
  # for lam in lams:
  for SNR in SNRs:
    Yall, Xall, sigma = gen_c(A,Nsamp,N,L,M,K,SNR)
    res.append(0)
    for Y,X in zip(Yall,Xall):
      Xcvx = cp.Variable(shape=(N,M),complex=True)
      obj = cp.norm(Y-A@Xcvx)**2 + lam*cp.sum(cp.norm(Xcvx,p=2,axis=1))
      p = cp.Problem(cp.Minimize(obj))
      p.solve()
      temp = np.linalg.norm(Xcvx.value-X)**2/np.linalg.norm(X)**2
      res[-1] += temp
      print(temp)
    res[-1] = 10*np.log10(res[-1]/Nsamp)
  plt.plot(SNRs,res)
  plt.show()

def worker(inputs):
  E, p, Yall, lams1,lams2, ind = inputs
  i, j, nsamp = ind
  Y = Yall[nsamp]
  lam1, lam2 = lams1[ind[0]], lams2[ind[1]]

  Phi = cp.Constant(p.Phi.real) + 1j*cp.Constant(p.Phi.imag)
  Zcvx = cp.Variable(shape=(p.N,p.Ng),complex=True)
  Xcvx = cp.Variable(shape=(p.N,p.M),complex=True)
  obj = cp.norm(Y-p.A@Xcvx)**2 + lam1*cp.sum(cp.norm(Xcvx,p=2,axis=1)) + lam2*cp.norm(Zcvx, p=1)
  # set_trace()
  c = [Zcvx@Phi.T == Xcvx] # + [cp.imag(cp.matmul(Zcvx,p.Phi.T)) == cp.imag(Xcvx)]
  prob = cp.Problem(cp.Minimize(obj), c)
  prob.solve()
  E.append({'Xhat':Xcvx.value, 'Zhat':Zcvx.value, 'ind':ind})

def f2():
  M = 8
  N = 50
  L = 12
  K = 3
  P = 8
  channel_sparsity = 2
  # A = np.random.normal(size=(L,N)) + 1j*np.random.normal(size=(L,N))
  # Phi = np.random.normal(size=(P,M)) + 1j*np.random.normal(size=(P,M))
  # X = np.zeros((N,M))
  # X[np.random.permutation(N)[:K]] = np.random.normal(size=(K,M))
  # noise = np.random.normal(size=(L,M))/5
  # Y = A@X + noise
  # SNRs = [10,15]
  SNR = 10
  Nsamp = 1
  
  # print('SNR=', 10*np.log10(np.linalg.norm(A@X)**2/np.linalg.norm(noise)**2))
  res = []
  lams2 = np.logspace(-2,0,10)
  lams1 = np.logspace(-2,0,10)
  lam = 3.1
  # lam2 = 1
  
  p = problem(*(N,L,M,P,K,(M,1),channel_sparsity))
  Phi = cp.Constant(p.Phi.real) + 1j*cp.Constant(p.Phi.imag)
  # set_trace()
  for lam2 in lams:
    print('lam2 =', lam2)
  # for SNR in SNRs:
    # Yall, Xall, sigma = gen_c(A,Nsamp,N,L,M,K,SNR)
    Yall, Zall, sigma = gen_c_2(p,Nsamp,channel_sparsity,N,L,M,P,K,SNR)
    res.append(0)
    for Y,Z in zip(Yall,Zall):
      Zcvx = cp.Variable(shape=(N,P),complex=True)
      Xcvx = cp.Variable(shape=(N,M),complex=True)
      obj = cp.norm(Y-p.A@Xcvx)**2 + lam*cp.sum(cp.norm(Xcvx,p=2,axis=1)) + lam2*cp.norm(Zcvx, p=1)
      # set_trace()
      c = [Zcvx@Phi.T == Xcvx] # + [cp.imag(cp.matmul(Zcvx,p.Phi.T)) == cp.imag(Xcvx)]
      prob = cp.Problem(cp.Minimize(obj), c)
      prob.solve()
      temp = np.linalg.norm(Xcvx.value-Z@p.Phi.T)**2/np.linalg.norm(Z)**2
      res[-1] += temp
      print(temp)
    res[-1] = 10*np.log10(res[-1]/Nsamp)
  plt.plot(lams,res)
  plt.show()

def mp():
  M = 8
  N = 50
  L = 12
  K = 3
  P = 8
  channel_sparsity = 2
  # A = np.random.normal(size=(L,N)) + 1j*np.random.normal(size=(L,N))
  # Phi = np.random.normal(size=(P,M)) + 1j*np.random.normal(size=(P,M))
  # X = np.zeros((N,M))
  # X[np.random.permutation(N)[:K]] = np.random.normal(size=(K,M))
  # noise = np.random.normal(size=(L,M))/5
  # Y = A@X + noise
  # SNRs = [10,15]
  SNR = 10
  Nsamp = 5
  
  # print('SNR=', 10*np.log10(np.linalg.norm(A@X)**2/np.linalg.norm(noise)**2))
  res = []
  Nlam1 = 10
  Nlam2 = 10
  lams1 = np.logspace(-2,0, Nlam1)
  lams2 = np.logspace(-2,0, Nlam2)

  lam = 3.1
  # lam2 = 1
  
  p = problem(*(N,L,M,P,K,(M,1),channel_sparsity))

  ind = []
  for i in range(Nlam1):
    for j in range(Nlam2):
      for nsamp in range(Nsamp):
        ind.append((i,j,nsamp))

  manager = Manager()
  E = manager.list()

  Yall, Zall, sigma = gen_c_2(p,Nsamp,channel_sparsity,N,L,M,P,K,SNR)

  Nworker = Nlam1*Nlam2*Nsamp
  inputs = list(zip([E]*Nworker, [p]*Nworker, [Yall]*Nworker, [lams1]*Nworker, [lams2]*Nworker, ind))
  
  with Pool() as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(worker, inputs), total=len(inputs)):
        pass
    # pool.map(worker, inputs)

  NMSE = np.zeros((len(lams1),len(lams2),Nsamp))
  for e in E:
    i,j,nsamp = e['ind']
    NMSE[i,j,nsamp] = np.linalg.norm(e['Xhat']-Zall[nsamp]@p.Phi.T)**2/np.linalg.norm(Zall[nsamp]@p.Phi.T)**2
  NMSE = 10*np.log10(np.mean(NMSE, axis=-1))
  for nmse in NMSE:
    plt.plot(lams2, nmse)
  plt.legend([str(l) for l in lams2])
  plt.show()
  
if __name__ == '__main__':
  # f2()
  mp()
      # print(10*np.log10(np.linalg.norm(Xcvx.value-X)**2/np.linalg.norm(X)**2))
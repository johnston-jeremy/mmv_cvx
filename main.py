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
  lam1, lam2 = lams1[i], lams2[j]

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

def mp(L,M,K):
  # M = 8
  N = 50
  # L = 12
  # K = 3
  P = 2*M
  channel_sparsity = 2
  # A = np.random.normal(size=(L,N)) + 1j*np.random.normal(size=(L,N))
  # Phi = np.random.normal(size=(P,M)) + 1j*np.random.normal(size=(P,M))
  # X = np.zeros((N,M))
  # X[np.random.permutation(N)[:K]] = np.random.normal(size=(K,M))
  # noise = np.random.normal(size=(L,M))/5
  # Y = A@X + noise

  # SNRs = [10,15]
  SNR = 10
  # Nsamp = 10

  # Yall, Zall, sigma = gen_c_2(p,Nsamp,channel_sparsity,N,L,M,P,K,SNR)
  D = np.load('./testdata/data_L='+str(L)+'_M='+str(M)+'_K='+str(K)+'_SNR='+str(SNR)+'.npy', allow_pickle=True).item()
  # set_trace()
  Ytest, Xtest, p = D['Y'], D['X'], D['p']
  Yall = Ytest[0] + 1j*Ytest[1]
  Xall = Xtest[0] + 1j*Xtest[1]
  Nsamp = Xall.shape[0]
  Nsamp = 2
  
  # print('SNR=', 10*np.log10(np.linalg.norm(A@X)**2/np.linalg.norm(noise)**2))
  res = []
  # Nlam1 = 2
  # Nlam2 = 2
  # lams1 = np.logspace(-3,-1, Nlam1)
  # lams2 = np.logspace(-4,0, Nlam2)

  Nlam1 = 1
  Nlam2 = 1
  lams1 = [0.1]
  lams2 = [0.1]

  # p = problem(*(N,L,M,P,K,(M,1),channel_sparsity))

  ind = []
  for i in range(Nlam1):
    for j in range(Nlam2):
      for nsamp in range(Nsamp):
        ind.append((i,j,nsamp))

  manager = Manager()
  E = manager.list()

  

  Nworker = Nlam1*Nlam2*Nsamp
  inputs = list(zip([E]*Nworker, [p]*Nworker, [Yall]*Nworker, [lams1]*Nworker, [lams2]*Nworker, ind))
  
  with Pool() as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(worker, inputs), total=len(inputs)):
        pass
    # pool.map(worker, inputs)

  NMSE = np.zeros((Nlam1,Nlam2,Nsamp))
  for e in E:
    i,j,nsamp = e['ind']
    NMSE[i,j,nsamp] = np.linalg.norm(e['Xhat']-Xall[nsamp])**2/np.linalg.norm(Xall[nsamp])**2
    # NMSE[i,j,nsamp] = np.linalg.norm(e['Zhat']@p.Phi.T-Xall[nsamp])**2/np.linalg.norm(Xall[nsamp])**2
  NMSE = 10*np.log10(np.mean(NMSE, axis=-1))

  return NMSE, lams1, lams2, (L,M,K)

def plot_nmse(ax, NMSE, lams1, lams2, LMK):
  L,M,K = LMK
  for nmse in NMSE:
    ax.plot(lams2, nmse)
  ax.set_xscale('log')
  ax.legend([str(l) for l in lams1])
  ax.set_title('L, M, K = ' + str((L,M,K)))

def lam_tradeoff():
  M = 8
  K = 3
  figL, axL = plt.subplots(5,1)
  i = 0
  for L in [4,8,12,16,20]:
  # for L in [12]:
    NMSE, lams1, lams2, LMK = mp(L, M, K)
    plot_nmse(axL[i], NMSE, lams1, lams2, LMK)
    i += 1
  
  L = 12
  K = 3
  i = 0
  figM, axM = plt.subplots(4,1)
  for M in [4,8,12,16]:
    NMSE, lams1, lams2, LMK = mp(L, M, K)
    plot_nmse(axM[i], NMSE, lams1, lams2, LMK)
    i += 1

  M = 8
  L = 12
  i = 0
  figK, axK = plt.subplots(6,1)
  for K in [3,4,5,6,7,8]:
    NMSE, lams1, lams2, LMK = mp(L, M, K)
    plot_nmse(axK[i], NMSE, lams1, lams2, LMK)
    i += 1

  plt.show()
  
def LMK():
  M = 8
  K = 3
  figL, axL = plt.subplots()
  i = 0
  NMSE_L = []
  Llist = [4,8,12,16,20]
  for L in Llist:
  # for L in [12]:
    NMSE, lams1, lams2, LMK = mp(L, M, K)
    NMSE_L.append(NMSE[0,0])
  axL.plot(Llist, NMSE_L)
  
  L = 12
  K = 3
  i = 0
  figM, axM = plt.subplots()
  NMSE_M = []
  Mlist = [4,8,12,16]
  for M in Mlist:
    NMSE, lams1, lams2, LMK = mp(L, M, K)
    NMSE_M.append(NMSE[0,0])
  axM.plot(Mlist, NMSE_M)
  

  M = 8
  L = 12
  i = 0
  figK, axK = plt.subplots()
  Klist = [3,4,5,6,7,8]
  NMSE_K = []
  for K in Klist:
    NMSE, lams1, lams2, LMK = mp(L, M, K)
    NMSE_K.append(NMSE[0,0])
  axK.plot(Klist, NMSE_K)

  plt.show()

if __name__ == '__main__':
  # NMSE, lams1, lams2, LMK = mp(L=12,M=8,K=3)
  LMK()

  # fig, ax = plt.subplots()
  # plot_nmse(ax, NMSE, lams1, lams2, LMK)
  # plt.show()

  # print(NMSE)

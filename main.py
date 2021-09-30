from vamp import vamp
import tqdm
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from data import gen_c, gen_c_2
from problem import problem
from pdb import set_trace
from multiprocessing import Pool, Manager

# N,L,M,K,J,SNR
params_mmse = {}
# L
params_mmse[(50,4,8,3,2,10)] = 1 #0.05 # 50 iter
params_mmse[(50,8,8,3,2,10)] = 1 #0.03 # 50 iter
params_mmse[(50,12,8,3,2,10)] = 1 #0.3 # 50 iter
params_mmse[(50,16,8,3,2,10)] = 1 #0.3 # 50 iter
params_mmse[(50,20,8,3,2,10)] = 1 #0.3 # 15 iter
# M
params_mmse[(50,12,4,3,2,10)] = 1 #0.3 # 50 iter
params_mmse[(50,12,12,3,2,10)] = 1 #0.3 # 100 iter
params_mmse[(50,12,16,3,2,10)] = 1 #0.3 # 100 iter
# K
params_mmse[(50,12,8,4,2,10)] = 1 # 0.3 # 100 iter
params_mmse[(50,12,8,5,2,10)] = 1 #0.3 # 100 iter
params_mmse[(50,12,8,6,2,10)] = 1 #0.5 # 100 iter
params_mmse[(50,12,8,7,2,10)] = 1 # 0.5 # 100 iter
params_mmse[(50,12,8,8,2,10)] = 1 # 0.3 # 100 iter
# SNR
params_mmse[(50,12,8,3,2,0)] = 1 #0.3 # 100 iter
params_mmse[(50,12,8,3,2,5)] = 1 #0.3 # 100 iter
params_mmse[(50,12,8,3,2,15)] = 1 #0.3 # 150 iter
params_mmse[(50,12,8,3,2,20)] = 1 #0.15 # 250 iter

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

def worker_oracle(inputs):
  E, prob, Yall, Xall,lams1,lams2, ind = inputs
  i, j, nsamp = ind
  Y = Yall[nsamp]
  X = Xall[nsamp]

  I = np.where(np.linalg.norm(X, axis=1) > 0)[0]
  # print(I)
  Xhat = np.zeros((prob.N,prob.M), dtype=complex)
  Xhat[I] = np.linalg.pinv(prob.A[:,np.array(I)])@Y

  E.append({'Xhat':Xhat, 'ind':ind})

def worker_omp(inputs):
  E, prob, Yall, Xall,lams1,lams2, ind = inputs
  i, j, nsamp = ind
  Y = Yall[nsamp]
  A = prob.A
  Yk = Y
  I = []
  while(len(I) < prob.K):
    Z = A.T.conj() @ Yk
    k = np.argmax(np.linalg.norm(Z, axis=1))
    if k not in I:
      I.append(k)
    Yk = Yk - np.outer(A[:,k], Z[k])

  X = np.zeros((prob.N,prob.M), dtype=complex)
  X[I] = np.linalg.pinv(A[:,np.array(I)])@Y
  E.append({'Xhat':X, 'ind':ind})

def worker_vampmmse(inputs):
  E, prob, Yall,Xall, lams1,lams2, ind = inputs
  i, j, nsamp = ind
  Y = Yall[nsamp]
  X = vamp(Y, prob, onsager=1)
  E.append({'Xhat':X, 'ind':ind})

def worker_mfocuss(inputs):
  E, prob, Yall, Xall,lams1,lams2, ind = inputs
  i, j, nsamp = ind
  Y = Yall[nsamp]
  Xk = np.linalg.pinv(prob.A)@Y
  # Xk = np.random.normal(size=(prob.N,prob.M)) + 1j*np.random.normal(size=(prob.N,prob.M))
  p = 0.8
  tol = 0.01
  while(1):
    Xprev = Xk
    Wk = np.diag(np.linalg.norm(Xk, axis=1))**(1-p/2)
    Qk = np.linalg.pinv(prob.A@Wk)@Y
    Xk = Wk@Qk
    if np.linalg.norm(Xprev-Xk)/np.linalg.norm(Xk) < tol:
      break
  E.append({'Xhat':Xk, 'ind':ind})
  
def worker(inputs):
  E, p, Yall, Xall, lams1,lams2, ind = inputs
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

def worker3(inputs):
  E, p, Yall, lams1,lams2, ind = inputs
  i, j, nsamp = ind
  Y = Yall[nsamp]
  lam1, lam2 = lams1[i], lams2[j]

  Phi = cp.Constant(p.Phi.real) + 1j*cp.Constant(p.Phi.imag)
  Zcvx = cp.Variable(shape=(p.N,p.Ng),complex=True)
  Xcvx = cp.Variable(shape=(p.N,p.M),complex=True)
  # obj = lam1*cp.sum(cp.norm(Zcvx@Phi.T,p=2,axis=1)) + cp.norm(Zcvx, p=1)
  obj = cp.sum(cp.norm(Xcvx,p=2,axis=1)) + cp.norm(Zcvx, p=1)
  # set_trace()
  c = [Y == p.A@Xcvx] + [Zcvx@Phi.T == Xcvx] # + [cp.imag(cp.matmul(Zcvx,p.Phi.T)) == cp.imag(Xcvx)]
  # c = [cp.norm(Y - p.A@Zcvx@Phi.T)**2 <= lam2*cp.norm(Y)**2] # + [cp.imag(cp.matmul(Zcvx,p.Phi.T)) == cp.imag(Xcvx)]
  prob = cp.Problem(cp.Minimize(obj), c)
  prob.solve(solver='MOSEK')
  E.append({'Xhat':Zcvx.value@p.Phi.T, 'Zhat':Zcvx.value, 'ind':ind})

def worker2(inputs):
  E, p, Yall,Xall, _, lams2, ind = inputs
  i, j, nsamp = ind
  Y = Yall[nsamp]
  lam2 = lams2[j]

  Xcvx = cp.Variable(shape=(p.N,p.M),complex=True)
  obj = cp.norm(Y-p.A@Xcvx)**2 + lam2*cp.sum(cp.norm(Xcvx,p=2,axis=1))
  prob = cp.Problem(cp.Minimize(obj))
  prob.solve()
  E.append({'Xhat':Xcvx.value, 'ind':ind})

def worker4(inputs):
  E, p, Yall,Xall, _, lams2, ind = inputs
  i, _, nsamp = ind
  Y = Yall[nsamp]

  Xcvx = cp.Variable(shape=(p.N,p.M),complex=True)
  obj = cp.sum(cp.norm(Xcvx,p=2,axis=1))
  c = [Y == p.A@Xcvx]
  prob = cp.Problem(cp.Minimize(obj), c)
  prob.solve()
  E.append({'Xhat':Xcvx.value, 'ind':ind})

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

def mp(L,M,K,method):
  print(L,M,K,method)
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
  Nsamp = 12
  
  # print('SNR=', 10*np.log10(np.linalg.norm(A@X)**2/np.linalg.norm(noise)**2))
  res = []
  # Nlam1 = 1
  # lams1 = np.logspace(-2,0, Nlam1)
  # Nlam2 = 5
  # lams2 = np.logspace(-2,0, Nlam2)

  Nlam1 = 1
  lams1 = [0.1]
  Nlam2 = 1
  lams2 = [0.1]


  # p = problem(*(N,L,M,P,K,(M,1),channel_sparsity))

  ind = []
  for i in range(Nlam1):
    for j in range(Nlam2):
      for nsamp in range(Nsamp):
        ind.append((i,j,nsamp))

  
  if method == 'cvx':
    worker_handle = worker3
  elif method == 'mfocuss':
    worker_handle = worker_mfocuss
  elif method == 'vampmmse':
    ksi = 1
    omega = p.N/p.L
    epsilon = p.K/p.N
    p.maxiter = 500
    damp1 = 0.6
    damp2 = 0
    p.lam = 1
    p.damp = damp1, damp2
    p.denoiser = 'mmse'
    p.onsager = 1
    p.beta = params_mmse[(N,L,M,K,channel_sparsity,SNR)]
    p.params = omega, epsilon, p.beta, p.sigma_noise, ksi, p.maxiter, p.alpha
    worker_handle = worker_vampmmse
  elif method == 'omp':
    worker_handle = worker_omp
  elif method == 'oracle':
    worker_handle = worker_oracle

  manager = Manager()
  E = manager.list()
  Nworker = Nlam1*Nlam2*Nsamp
  inputs = list(zip([E]*Nworker, [p]*Nworker, [Yall]*Nworker, [Xall]*Nworker, [lams1]*Nworker, [lams2]*Nworker, ind))

  with Pool() as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(worker_handle, inputs), total=len(inputs)):
        pass

  NMSE = np.zeros((Nlam1,Nlam2,Nsamp))
  # NMSE2 = np.zeros((Nlam1,Nlam2,Nsamp))
  for e in E:
    i,j,nsamp = e['ind']
    NMSE[i,j,nsamp] = np.linalg.norm(e['Xhat']-Xall[nsamp])**2/np.linalg.norm(Xall[nsamp])**2
    # NMSE2[i,j,nsamp] = np.linalg.norm(e['Zhat']@p.Phi.T-Xall[nsamp])**2/np.linalg.norm(Xall[nsamp])**2
  NMSE = 10*np.log10(np.mean(NMSE, axis=-1))
  # NMSE2 = 10*np.log10(np.mean(NMSE2, axis=-1))

  return NMSE, lams1, lams2, (L,M,K)

def plot_nmse(ax, NMSE, lams1, lams2, LMK):
  L,M,K = LMK

  for nmse in NMSE:
    ax.plot(lams2, nmse)
  ax.set_xscale('log')
  ax.legend([str(l) for l in lams1])
  ax.set_title('L, M, K = ' + str((L,M,K)))

def lam_tradeoff(method, *args):
  if 'L' in args:
    M = 8
    K = 3
    figL, axL = plt.subplots(5,1)
    i = 0
    for L in [4,8,12,16,20]:
    # for L in [12]:
      NMSE, lams1, lams2, LMK = mp(L, M, K, method)
      plot_nmse(axL[i], NMSE, lams1, lams2, LMK)
      i += 1
      print('L =', L, NMSE)
  
  if 'M' in args:
    L = 12
    K = 3
    i = 0
    figM, axM = plt.subplots(4,1)
    for M in [4,8,12,16]:
      NMSE, lams1, lams2, LMK = mp(L, M, K, method)
      plot_nmse(axM[i], NMSE, lams1, lams2, LMK)
      i += 1
      print('M =', M, NMSE)

  if 'K' in args:
    M = 8
    L = 12
    i = 0
    figK, axK = plt.subplots(6,1)
    for K in [3,4,5,6,7,8]:
      NMSE, lams1, lams2, LMK = mp(L, M, K, method)
      plot_nmse(axK[i], NMSE, lams1, lams2, LMK)
      i += 1
      print('K =', K, NMSE)

  plt.show()
  
def lam_tradeoff_2():
  M = 8
  K = 3
  figL, axL = plt.subplots(5,1)
  i = 0
  for L in [4,8,12,16,20]:
  # for L in [12]:
    NMSE, lams1, lams2, LMK = mp(L, M, K)
    plot_nmse(axL[i], NMSE.T, [None], lams1, LMK)
    i += 1
  
  L = 12
  K = 3
  i = 0
  figM, axM = plt.subplots(4,1)
  for M in [4,8,12,16]:
    NMSE, lams1, lams2, LMK = mp(L, M, K)
    plot_nmse(axM[i], NMSE.T, [None], lams1, LMK)
    i += 1

  M = 8
  L = 12
  i = 0
  figK, axK = plt.subplots(6,1)
  for K in [3,4,5,6,7,8]:
    NMSE, lams1, lams2, LMK = mp(L, M, K)
    plot_nmse(axK[i], NMSE.T, [None], lams1, LMK)
    i += 1

  plt.show()
  
def LMK(method, *args):
  M = 8
  K = 3
  NMSE_L = []
  Llist = [4,8,12,16,20]
  if 'L' in args:
    for L in Llist:
    # for L in [12]:
      NMSE, lams1, lams2, LMK = mp(L, M, K, method)
      NMSE_L.append(NMSE[0,0])
  
  L = 12
  K = 3
  NMSE_M = []
  Mlist = [4,8,12,16]
  if 'M' in args:
    for M in Mlist:
      NMSE, lams1, lams2, LMK = mp(L, M, K, method)
      NMSE_M.append(NMSE[0,0])

  M = 8
  L = 12
  Klist = [3,4,5,6,7,8]
  NMSE_K = []
  if 'K' in args:
    for K in Klist:
      NMSE, lams1, lams2, LMK = mp(L, M, K, method)
      NMSE_K.append(NMSE[0,0])

  print('L', NMSE_L)
  print('M', NMSE_M)
  print('K', NMSE_K)
  
  if 'plot' in args:
    figK, axK = plt.subplots()
    figM, axM = plt.subplots()
    figL, axL = plt.subplots()
    axL.plot(Llist, NMSE_L)
    axM.plot(Mlist, NMSE_M)
    axK.plot(Klist, NMSE_K)
    plt.show()

if __name__ == '__main__':
  # lam_tradeoff('cvx','L', 'M', 'K')
  
  LMK('cvx','L', 'M', 'K')
  import sys
  sys.exit()

  # LMK('mfocuss')

  # NMSE, lams1, lams2, L_M_K = mp(12,8,3, 'oracle')
  # print(NMSE.squeeze())
  # set_trace()

  M = 8
  L = 8
  K = 3
  method = 'vampmmse'
  res = []
  betas = np.logspace(-1.5,-0.5,25)
  for params_mmse[(50,L,M,K,2,10)] in betas:
  # for epsilon in betas:
    print(params_mmse[(50,L,M,K,2,10)])
    # print(epsilon)
    NMSE, lams1, lams2, L_M_K = mp(L, M, K, method)
    res.append(NMSE.squeeze())
  plt.plot(betas, res)
  plt.xscale('log')
  plt.show()
  # LMK('vampmmse', 'K')
  
  # lam_tradeoff('L')

  # fig, ax = plt.subplots()
  # plot_nmse(ax, NMSE, lams1, lams2, LMK)
  # plt.show()

  # print(NMSE)

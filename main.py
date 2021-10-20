from detect import detect, detect_AP
from vamp import vamp
import tqdm
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from data import gen_c, gen_c_cell_free
from problem import problem
from pdb import set_trace
from multiprocessing import Pool, Manager
from workers import *

def get_method_params():
  # N,L,M
  params_ista = {}
  # L
  params_ista[(50,4,8,3,2,10)] = 2.5
  params_ista[(50,8,8,3,2,10)] = 1.5
  params_ista[(50,12,8,3,2,10)] = 0.25
  params_ista[(50,16,8,3,2,10)] = 0.15
  params_ista[(50,20,8,3,2,10)] = 0.15
  # M
  params_ista[(50,12,4,3,2,10)] = 0.18
  params_ista[(50,12,12,3,2,10)] = 0.12
  params_ista[(50,12,16,3,2,10)] = 0.11
  # K
  params_ista[(50,12,8,4,2,10)] = 0.25
  params_ista[(50,12,8,5,2,10)] = 0.25
  params_ista[(50,12,8,6,2,10)] = 0.16
  params_ista[(50,12,8,7,2,10)] = 0.16
  params_ista[(50,12,8,8,2,10)] = 0.19
  # SNR
  params_ista[(50,12,8,3,2,0)] = 0.55
  params_ista[(50,12,8,3,2,5)] = 0.35
  params_ista[(50,12,8,3,2,15)] = 0.1
  params_ista[(50,12,8,3,2,20)] = 0.1

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

  #  N,L,M,K,J,SNR
  params_admm3 = {}
  # L
  params_admm3[(50,4,8,3,2,10)] = 1.74e-01, 5.74e-02, 2.17e-01, 8.95e-02, 7.17e-02
  # params_admm3[(50,4,8,3,2,10)] = 4.32e-02, 5.27e-02, 7.37e-01, 1.25e-01, 2.89e-02 #
  params_admm3[(50,8,8,3,2,10)] = 1.20e+00, 5.74e-03, 2.23e+00, 1.56e-01, 4.14e-01 #
  # params_admm3[(50,12,8,3,2,10)] = 6.18e-01, 8.95e-03, 2.23e+00, 2.44e-01, 4.14e-01 #
  params_admm3[(50,12,8,3,2,10)] = 6.9e-01, 1.0e-02, 2.0e+00, 1.4e-01, 7.2e-01
  params_admm3[(50,16,8,3,2,10)] = 3.96e-01, 1.74e-02, 1.79e+00, 2.44e-01, 6.44e-01 # 
  params_admm3[(50,20,8,3,2,10)] = 1.20e+00, 5.74e-03, 1.15e+00, 1.56e-01, 6.44e-01 #
  # M
  params_admm3[(50,12,4,3,2,10)] = 7.71e-01, 1.74e-02, 1.15e+00, 2.44e-01, 4.14e-01 # 
  params_admm3[(50,12,12,3,2,10)] = 1.20e+00, 8.95e-03, 1.79e+00, 2.44e-01, 6.44e-01 #
  # params_admm3[(50,12,16,3,2,10)] = 3.96e-01, 1.12e-02, 1.79e+00, 1.56e-01, 4.14e-01 #
  params_admm3[(50,12,16,3,2,10)] = 1.20e+00, 5.14e-03, 1.79e+00, 2.18e-01, 3.70e-01
  
  # K
  # params_admm3[(50,12,8,4,2,10)] = 1.20e+00, 1.12e-02, 1.15e+00, 1.56e-01, 6.44e-01 # 
  # # params_admm3[(50,12,8,5,2,10)] = 7.71e-01, 8.95e-03, 3.48e+00, 2.44e-01, 6.44e-01 #
  # params_admm3[(50,12,8,5,2,10)] = 1.20e+00, 1.12e-02, 1.15e+00, 1.56e-01, 6.44e-01
  # params_admm3[(50,12,8,6,2,10)] = 7.71e-01, 1.12e-02, 1.79e+00, 2.44e-01, 4.14e-01 # 
  # params_admm3[(50,12,8,7,2,10)] = 1.20e+00, 1.74e-02, 1.15e+00, 2.44e-01, 6.44e-01 # 
  # params_admm3[(50,12,8,8,2,10)] = 7.71e-01, 8.95e-03, 3.48e+00, 2.44e-01, 4.14e-01 # 

  params_admm3[(50,12,8,4,2,10)] = 6.9e-01, 1.0e-02, 2.0e+00, 1.4e-01, 7.2e-01 # 
  params_admm3[(50,12,8,5,2,10)] = 6.9e-01, 1.0e-02, 2.0e+00, 1.4e-01, 7.2e-01
  params_admm3[(50,12,8,6,2,10)] = 6.9e-01, 1.0e-02, 2.0e+00, 1.4e-01, 7.2e-01 # 
  params_admm3[(50,12,8,7,2,10)] = 6.9e-01, 1.0e-02, 2.0e+00, 1.4e-01, 7.2e-01 # 
  params_admm3[(50,12,8,8,2,10)] = 6.9e-01, 1.0e-02, 2.0e+00, 1.4e-01, 7.2e-01 # 

  
  # SNR
  params_admm3[(50,12,8,3,2,0)] = 3.96e-01, 1.74e-02, 1.79e+00, 8.04e-02, 3.31e-01 # 
  params_admm3[(50,12,8,3,2,5)] = 7.71e-01, 1.74e-02, 1.15e+00, 2.44e-01, 4.14e-01 # 
  params_admm3[(50,12,8,3,2,15)] = 3.96e-01, 8.95e-03, 3.48e+00, 2.44e-01, 6.44e-01 # 
  params_admm3[(50,12,8,3,2,20)] = 1.20e+00, 8.95e-03, 1.15e+00, 1.56e-01, 6.44e-01 #

  params_admm3[(50,12,8,3,2,0)] = 6.90e-01, 1.00e-02, 1.17e+00, 1.40e-01, 4.21e-01  # 
  params_admm3[(50,12,8,3,2,5)] = 6.90e-01, 1.00e-02, 1.17e+00, 1.40e-01, 4.21e-01  # 
  params_admm3[(50,12,8,3,2,15)] = 6.90e-01, 1.00e-02, 1.17e+00, 1.40e-01, 4.21e-01  # 
  params_admm3[(50,12,8,3,2,20)] = 6.90e-01, 1.00e-02, 1.17e+00, 1.40e-01, 4.21e-01  #
  

  #  N,L,M,K,J,SNR
  params_admm1 = {}
  # # L
  # params_admm1[(50,4,8,3,2,10)] = 0.822, 0.072, 0.057 # 50 iter
  # params_admm1[(50,8,8,3,2,10)] = 0.057, 0.024, 0.339 # 50 iter
  # params_admm1[(50,12,8,3,2,10)] = 0.217, 0.057, 0.008 # 50 iter
  # params_admm1[(50,16,8,3,2,10)] = 0.217, 0.112, 0.006 # 50 iter
  # params_admm1[(50,20,8,3,2,10)] = 0.072, 0.03, 0.527 # 15 iter
  # # M
  # params_admm1[(50,12,4,3,2,10)] = 0.217, 0.057, 0.008 # 50 iter
  # params_admm1[(50,12,12,3,2,10)] = 0.217, 0.057, 0.008 # 100 iter
  # params_admm1[(50,12,16,3,2,10)] = 0.217, 0.057, 0.008 # 100 iter
  
  # # K
  # params_admm1[(50,12,8,4,2,10)] = 0.174, 0.057, 0.008 # 100 iter
  # params_admm1[(50,12,8,5,2,10)] = 0.09, 0.03, 0.339 # 100 iter
  # params_admm1[(50,12,8,6,2,10)] = 0.217, 0.112, 0.015 # 100 iter
  # params_admm1[(50,12,8,7,2,10)] = 0.174, 0.015, 0.339 # 100 iter
  # params_admm1[(50,12,8,8,2,10)] = 0.174, 0.024, 0.217 # 100 iter

  # params_admm1[(50,12,8,4,2,10)] = 0.217, 0.057, 0.1 # 100 iter
  # params_admm1[(50,12,8,5,2,10)] = 0.217, 0.057, 0.1 # 100 iter
  # params_admm1[(50,12,8,6,2,10)] = 0.217, 0.057, 0.1 # 100 iter
  # params_admm1[(50,12,8,7,2,10)] = 0.217, 0.057, 0.1 # 100 iter
  # params_admm1[(50,12,8,8,2,10)] = 0.217, 0.057, 0.1 # 100 iter

  # # SNR
  # params_admm1[(50,12,8,3,2,0)] = 0.822, 0.057, 0.006 # 100 iter
  # params_admm1[(50,12,8,3,2,5)] = 0.217, 0.015, 0.339 # 100 iter
  # params_admm1[(50,12,8,3,2,15)] = 0.003, 0.03, 0.217 # 150 iter
  # params_admm1[(50,12,8,3,2,20)] = 0.01, 0.046, 0.112 # 250 iter


  params_admm1[( 50, 12, 4, 3, 2, 10, )] = 0.15677120649209492, 0.22815461799515804, 0.011337715629354008, 0.3320413925317422,
  params_admm1[( 50, 12, 8, 3, 2, 10, )] = 0.22815461799515804, 0.22815461799515804, 0.007790450939353023, 0.3320413925317422,
  params_admm1[( 50, 12, 12, 3, 2, 10, )] = 0.22815461799515804, 0.22815461799515804, 0.01133771562935402, 0.4832314477051623,
  params_admm1[( 50, 12, 16, 3, 2, 10, )] = 0.22815461799515804, 0.4832314477051623, 0.007790450939353031, 0.07401851617833326,
  params_admm1[( 50, 12, 8, 4, 2, 10, )] = 0.22815461799515804, 0.050860123656485505, 0.005353029465771182, 1.4895122470273374,
  params_admm1[( 50, 12, 8, 5, 2, 10, )] = 0.22815461799515804, 0.7032636210526189, 0.007790450939353023, 0.0036782112722982064,
  params_admm1[( 50, 12, 8, 6, 2, 10, )] = 0.22815461799515804, 0.3320413925317422, 0.07401851617833326, 0.034947366036367444,
  params_admm1[( 50, 12, 8, 7, 2, 10, )] = 0.33204139253174225, 0.15677120649209492, 0.10772173450159421, 2.167739250873008,
  params_admm1[( 50, 12, 8, 8, 2, 10, )] = 0.22815461799515804, 0.07401851617833326, 0.007790450939353023, 1.023484135903761,
  params_admm1[( 50, 4, 8, 3, 2, 10, )] = 0.00012559432157547895, 0.03494736603636743, 0.011337715629354008, 2.167739250873008,
  params_admm1[( 50, 8, 8, 3, 2, 10, )] = 0.4832314477051623, 0.22815461799515804, 0.034947366036367444, 0.7032636210526189,
  params_admm1[( 50, 16, 8, 3, 2, 10, )] = 0.15677120649209492, 0.7032636210526189, 0.01133771562935402, 0.050860123656485505,
  params_admm1[( 50, 20, 8, 3, 2, 10, )] = 0.15677120649209492, 0.15677120649209492, 0.0025273984105956223, 0.4832314477051623,
  

  

  return {'vampista':params_ista, 'vampmmse':params_mmse, 'admm3':params_admm3, 'admm1':params_admm1}

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

def mp_lam(L,M,K,method):
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
  Nsamp = 10
  p = problem(*(N,L,M,P,K,(M,1),channel_sparsity))

  Yall, Xall, Zall, sigma = gen_c_2(p,Nsamp,channel_sparsity,N,L,M,P,K,SNR)
  # D = np.load('./testdata/data_L='+str(L)+'_M='+str(M)+'_K='+str(K)+'_SNR='+str(SNR)+'.npy', allow_pickle=True).item()
  # set_trace()
  # Yall, Xall, p = D['Y'], D['X'], D['p']
  
  # Ytest, Xtest, p = D['Y'], D['X'], D['p']
  # Yall = Ytest[0] + 1j*Ytest[1]
  # Xall = Xtest[0] + 1j*Xtest[1]
  
  # Nsamp = Xall.shape[0]
  # Nsamp = 12
  # set_trace()
  
  # print('SNR=', 10*np.log10(np.linalg.norm(A@X)**2/np.linalg.norm(noise)**2))
  res = []
  # Nlam1 = 5
  # lams1 = np.logspace(-3,0, Nlam1)
  # Nlam2 = 5
  # lams2 = np.logspace(-3,0, Nlam2)

  Nlam1 = 1
  lams1 = [0.1]
  Nlam2 = 1
  lams2 = [0.1]



  ind = []
  for i in range(Nlam1):
    for j in range(Nlam2):
      for nsamp in range(Nsamp):
        ind.append((i,j,nsamp))

  
  if method == 'cvx':
    worker_handle = worker
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
    p.params = omega, epsilon, p.beta, None, ksi, p.maxiter, p.alpha
    worker_handle = worker_vampmmse
  elif method == 'omp':
    worker_handle = worker_omp
  elif method == 'oracle':
    Yall,Xall, Zall,_ = gen_c_2(p, Nsamp,channel_sparsity,N,L,M,P,K,SNR)
    np.save('./testdata/data_L='+str(L)+'_M='+str(M)+'_K='+str(K)+'_SNR='+str(SNR)+'.npy',{'Y':Yall,'X':Xall, 'Z':Zall, 'p':p})
    worker_handle = worker_oracle

  manager = Manager()
  E = manager.list()
  Nworker = Nlam1*Nlam2*Nsamp
  if method == 'oracle':
    inputs = list(zip([E]*Nworker, [p]*Nworker, [Yall]*Nworker, [Xall]*Nworker, [Zall]*Nworker, [lams1]*Nworker, [lams2]*Nworker, ind))
  else:
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

def vamp_setup(p, mode):
  if mode == 'vampista':
    # p.beta = 1/p.M # channel variance
    p.beta = 1 # channel variance
    ksi = 1
    omega = p.N/p.L
    epsilon = p.K/p.N
    p.maxiter_vamp = 1000

    damp1 = 0.6
    damp2 = 0
    p.damp = damp1, damp2
    p.lam = get_method_params()[mode][(p.N,p.L,p.M,p.K,p.J,p.SNR)]
    p.damp_init = 0.7
    p.maxiter_ista = 20
    p.onsager = 0
    p.istawarmstart = True
    p.denoiser = 'ista'
  
  elif mode == 'vampmmse':
    p.epsilon = p.K/p.N
    p.maxiter_vamp = 1000
    damp1 = 0.6
    damp2 = 0
    p.lam = 1
    p.damp = damp1, damp2
    p.denoiser = 'mmse'
    p.beta = get_method_params()[mode][(p.N,p.L,p.M,p.K,p.J,p.SNR)]

def admm3_setup(p):
  p.sigma, p.mu, p.rho, p.taux, p.tauz = get_method_params()['admm3'][(p.N,p.L,p.M,p.K,p.J,p.SNR)]
  # p.sigma, p.mu, p.rho, p.taux, p.tauz = 6.9e-01, 1.0e-02, 2.0e+00, 1.4e-01, 7.2e-01
  p.maxiter = 1000

def admm1_setup(p):
  p.mu, p.beta, p.taux, p.gamma = get_method_params()['admm1'][(p.N,p.L,p.M,p.K,p.J,p.SNR)]
  p.mu, p.beta, p.taux, p.gamma = 0.22815461799515804, 0.22815461799515804, 0.007790450939353023, 0.3320413925317422,
  p.maxiter = 1000

def mp_samples(method, Yall, Xall, Zall, p):
  print(method)

  p.tol = 1e-6

  # D = np.load('./testdata/data_L='+str(L)+'_M='+str(M)+'_K='+str(K)+'_SNR='+str(SNR)+'.npy', allow_pickle=True).item()
  # set_trace()
  # Yall, Xall, p = D['Y'], D['X'], D['p']
  
  # Ytest, Xtest, p = D['Y'], D['X'], D['p']
  # Yall = Ytest[0] + 1j*Ytest[1]
  # Xall = Xtest[0] + 1j*Xtest[1]

  Nsamp = Xall.shape[0]
  ind = range(Nsamp)

  
  if method == 'cvx':
    worker_handle = worker
  elif method == 'mfocuss':
    worker_handle = worker_mfocuss
  elif method == 'vampmmse':
    vamp_setup(p,method)
    worker_handle = worker_vamp
  elif method == 'vampista':
    vamp_setup(p,method)
    worker_handle = worker_vamp
  elif method == 'admm3':
    admm3_setup(p)
    worker_handle = worker_admm3
  elif method == 'admm1':
    admm1_setup(p)
    worker_handle = worker_admm1
  elif method == 'omp':
    worker_handle = worker_omp
  elif method == 'oracle':
    worker_handle = worker_oracle

  manager = Manager()
  E = manager.list()
  Nworker = len(ind)
  if method == 'oracle':
    inputs = list(zip([E]*Nworker, [p]*Nworker, [Yall]*Nworker, [Xall]*Nworker, [Zall]*Nworker, ind))
  else:
    inputs = list(zip([E]*Nworker, [p]*Nworker, Yall, ind))

  with Pool() as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(worker_handle, inputs), total=len(inputs)):
        pass

  NMSE = np.zeros((Nsamp,))
  for e in E:
    nsamp = e['ind']
    NMSE[nsamp] = np.linalg.norm(e['Xhat']-Xall[nsamp])**2/np.linalg.norm(Xall[nsamp])**2
  # print(NMSE)
  NMSE = 10*np.log10(np.mean(NMSE))

  return NMSE, E

def mp_jobs(L,M,K,method):
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
  return inputs, E

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
  
def LMK(method, data, *args):
  
  Llist = data['Llist']
  Mlist = data['Mlist']
  Klist = data['Klist']
  
  M = 8
  K = 3
  NMSE_L = []
  for L in Llist:
    Yall, Xall, Zall, p = data[(L,M,K)]
    NMSE, _ = mp_samples(method, Yall, Xall, Zall, p)
    print(NMSE)
    NMSE_L.append(NMSE)
  
  L = 12
  K = 3
  NMSE_M = []
 
  for M in Mlist:
    Yall, Xall, Zall, p = data[(L,M,K)]
    NMSE, _ = mp_samples(method, Yall, Xall, Zall, p)
    print(NMSE)
    NMSE_M.append(NMSE)

  M = 8
  L = 12
  NMSE_K = []
  for K in Klist:
    Yall, Xall, Zall, p = data[(L,M,K)]
    NMSE, _ = mp_samples(method, Yall, Xall, Zall, p)
    print(NMSE)
    NMSE_K.append(NMSE)

  if len(NMSE_L) > 0: print('L', NMSE_L)
  if len(NMSE_M) > 0: print('M', NMSE_M)
  if len(NMSE_K) > 0: print('K', NMSE_K)
  
  if 'plot' in args:
    if len(NMSE_L) > 0: 
      figL, axL = plt.subplots()
      axL.plot(Llist, NMSE_L)
    if len(NMSE_M) > 0: 
      figM, axM = plt.subplots()
      axM.plot(Mlist, NMSE_M)
    if len(NMSE_K) > 0: 
      figK, axK = plt.subplots()
      axK.plot(Klist, NMSE_K)

    plt.show()
  return NMSE_L, NMSE_M, NMSE_K

def LMK_jobs(method, *args):
  M = 8
  K = 3
  NMSE_L = []
  Llist = [4,8,12,16,20]
  EL = []
  inputsL = []
  if 'L' in args:
    for L in Llist:
    # for L in [12]:
      inputs, E = mp_jobs(L, M, K, method)
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

def generate_data(Nsamp,L,M,K,mode,*args):
  N = 50
  P = 2*M
  J = 2
  SNR = 10
  p = problem(*(N,L,M,P,K,(M,1),J,SNR))


  if 'cellfree' in args:
    Na = 9  
    Yall, Xall, Zall, sigma = gen_c_cell_free(p, Nsamp, Na, mode)
  else:
    Yall, Xall, Zall, sigma = gen_c(p, Nsamp, mode)

  return Yall, Xall, Zall, p

def lams_experiment():
  # LMK('mfocuss')

  NMSE, lams1, lams2, L_M_K = mp(12,8,3, 'cvx')
  print(lams2)
  print(NMSE.squeeze())
  import sys
  sys.exit()

  M = 4
  L = 12
  K = 3
  method = 'vampmmse'
  res = []
  betas = np.logspace(-1,0.5,25)
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

def roc():
  M = 8
  K = 8
  L = 12
  Nsamp = 10
  Yall, Xall, Zall, p = generate_data(Nsamp,L,M,K,'mmwave','cellfree')
  # methods = ['admm1','vampmmse']
  methods = ['admm1']

  Dpfa = {}
  Dpmd = {}
  for method in methods:
    Pfa = 0
    Pmd = 0
    for nsamp in range(Nsamp):
      Y,X,Z = Yall[nsamp], Xall[nsamp], Zall[nsamp]
      _, E = mp_samples(method, Y, X, Z, p)
     
      pfa, pmd, tt = detect_AP([e['Xhat'] for e in E], [X[e['ind']] for e in E])
      Pfa += pfa/Nsamp
      Pmd += pmd/Nsamp
    
    Dpfa[method] = Pfa
    Dpmd[method] = Pmd
  for method in methods:
    print(method)
    print('[' + ','.join('{:.2e}'.format(p) for p in Dpfa[method]) + ']')
    print('[' + ','.join('{:.2e}'.format(p) for p in Dpmd[method]) + ']')
    # [print(p), print(',') for p in Pfa]
    # [print(p), print(',') for p in Pmd]
      # print(tt)
      # plt.plot(np.log10(Pfa),np.log10(Pmd))
      # plt.show()
    # set_trace()

def main():
  Nsamp = 100

  data = {}
  # lam_tradeoff('cvx','L', 'M', 'K')
  Llist = [4,8,12,16,20]
  # Llist = []
  Mlist = [4,8,12,16]
  # Mlist = []
  Klist = [3,4,5,6,7,8]
  # Klist = [5,6,7,8]
  # Klist = [7]
  for L in Llist:
    M = 8
    K = 3
    data[(L,M,K)] = generate_data(Nsamp,L,M,K,'mmwave')

  for M in Mlist:
    L = 12
    K = 3
    data[(L,M,K)] = generate_data(Nsamp,L,M,K,'mmwave')

  for K in Klist:
    M = 8
    L = 12
    data[(L,M,K)] = generate_data(Nsamp,L,M,K,'mmwave')


  data['Llist'] = Llist
  data['Mlist'] = Mlist
  data['Klist'] = Klist

  # methods = ['admm1','admm3','vampmmse', 'vampista']
  # methods = ['vampista']
  methods = ['admm1','vampmmse']
  NMSE_L, NMSE_M, NMSE_K = {'var':'L'}, {'var':'M'}, {'var':'K'}
  for method in methods:
    NMSE_L[method], NMSE_M[method], NMSE_K[method] = LMK(method, data)

  for n in [NMSE_L, NMSE_M, NMSE_K]:
    print(n['var'])
    for method in methods:
      print(method, n[method])
    

# def plotroc():
#   pfa = 
if __name__ == '__main__':
  # main()
  roc()
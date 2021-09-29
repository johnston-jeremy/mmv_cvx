import numpy as np
from vamp import vamp, vamp_ista
from multiprocessing import Pool, Manager
import matplotlib.pyplot as plt
from pdb import set_trace
import tqdm

def vamp_worker(inputs):
  Y, X, p, onsager, E, P = inputs
  e, PfaPmd, err_ista_temp = vamp(Y, X, p, onsager)
  E.append(e)
  P.append(PfaPmd)

def vamp_worker_nmse(inputs):
  Y, X, p, onsager, E, P = inputs
  e, err_ista_temp = vamp(Y, X, p, onsager)
  E.append(e)

def vamp_worker_2(inputs):
  Y, Xhat, p, idx = inputs
  # set_trace()
  xhat = vamp_ista(Y, p)
  Xhat.append((idx, xhat))

def amp_multiprocessing(p,Y,X):
  # p.sigma_noise = sigma_noise

  from time import time
  tt = time()
  Nsamp = Y.shape[0]

  # print('lam =', p.lam)
  
  err_vamp_ista = 0
  err_vamp_mmse = 0
  err_ista = 0

  Pfa = []
  Pmd = []

  manager = Manager()
  E = manager.list()
  P = manager.list()
  inputs = zip(Y, X, [p]*Nsamp, [p.onsager]*Nsamp, [E]*Nsamp, [P]*Nsamp)
  with Pool() as pool:
    pool.map(vamp_worker, inputs)
  
  err_vamp_ista = np.sum(E, axis=0)

  
  PfaPmd = np.squeeze(np.mean(np.array(P),axis=0))

  # plt.plot([e[-1] for e in E])


  # for i in range(Nsamp):
  #   # p.denoiser = 'ista'
  #   # e, g, err_ista_temp = amp_ista_distributed(Y[i], X[i], p)
  #   e, g, err_ista_temp = vamp(Y[i], X[i], p, onsager=p.onsager)
  #   Pfa.append(g[0])
  #   Pmd.append(g[1])
  #   err_vamp_ista = err_vamp_ista + e
  #   err_ista = err_ista + err_ista_temp

  # Pfa = np.mean(Pfa, axis=0)
  # Pmd = np.mean(Pmd, axis=0)
  # print(Pfa)
  # print(1-Pmd)
  # plt.plot(np.linspace(0,1,10),Pfa)
  # plt.plot(np.linspace(0,1,10),Pmd)
  # plt.figure()
  # plt.scatter(Pfa,Pmd)
  
  # set_trace()
  # print('amp_ista time =', (time() - tt)/Nsamp)
  print('err_vamp_ista',        \
        'error =',              \
        10*np.log10(err_vamp_ista[-1]/Nsamp),
        'Pfa,Pmd:', PfaPmd)
  # plt.figure()

  # set_trace()
  # plt.figure()
  # plt.plot(10*np.log10(err_vamp_ista/Nsamp))
  # [(plt.figure(), plt.plot(10*np.log10(e))) for e in E]
  # plt.figure()
  # plt.plot(10*np.log10(np.sum(err_ista,axis=0)/Nsamp))
  # plt.title('amp-ista' + str(p.onsager) + 'error =' + str(np.round(10*np.log10(err_vamp_ista[-1]/Nsamp),2)))
  # plt.show()
  return np.round(10*np.log10(err_vamp_ista[-1]/Nsamp),2), PfaPmd

def amp_multiprocessing_nmse(p,Y,X):
  # p.sigma_noise = sigma_noise

  from time import time
  tt = time()
  Nsamp = Y.shape[0]

  # print('lam =', p.lam)
  
  err_vamp_ista = 0
  err_vamp_mmse = 0
  err_ista = 0


  manager = Manager()
  E = manager.list()
  P = manager.list()
  inputs = zip(Y, X, [p]*Nsamp, [p.onsager]*Nsamp, [E]*Nsamp, [P]*Nsamp)
  with Pool() as pool:
    pool.map(vamp_worker, inputs)
  
  err_vamp_ista = np.sum(E, axis=0)

  
  return np.round(10*np.log10(err_vamp_ista[-1]/Nsamp),2)

def amp_multiprocessing_2(p,Y,X):
  # stores Xhat
  # p.sigma_noise = sigma_noise

  Nsamp = Y.shape[0]

  # print('lam =', p.lam)
  
  err_vamp_ista = 0
  err_vamp_mmse = 0
  err_ista = 0

  Pfa = []
  Pmd = []

  manager = Manager()
  Xhat_man = manager.list()
  inputs = zip(Y, [Xhat_man]*Nsamp, [p]*Nsamp, range(Nsamp))
  with Pool() as pool:
  #   pool.map(vamp_worker_2, inputs)
  
    for _ in tqdm.tqdm(pool.imap_unordered(vamp_worker_2, inputs), total=Nsamp):
      pass

  err = 0
  
  Xhat = [None]*Nsamp

  for x in Xhat_man:
    Xhat[x[0]] = x[1]

  for x,xhat in zip(X,Xhat):
    err += np.sum(np.abs(x-xhat)**2)/np.sum(np.abs(x)**2)

  print('err_vamp_ista',        \
        'Average NMSE =',              \
        10*np.log10(err/Nsamp))

  return Xhat

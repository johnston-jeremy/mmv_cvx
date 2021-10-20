from multiprocessing import set_start_method
from readnets import get_numlayers,get_path,get_final_epoch, parse
from ampistanet_reduced import AMPNet
import admmnet3
import tensorflow as tf
from pdb import set_trace
import numpy as np

def get_params_list():
  params_list = [ (50,12,4,2*4,3,(4,1),2, 10),
                  (50,12,8,2*8,3,(8,1),2, 10),
                  (50,12,12,2*12,3,(12,1),2, 10),
                  (50,12,16,2*16,3,(16,1),2, 10),
                  (50,12,8,2*8,4,(8,1),2, 10),
                  (50,12,8,2*8,5,(8,1),2, 10),
                  (50,12,8,2*8,6,(8,1),2, 10),
                  (50,12,8,2*8,7,(8,1),2, 10),
                  (50,12,8,2*8,8,(8,1),2, 10),
                  (50, 4,8,2*8,3,(8,1),2, 10),
                  (50, 8,8,2*8,3,(8,1),2, 10),
                  (50, 16,8,2*8,3,(8,1),2, 10),
                  (50, 20,8,2*8,3,(8,1),2, 10)]
  return params_list                  
def gen_net(method, p, n):
  if method == 'admm3':
    print('ADMM3 Net')
    p.sigma, p.mu, p.rho, p.taux, p.tauz = 6.9e-01, 1.0e-02, 2.0e+00, 1.4e-01, 7.2e-01
    tf.keras.backend.set_floatx('float32')
    return admmnet3.ADMMNet(p, n)
  elif method == 'ampista':
    print('AMP-ISTA Net')
    p.lam = .1
    p.damp_init = 0.6
    tf.keras.backend.set_floatx('float32')
    num_stages = {'amp':n, 'ista':5}
    return AMPNet(p, num_stages)

def eval_via_nets():

  Ntest = 1000
  Nsamp = 1

  netlist = []
  for params in get_params_list():
    netlist.append([])
    netlist[-1].append(params)
    netlist[-1].append(get_path(params))

  err = []
  pfapmd = []
  method_params = get_method_params()
  # rootpath = '/Users/jeremyjohnston/Documents/mmv/nets/results/'
  rootpath = '/Users/Jeremy/Documents/GitHub/mmv/nets/results/'
  # rootpath = '/Users/Jeremy/Documents/mmv/dunn/nets/results/'
  # rootpath = '/Users/Jeremy/Documents/mmv/isit_results/'

  from time import time
  tt = 0
  for i in netlist:
    N,L,M,Ng,K,Ntxrx,J,SNR = i[0]
    p = problem(*(N,L,M,Ng,K,Ntxrx,J))
    
    for s in i[1]:
      if parse(s)[0] != 'ampmmse':
        break

    finalepoch = get_final_epoch(rootpath + s)
    p.A = np.load(rootpath + s + '/' + finalepoch + '/A.npy')
    A = np.copy(p.A)
    _, _, Ytest, Xtest = gen_data(p, Nsamp, Ntest, SNR)
    Ytestc = Ytest[0] + 1j*Ytest[1]
    Xtestc = Xtest[0] + 1j*Xtest[1]
    np.save('./testdata/data_L='+str(L)+'_M='+str(M)+'_K='+str(K)+'_SNR='+str(SNR)+'.npy',{'Y':Ytest,'X':Xtest, 'p':p})

    err[-1]['N'] = N
    err[-1]['L'] = L
    err[-1]['M'] = M
    err[-1]['K'] = K
    err[-1]['SNR'] = SNR

    pfapmd[-1]['N'] = N
    pfapmd[-1]['L'] = L
    pfapmd[-1]['M'] = M
    pfapmd[-1]['K'] = K
    pfapmd[-1]['SNR'] = SNR

    for s in i[1]:
      s_parsed = parse(s)
      method = s_parsed[0]
      finalepoch = get_final_epoch(rootpath + s)

      n = int(get_numlayers(s_parsed[5]))
      net = gen_net(method, p, n)
      net = load(net, rootpath + s + '/' + finalepoch + '/weights')

      Pfa = []
      Pmd = []

      if method == 'ampmmse':
        p.A = np.load(rootpath + s + '/' + finalepoch + '/A.npy')
        _, _, Ytemp, Xtemp = gen_data(p, Nsamp, Ntest, SNR)
        Xhat = net(Ytemp)
        err[-1][method+'_net'] = nmse(Xtemp, Xhat)
        Xtempc = Xtemp[0] + 1j*Xtemp[1]
      else:
        Xhat = net(Ytest)
        err[-1][method+'_net'] = nmse(Xtest, Xhat)
      
      
      Xtestc = Xtest[0] + 1j*Xtest[1]
      Xhatc = Xhat[0].numpy() + 1j*Xhat[1].numpy()
      if method == 'ampmmse':
        for x,xhat in zip(Xtempc,Xhatc):
          # PfaPmd.append(detect(x,xhat))
          fa, md = detect(xhat,x)
          
          Pfa.append(fa)
          Pmd.append(md)

          # import matplotlib.pyplot as plt
          # plt.plot(np.linalg.norm(xhat,axis=1),'x')
          # plt.plot(np.linalg.norm(x,axis=1),'o')
          # plt.show()
          # set_trace()
      else:
        for x,xhat in zip(Xtestc,Xhatc):
          # PfaPmd.append(detect(x,xhat))
          fa, md = detect(xhat,x)
          
          Pfa.append(fa)
          Pmd.append(md)

      # set_trace()
      pfapmd[-1][method+'_net'] = np.mean(np.array(Pfa),axis=0).tolist(),np.mean(np.array(Pmd),axis=0).tolist()
    
    p.A = A

    method = 'ampista'
    vamp_setup_ista(p)
    p.lam = method_params[method][(N,L,M,K,J,SNR)]
    # amp_param_test(p,Ytestc,Xtestc)
    err[-1][method], pfapmd[-1][method] = amp_multiprocessing(p,Ytestc,Xtestc)

    
    
    method = 'admm3'
    # p.sigma, p.mu, p.rho, p.taux, p.tauz = 6.9e-01, 1.0e-02, 2.0e+00, 1.4e-01, 7.2e-01
    p.sigma, p.mu, p.rho, p.taux, p.tauz = method_params[method][(N,L,M,K,J,SNR)]
    maxiter = 500
    params = p.N, p.L, p.M, p.Ng, p.sigma, p.mu, p.rho, p.taux, p.tauz, maxiter
    # e = admm3.admm_multiprocess(p, Ytestc, params, Xtestc)
    # err[-1][method] = 10*np.log10(e[-1]/Ntest)
    Xhat, e = admm_problem3(p.A, p.Phi, Ytestc, params, Xtestc)
    err[-1][method] = 10*np.log10(e[-1])
    pfa_admm3 = 0
    pmd_admm3 = 0
    for xhat,x in zip(Xhat,Xtestc):
      u,v = detect(xhat,x)
      pfa_admm3 += np.array(u)/Ntest
      pmd_admm3 += np.array(v)/Ntest
    pfapmd[-1][method] = pfa_admm3, pmd_admm3
    
    # set_trace()
  # set_trace()
  for d in err:
    print(d['N'],d['L'],d['M'],d['K'],d['SNR'],np.round(d['ampista'],2),np.round(d['admm3'],2),np.round(d['ampista_net'],2),np.round(d['admm3_net'],2))

  # [print(e) for e in err]
  # [print(e) for e in pfapmd]
  # print(pfapmd)
  
  # np.save('./errors', err)
  # np.save('./pfapmd', pfapmd)

def extract_A():
  rootpath = '/Users/jeremyjohnston/Documents/mmv_cvx/nets/results/'
  Aall = {}
  netlist = []
  for params in get_params_list():
    netlist.append([])
    netlist[-1].append(params)
    netlist[-1].append(get_path(params))
  
  for i in netlist:
    # set_trace()
    s = i[1][0]
    L,M,K = i[0][1], i[0][2], i[0][4]
    finalepoch = get_final_epoch(rootpath + s)
    A = np.load(rootpath + s + '/' + finalepoch + '/A.npy')
    Aall[(L,M,K)] = A

  np.save('./Aall.npy', Aall)
  
if __name__ == '__main__':
  A = np.load('./Aall.npy', allow_pickle=True).tolist()
  set_trace()
  extract_A()
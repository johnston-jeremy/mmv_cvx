import numpy as np
import numpy.linalg as la

def detect(Xhat,Xtrue):
  N,_ = Xhat.shape
  Pfa = []
  Pmd = []
  # set_trace()
  temp = la.norm(Xhat,axis=1)
  tt = np.logspace(np.log10(0.01),np.log10(1),30)*np.max(temp)
  true = (la.norm(Xtrue,axis=1) > 0).astype(int)
  Nt = sum(true)
  for t in tt:
    pred = (temp > t).astype(int)
    Pfa.append(sum((pred - true) > 0)/(N-Nt))
    Pmd.append(sum((true - pred) > 0)/Nt)
    if np.any(np.isnan(Pfa)) or np.any(np.isnan(Pmd)):
      import pdb
      pdb.set_trace()

  return Pfa, Pmd, tt

def detect_AP(Xhat,Xtrue):
  # Xhat and Xtrue are lists of matrices

  N = Xhat[0].shape[0]
  Nap = len(Xhat)
  Pfa = []
  Pmd = []

  Nthresh = 50
  pred = [0]*Nthresh

  temp = la.norm(Xhat[0],axis=1)
  true = (la.norm(Xtrue[0],axis=1) > 0).astype(int)
  Nt = sum(true)
  tt = np.logspace(np.log10(0.01),np.log10(1.5),Nthresh)
  for xhat in Xhat:
    temp = la.norm(xhat,axis=1)
    
    # detect
    for i in range(len(tt)):
      pred[i] += (temp > tt[i]).astype(int)

  # gather and majority vote
  for i in range(len(tt)):
    pred[i] = (pred[i]/Nap > 0.5).astype(int)
    # set_trace()
    Pfa.append(sum((pred[i] - true) > 0)/(N-Nt))
    Pmd.append(sum((true - pred[i]) > 0)/Nt)

    if sum((pred[i] - true) > 0)/(N-Nt) > 1:
      set_trace()
    if np.any(np.isnan(Pfa)) or np.any(np.isnan(Pmd)):
      import pdb
      pdb.set_trace()

  return np.array(Pfa), np.array(Pmd), tt





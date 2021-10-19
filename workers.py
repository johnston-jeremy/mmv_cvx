from vamp import vamp
import tqdm
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from data import gen_c
from problem import problem
from pdb import set_trace
from admm3 import admm_problem3
from admm1 import admm_problem1

def worker_oracle(inputs):
  E, prob, Yall, Xall, Zall, lams1,lams2, ind = inputs
  i, j, nsamp = ind
  Y = Yall[nsamp]
  X = Xall[nsamp]
  Z = Zall[nsamp]

  I = np.where(np.linalg.norm(X, axis=1) > 0)[0]
  # print(I)
  Xhat = np.zeros((prob.N,prob.M), dtype=complex)
  # Xhat[I] = np.linalg.pinv(prob.A[:,np.array(I)])@Y
  A = prob.A[:,np.array(I)]
  # A = prob.A
  # B = prob.Phi.T
  # Zhat = np.linalg.pinv(prob.A[:,np.array(I)])@Y@np.linalg.pinv(prob.Phi.T)
  # Zhat = np.linalg.inv(A.T.conj()@A)@A.T.conj()@Y@np.linalg.pinv(B) #B.T.conj()@np.linalg.inv(B@B.T.conj())
  Zhat = np.zeros((prob.N,prob.Ng), dtype=complex)
  Phiall = []
  for n in I:
    Ipaths = np.where(np.abs(Z[n]) > 0)[0]
    Phiall.append(prob.Phi[:,Ipaths])

    # Phi = prob.Phi[:,Ipaths]
    # B = Phi.T

    # rhs = A.T.conj()@Y@B.T.conj()@np.linalg.inv(B@B.T.conj())

    # rhs = (np.linalg.pinv(A.T.conj()@A)@(A.T.conj())@Y@(B.T.conj())).T
    # Zhat[n,Ipaths] = np.diag((np.linalg.pinv(B.conj()@B.T) @ rhs).T)
  Zcvx = [cp.Variable(prob.J, complex=True) for _ in I]
  exp = 0
  for i in range(len(I)):
    a = cp.Constant((A[:,i][:,None]).real) + cp.multiply(1j,cp.Constant((A[:,i][:,None]).imag))
    Phitemp = cp.Constant(Phiall[i].T.real) + cp.multiply(1j,cp.Constant(Phiall[i].T.imag))
    exp += (a @ (Zcvx[i] @ Phitemp)[:,None].T)
  obj = cp.norm(Y -exp)**2
  p = cp.Problem(cp.Minimize(obj))
  p.solve()
  # i = 0
  # for r in rhs.T:
  #   Zhat[:,i] = np.linalg.solve(A.T.conj()@A, r)
  #   i += 1
  # Zhat = np.linalg.solve(B.conj()@B.T, rhs).T
  # Zhat[I] = (np.linalg.pinv(B.conj()@B.T) @ rhs).T
  for i in range(len(I)):
    Ipaths = np.where(np.abs(Z[I[i]]) > 0)[0]
    Zhat[I[i],Ipaths] = Zcvx[i].value
  Xhat = Zhat @ prob.Phi.T
  # Zhat = np.linalg.pinv(prob.Phi)@Xhat.T
  # Xhat = (prob.Phi@Zhat).T
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

def worker_vamp(inputs):
  E, prob, Y, nsamp = inputs
  X = vamp(Y, prob)
  E.append({'Xhat':X, 'ind':nsamp})

def worker_admm3(inputs):
  E, prob, Y, nsamp = inputs
  X = admm_problem3(Y, prob)
  E.append({'Xhat':X, 'ind':nsamp})

def worker_admm1(inputs):
  E, prob, Y, nsamp = inputs
  X = admm_problem1(Y, prob)
  E.append({'Xhat':X, 'ind':nsamp})

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
  # obj = cp.norm(Y-p.A@(Zcvx@Phi.T))**2 + lam1*cp.sum(cp.norm(Zcvx@Phi.T,p=2,axis=1)) + lam2*cp.norm(Zcvx, p=1)  
  # set_trace()
  c = [Zcvx@Phi.T == Xcvx] # + [cp.imag(cp.matmul(Zcvx,p.Phi.T)) == cp.imag(Xcvx)]
  prob = cp.Problem(cp.Minimize(obj),c)
  prob.solve()
  Xhat = Zcvx.value @ (p.Phi.T)
  E.append({'Xhat':Xhat, 'Zhat':Zcvx.value, 'ind':ind})

def worker3(inputs):
  E, p, Yall, Xall, lams1,lams2, ind = inputs
  i, j, nsamp = ind
  Y = Yall[nsamp]
  lam1, lam2 = lams1[i], lams2[j]

  Phi = cp.Constant(p.Phi.real) + 1j*cp.Constant(p.Phi.imag)
  Zcvx = cp.Variable(shape=(p.N,p.Ng),complex=True)
  Xcvx = cp.Variable(shape=(p.N,p.M),complex=True)
  obj = lam1*cp.sum(cp.norm(Zcvx@Phi.T,p=2,axis=1)) + cp.norm(Zcvx, p=1)
  # obj = cp.sum(cp.norm(Xcvx,p=2,axis=1)) + cp.norm(Zcvx, p=1)
  # set_trace()
  c = [Y == p.A@Xcvx] + [Zcvx@Phi.T == Xcvx] # + [cp.imag(cp.matmul(Zcvx,p.Phi.T)) == cp.imag(Xcvx)]
  # c = [cp.norm(Y - p.A@Zcvx@Phi.T)**2 <= lam2*cp.norm(Y)**2] # + [cp.imag(cp.matmul(Zcvx,p.Phi.T)) == cp.imag(Xcvx)]
  prob = cp.Problem(cp.Minimize(obj), c)
  prob.solve(solver='MOSEK')
  E.append({'Xhat':Zcvx.value@p.Phi.T, 'Zhat':Zcvx.value, 'ind':ind})

def worker2(inputs):
  E, p, Yall, Xall, lams1, lams2, ind = inputs
  i, j, nsamp = ind
  Y = Yall[nsamp]
  lam1 = lams1[i]
  lam2 = lams2[j]

  Phi = cp.Constant(p.Phi.real) + cp.multiply(1j,cp.Constant(p.Phi.imag))
  # Phi = cp.Constant(p.Phi)

  Xcvx = cp.Variable(shape=(p.N,p.M),complex=True)
  obj = cp.norm(Y-p.A@Xcvx)**2 + lam1*cp.sum(cp.norm(Xcvx,p=2,axis=1))
  prob = cp.Problem(cp.Minimize(obj))
  prob.solve()

  Xhat = Xcvx.value
  
  for n in range(p.N):
    Zcvx = cp.Variable(shape=(p.Ng,),complex=True)
    obj = cp.norm(Xhat[n] - Phi@Zcvx)**2 + lam2 * cp.norm(Zcvx, p=1)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver='MOSEK',verbose=False)
    Xhat[n] = Zcvx.value @ (p.Phi.T)

  E.append({'Xhat':Xhat, 'ind':ind})

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

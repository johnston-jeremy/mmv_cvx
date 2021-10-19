from main import generate_data
import numpy as np
from admm1 import admm_problem1
from pdb import set_trace
from multiprocessing import Manager, Pool
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def worker_single(inputs):
  A, Phi, y, params, x, E = inputs
  Xhat, err = admm_problem3(A, Phi, y, params, x)
  E.append(err)

def worker_multiple(inputs):
  for P in admm_paramgrid:
    sigma, mu, rho, taux, tauz = P
    params = p.N, p.L, p.M, p.Ng, sigma, mu, rho, taux, tauz, maxiter1
    inputs = zip([p.A]*Nsamp, [p.Phi]*Nsamp, Y, [params]*Nsamp, X, [E]*Nsamp)
    # from time import time
    # T = time()
    # with Pool() as pool:
    #   pool.map(worker, inputs)
    # error_hist_avg = 10*np.log10(np.sum(E, axis=0)/Nsamp)
    # print(time() - T)
  
    error_hist = 0
    for i in range(Nsamp):
      Xhat, err = admm_problem3(p.A, p.Phi, Y[i], params, X[i])
      error_hist = error_hist + err
    error_hist_avg = 10*np.log10(error_hist/Nsamp)
    
    
    # plt.figure()
    # plt.plot(err1)
    # plt.title('admm3, alpha =' + str(alpha))
    errors.append(error_hist_avg[-1])
    

    print('{:.1e}'.format(sigma), \
          '{:.1e}'.format(mu), \
          '{:.1e}'.format(rho), \
          '{:.1e}'.format(taux), \
          '{:.1e}'.format(tauz), \
          '{:.2e}'.format(errors[-1]))

def detect(Xhat,Xtrue):
  N,_ = Xhat.shape
  Pfa = []
  Pmd = []
  # set_trace()
  temp = np.linalg.norm(Xhat,axis=1)
  tt = np.logspace(np.log10(0.01),np.log10(0.5),20)*np.max(temp)
  # tt = [0.3*np.max(temp)]
  true = (np.linalg.norm(Xtrue,axis=1) > 0).astype(int)
  Nt = sum(true)
  for t in tt:
    pred = (temp > t).astype(int)
    Pfa.append(sum((pred - true) > 0)/(N-Nt))
    Pmd.append(sum((true - pred) > 0)/Nt)

  return Pfa, Pmd

def rls(X,Xtrue,Y,p):
  Xnorm = np.linalg.norm(X,axis=1)
  ind = np.where(Xnorm > 0.2*np.max(Xnorm))[0]
  A = p.A[:,ind]
  indtrue = np.where(np.linalg.norm(Xtrue,axis=1)>0)[0]
  # set_trace()
  
  Xhat = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T.conj(),A)),A.T.conj()),Y)
  Xx = np.zeros_like(Xtrue)
  Xx[ind] = Xhat
  return np.sum(np.abs(Xtrue - Xx)**2)/np.sum(np.abs(Xtrue)**2)

def admm1_param_test():
  maxiter1 = 300
  p,Y,X = gen_problem_and_data(N=50,
                               L=12,
                               K=3,
                               Nsamp=100, 
                               SNR=10, 
                               Ntxrx=(8,1))

  admm_paramgrid = admm1_param_grid(0)
  Nsamp = p.Nsamp
  errors = []
  errors_hist = []
  PfaPmd = []
  # print('problem 3')
  # print('sigma\t','mu\t','rho\t','taux\t','tauz\t', 'error')
  
  manager = Manager()
  E = manager.list()

  for P in admm_paramgrid:
    mu,beta,taux = P
    params = p.N, p.L, p.M, mu,beta,taux, maxiter1
    inputs = zip([p.A]*Nsamp, Y, [params]*Nsamp, X, [E]*Nsamp)
    # from time import time
    # T = time()
    # with Pool() as pool:
    #   pool.map(worker, inputs)
    # error_hist_avg = 10*np.log10(np.sum(E, axis=0)/Nsamp)
    # print(time() - T)
  
    error_hist = 0
    d = 0
    Pfa = []
    Pmd = []
    err_ls = 0
    for i in range(Nsamp):
      Xhat, err = admm_problem1(p.A, Y[i], params, X[i])
      error_hist = error_hist + err
      g = detect(Xhat, X[i])
      errls = rls(Xhat, X[i], Y[i], p)
      # print(err[-1])
      # print(err_ls)
      err_ls += errls
      # set_trace()
      Pfa.append(g[0])
      Pmd.append(g[1])
    Pfa = np.round(np.mean(Pfa, axis=0), 5)
    Pmd = np.round(np.mean(Pmd, axis=0), 5)

    # print((Pfa+Pmd)/2)
    print(np.argmin((Pfa+Pmd)/2),Pfa[np.argmin((Pfa+Pmd)/2)], Pmd[np.argmin((Pfa+Pmd)/2)])
    plt.figure()
    plt.scatter(Pfa,Pmd)
    plt.xlabel('Pfa')
    plt.ylabel('Pmd')
    # plt.ylim([0,0.8])
    # plt.xlim([0,0.8])
    PfaPmd.append((Pfa,Pmd))
    error_hist_avg = 10*np.log10(error_hist/Nsamp)
    err_ls = 10*np.log10(err_ls/Nsamp)
    
    
    plt.figure()
    plt.plot(error_hist_avg)
    errors.append(error_hist_avg[-1])
    errors_hist.append(error_hist_avg)

    print(
      '{:.1e}'.format(mu), \
      '{:.1e}'.format(beta), \
      '{:.1e}'.format(taux), \
      '{:.4e}'.format(errors[-1]), \
      '{:.4e}'.format(err_ls))

  # topN = 10
  # print('Top {}'.format(topN))
  # print('sigma\t','mu\t','rho\t','taux\t','tauz\t', 'error')
  # x = np.array(list(enumerate(errors)))
  # ind = np.lexsort((x[:,0],x[:,1]))
  # for i in ind[:topN]:
  #   sigma, mu, rho, taux, tauz = admm_paramgrid[i]
  #   print('{:.1e}'.format(sigma), \
  #         '{:.1e}'.format(mu), \
  #         '{:.1e}'.format(rho), \
  #         '{:.1e}'.format(taux), \
  #         '{:.1e}'.format(tauz), \
  #         '{:.2e}'.format(errors[i]))
  #   plt.plot(errors_hist[i])
  #   plt.title('i = ' + str(i))
  # plt.legend([str(np.round(admm_paramgrid[i][3],3)) for i in ind[:topN]])

  # print([i for i in zip(PfaPmd[ind[0]][0],PfaPmd[ind[0]][1])])
  # plt.figure()
  # for i in ind[:10]:
  #   Pfa,Pmd = PfaPmd[i]
  #   print(Pfa[0],Pmd[0])
  #   print(Pfa[1],Pmd[1])
  #   print(Pfa[2],Pmd[2])
  #   print(Pfa[3],Pmd[3])
  #   print(Pfa[4],Pmd[4])
  #   print('---')
  #   plt.scatter(Pfa,Pmd)
  # Pfa,Pmd = PfaPmd[ind[0]]
  # plt.legend([str(i) for i in ind[:topN]])
  
  plt.show()
  return errors[-1]

def admm1_param_grid(grid_size):
  
  # Best params - real gaussian
  # mu = 0.056
  # beta = 0.1
  # taux = 0.056

  mu =  np.logspace(0,1,grid_size)
  beta = np.logspace(-1,-0,grid_size)
  # taux = np.logspace(-1,-0.3,grid_size)
  taux = np.logspace(-4,-3,grid_size)

  mu,beta,taux = 2.1e-01, 8.4e-01, 5.9e-02
  # mu,beta,taux=  6.9e-02, 2.4e-02, 2.8e-01

  # custom = [[2.1e-01, 8.4e-01, 5.9e-02],[6.9e-02, 2.4e-02, 2.8e-01]]
  # mu = 0.07
  # beta = 0.021
  # taux = 0.026

  # mimo
  # mu = 0.5
  # beta = 1
  # taux = 0.2

  # mu = 1

  m, b, tx = np.meshgrid(mu,beta,taux)
  paramgrid = zip(m.ravel(), b.ravel(), tx.ravel())

  return list(paramgrid)
  # return custom

def admm_opt(var, *args):
  mu, beta, taux = var
  maxiter1 = 200
  p, Y, X = args
  params = p.N, p.L, p.M, mu, beta, taux, maxiter1
  Nsamp = Y.shape[0]
  error_hist = 0
  errors = []
  for i in range(Nsamp):
    Xhat, err = admm_problem1(p.A, Y[i], params, X[i])
    error_hist = error_hist + err
    # g = detect(Xhat, X[i])

  # PfaPmd.append((Pfa,Pmd))
  error_hist_avg = 10*np.log10(error_hist/Nsamp)
  
  
  # plt.figure()
  # plt.plot(err1)
  # plt.title('admm3, alpha =' + str(alpha))
  errors.append(error_hist_avg[-1])
  # errors_hist.append(error_hist_avg)

  print(
        '{:.1e}'.format(mu), \
        '{:.1e}'.format(beta), \
        '{:.1e}'.format(taux), \
        '{:.4e}'.format(errors[-1]))

  return error_hist_avg[-1]
  # L =15
# 1.2e-01 1.1e-02 3.8e-02
# 1.1e-01 1.1e-02 5.8e-02
# 3.4e-02 1.1e-02 3.9e-02
# -8.3e-03 7.3e-03 4.0e-01
# L = 10
# 2.1e-01 8.4e-01 5.9e-02
# 7.0e-02 3.4e-02 1.5e-03
def opt():
  maxiter1 = 300
  mu = 1e-2
  beta = 1e-1
  taux = 1e-2

  mu = 1e-2
  beta = 1e-1
  taux = 1e-2

  # mu, beta, taux

  args = gen_problem_and_data(N=50,
                               L=10,
                               K=3,
                               Nsamp=15, 
                               SNR=10, 
                               Ntxrx=(8,1))

  method = 'powell'
  method='nelder-mead'
  minimize(admm_opt, 
           x0=(mu, beta, taux), 
           args=args,
           method='nelder-mead',
           tol=1e-3,
           bounds=[(0,2)]*3)

def admm_opt_scalar(x, *args):
  pYX, d, var = args
  d[var] = x

  maxiter1 = 1000
  p, Y, X = pYX
  params = p.N, p.L, p.M, d['mu'], d['beta'], d['taux'], d['gamma'], maxiter1
  Nsamp = Y.shape[0]
  error_hist = 0
  errors = []
  
  avgerror = admm_multiprocess(p, Y, params, X)

  errors.append(10*np.log10(avgerror))
  # set_trace()
  print('{:.1e},'.format(d['mu']), \
        '{:.1e},'.format(d['beta']), \
        '{:.1e},'.format(d['taux']), \
        '{:.4e}'.format(errors[-1]),
        var)
  return avgerror

def admm_worker(inputs):
  Y, p, X, E = inputs
  Xhat = admm_problem1(Y, p)
  E.append((np.linalg.norm(X-Xhat)/np.linalg.norm(X))**2)

def admm_multiprocess(p, Y, params, X):
  Nsamp = Y.shape[0]
  manager = Manager()
  E = manager.list()
  inputs = zip(Y, [p]*Nsamp, X, [E]*Nsamp)
  with Pool() as pool:
    pool.map(admm_worker, inputs)
  # error_hist = np.sum(E, axis=0)
  # return error_hist
  return np.mean(E)

def minimize_grid(f, t1, t2, args):
  grid = np.logspace(t1,t2,10)
  res = [f(x, *args) for x in grid]
  ind = np.argsort(res)[0]
  return grid[ind], res[ind]

def mymin(T,p,Y,X):
  # mu, beta, taux = 2.1e-01, 8.4e-01, 5.9e-02
  # mu, beta, taux = [1e-1]*3
  mu, beta, taux, gamma = [0.5]*4
  # mu, beta, taux = 0.217, 0.057, 0.008

  d = {'mu':mu, 'beta':beta, 'taux':taux, 'gamma': gamma}
  variables = ['mu','beta','taux', 'gamma']
  # variables = ['mu','taux']
  # variables = ['beta']
  # zupper = {{'mu':0.7, 'beta':0.7, 'taux':0.7}}
  zupper = 0.4
  zlower = 4
  # zlower = {{'mu':4, 'beta':4, 'taux':4}}
  # pYX = gen_problem_and_data(N=50,
  #                             L=8,
  #                             K=3,
  #                             Nsamp=100, 
  #                             SNR=10, 
  #                             Ntxrx=(8,1))
  for t in range(T):
    zupper = zupper/(0.5*t+1)
    zlower = zlower/(0.5*t+1)
    print('t =', t, 'zupper =',zupper,'zlower =',zlower)
    # np.random.shuffle(variables)
    # print(variables)
    for var in variables:
      p.mu = d['mu']
      p.beta = d['beta']
      p.taux = d['taux']
      p.gamma = d['gamma']
      p.maxiter = 1000
      p.tol = 1e-6

      args = (p,Y,X), d, var

      d[var], pstar = minimize_grid(admm_opt_scalar, 
              np.log10(d[var]) - zlower, np.log10(d[var]) + zupper,
              args=args)

      # print(var, '=', d[var], 'NMSE =', pstar)

  print(d.values())
  return d.values()

def routine1():
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
  
  # params_list = [ (50, 16,8,2*8,3,(8,1),2, 10),
  #                 (50, 20,8,2*8,3,(8,1),2, 10)]

  # params_list = [(50,12,8,2*8,4,(8,1),2, 10),
  #                 (50,12,8,2*8,5,(8,1),2, 10),
  #                 (50,12,8,2*8,6,(8,1),2, 10),
  #                 (50,12,8,2*8,7,(8,1),2, 10),
  #                 (50,12,8,2*8,8,(8,1),2, 10)]
  # params_list = [(50,12,8,2*8,4,(8,1),2, 10)]
  
  hyperparams = []
  Nsamp = 5
  for N,L,M,_,K,_,_,SNR in params_list:
    Y, X, Z, p = generate_data(Nsamp,L,M,K,'mmwave')


    hyperparams.append(mymin(3,p,Y,X))
  
  i=0
  for N,L,M,_,K,_,_,SNR in params_list:
    key = [str(k)+',' for k in [N, L, M, K, SNR]]
    val = [str(h) + ',' for h in hyperparams[i]]
    print('params_admm1[(', *key, ')] =', *val)
    # print(*[str(np.round(h,3)) + ',' for h in hyperparams[i]])
    i += 1

if __name__ == '__main__':
  # opt()
  # admm1_param_test()
  # mymin(2)
  routine1()
    # plt.plot([admm3_param_test(l) for l in range(5,30,2)])
    # plt.show()
  # for l in range(0,16,3):
  #   print('l =', l)
  #   admm1_param_test(l)
  # admm1_param_test()
  # plt.show()
    # admm3_param_test()
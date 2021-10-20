import os

from problem import problem
from pdb import set_trace
import numpy as np

params_list = [ (50,12,4,2*4,3,(4,1),2, 10),
                (50,12,8,2*8,3,(8,1),2, 10), 
                (50,12,12,2*12,3,(12,1),2, 10),
                (50,12,16,2*16,3,(16,1),2, 10),
                (50,12,8,2*8,4,(8,1),2, 10),
                (50,12,8,2*8,5,(8,1),2, 10),
                (50,12,8,2*8,6,(8,1),2, 10),
                (50,12,8,2*8,7,(8,1),2, 10),
                (50,12,8,2*8,8,(8,1),2, 10),
                (50,12,8,2*8,3,(8,1),2, 0),
                (50,12,8,2*8,3,(8,1),2, 5),
                (50,12,8,2*8,3,(8,1),2, 15),
                (50,12,8,2*8,3,(8,1),2, 20)]

params_list = [(50, 4,8,2*8,3,(8,1),2, 10),
               (50, 8,8,2*8,3,(8,1),2, 10),
               (50, 16,8,2*8,3,(8,1),2, 10),
               (50, 20,8,2*8,3,(8,1),2, 10)]

# problem_list = [(problem(50,12,4,2*4,3,(4,1),2), 10),
#                 (problem(50,12,8,2*8,3,(8,1),2), 10), 
#                 (problem(50,12,12,2*12,3,(12,1),2), 10),
#                 (problem(50,12,16,2*16,3,(16,1),2), 10),
#                 (problem(50,12,8,2*8,4,(8,1),2), 10),
#                 (problem(50,12,8,2*8,5,(8,1),2), 10),
#                 (problem(50,12,8,2*8,6,(8,1),2), 10),
#                 (problem(50,12,8,2*8,7,(8,1),2), 10),
#                 (problem(50,12,8,2*8,8,(8,1),2), 10),
#                 (problem(50,12,8,2*8,3,(8,1),2), 0),
#                 (problem(50,12,8,2*8,3,(8,1),2), 5),
#                 (problem(50,12,8,2*8,3,(8,1),2), 15),
#                 (problem(50,12,8,2*8,3,(8,1),2), 20)]

def parse(x):
  words = []
  words.append('')
  for i in x:
      if i != '_':
        words[-1] += i
      elif i == '_':
        words.append('')
  return words

def parse_dims(x):
  words = []
  words.append('')
  for i in x:
      if i != 'x':
        words[-1] += i
      elif i == 'x':
        words.append('')
  return words

def parse_SNR(x):
  word = ''
  for i in x:
      if i in ['S','N','R','d','B','=']:
        pass
      else:
        word += i
  return word

def get_path(params):
  N,L,M,Ng,K,Ntxrx,J,SNR = params
  X = os.listdir('/Users/jeremyjohnston/Documents/mmv_cvx/nets/results')
  # X = os.listdir('/Users/Jeremy/Documents/GitHub/mmv_cvx/nets/results')
  # X = os.listdir('/Users/Jeremy/Documents/mmv-old/dunn/nets/results')
  # X = os.listdir('C:/Users/Jeremy/Documents/results_admm_ampista_3_25_21')
  # X = os.listdir('/Users/Jeremy/Documents/mmv/isit_results/')
  res = []
  for x in X:
    p = parse(x)
    # print(p)
    if len(p) == 12:
    
      d = parse_dims(p[1])
      if d[0] == str(N) and \
         d[1] == str(M) and \
         d[2] == str(L) and \
         p[2][-1] == str(J) and \
         p[3][-1] == str(K) and \
         parse_SNR(p[4]) == str(SNR):
         res.append(x)

  return res
    # return

def get_numlayers(x):
  res = ''
  for i in x:
    if i == 'l':
      return res
    res += i

def get_final_epoch(path):
  contents = os.listdir(path)
  for c in contents:
    p = parse(c)
    if p[0] == 'Epoch=30':
      return c

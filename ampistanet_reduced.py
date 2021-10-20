#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:40:31 2020

@author: jeremyjohnston
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import numpy.linalg as la
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

class AMPNet(tf.keras.Model):
  
  def __init__(self, p, num_stages, *args, **kwargs):
    super().__init__()

    self.N = p.N
    self.M = p.M
    self.Ng = p.Ng
    self.Phi = p.Phi
    self.Layers=[]
    istabundle = istaBundle(p, num_stages['ista'])
    # istabundle = istaBundle_tied(p, num_stages['ista'])
    # M3_init = np.float32(p.A)
    # top = tf.concat((p.A.real, -p.A.imag),axis=1)
    # bot = tf.concat((p.A.imag, p.A.real),axis=1)
    # M3_init = tf.concat((np.float32(top),np.float32(bot)), axis=0)
    # self.M3 = tf.Variable(initial_value=M3_init,
    #                      trainable=True, name='M3')

    # M3 = p.A.T.conj()
    # self.M3re1 = tf.Variable(initial_value=np.float32(M3.real),
    #                      trainable=True, name='M3re1')
    # self.M3re2 = tf.Variable(initial_value=np.float32(M3.real),
    #                      trainable=True, name='M3re2')
    # self.M3im1 = tf.Variable(initial_value=np.float32(M3.imag),
    #                      trainable=True, name='M3im1')
    # self.M3im2 = tf.Variable(initial_value=np.float32(M3.imag),
    #                      trainable=True, name='M3im2')

    for i in range(num_stages['amp']):
      self.Layers.append(ampBundle(p, istabundle, i))

    print('AMP-Net with', num_stages['amp'], 'stages')
    
    self.loss_fcn = tf.keras.losses.MeanSquaredError()

  def call(self, Y):
    Yre, Yim = Y

    Xre = tf.zeros((self.N,self.M), dtype=tf.float32)
    Xim = tf.zeros((self.N,self.M), dtype=tf.float32)
    # X = tf.concat((Xre,Xim), axis=0)
    
    Zre = tf.zeros((self.Ng, self.N))
    Zim = tf.zeros((self.Ng, self.N))
    # Z0 = tf.concat((Z0re,Z0im


    Rre = Yre
    Rim = Yim
    # import pdb; pdb.set_trace()
    for l in self.Layers:
      # Xre, Xim, Rre, Rim, Zre, Zim = l(Yre, Yim, Xre, Xim, Rre, Rim, Zre, Zim, self.M3re1, self.M3re2, self.M3im1, self.M3im2)
      Xre, Xim, Rre, Rim, Zre, Zim = l(Yre, Yim, Xre, Xim, Rre, Rim, Zre, Zim)
    # import pdb; pdb.set_trace()
    return Xre, Xim

  @tf.function
  def train_step(self, x, y_true):
    # set_trace()
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.loss_fcn(y_pred[0], y_true[0]) + self.loss_fcn(y_pred[1], y_true[1])

    trainable_variables = self.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss

class ampBundle(layers.Layer):
  def __init__(self, p, istabundle, layerid, *args):
    super().__init__()
    self.l1 = ampLayer1(p)
    self.l2 = ampLayer2(p,layerid)
    self.ista = istabundle
    M3 = p.A.T.conj()
    self.M3re = tf.Variable(initial_value=np.float32(M3.real),
                         trainable=True, name='M3re')
    self.M3im = tf.Variable(initial_value=np.float32(M3.imag),
                         trainable=True, name='M3im')

  # def call(self, Yre, Yim, Xre, Xim, Rre, Rim, Zre, Zim, M3re1, M3re2, M3im1, M3im2):
  def call(self, Yre, Yim, Xre, Xim, Rre, Rim, Zre, Zim):

    Xtildere, Xtildeim = self.l1(Xre, Xim, Rre, Rim, self.M3re, self.M3im)

    # with tf.GradientTape() as g:
    #   g.watch(Xtildere)
    #   g.watch(Xtildeim)
    #   Xre, Xim, Zre, Zim = self.ista(Xtildere, Xtildeim, Zre, Zim)
    # Jre = g.jacobian(Xre,Xtildere)
    # Jim = g.jacobian(Xim,Xtildeim)

    Xre, Xim, Zre, Zim = self.ista(Xtildere, Xtildeim, Zre, Zim)

    Rre, Rim = self.l2(Yre, Yim, Xre, Xim, Rre, Rim, self.M3re, self.M3im)

    return Xre, Xim, Rre, Rim, Zre, Zim

class istaBundle(layers.Layer):
  def __init__(self, p, num_stages, *args):
    super().__init__()
    self.Layers = []
    # top = tf.concat((p.Phi.real, -p.Phi.imag),axis=1)
    # bot = tf.concat((p.Phi.imag, p.Phi.real),axis=1)
    # self.Phi = tf.concat((np.float32(top),np.float32(bot)), axis=0)
    self.p = p
    for i in range(num_stages):
      self.Layers.append(istaStage(p))
    
    print('ISTA-Net with', num_stages, 'stages')

  def call(self, Xre, Xim, Zre, Zim):
    
    for l in self.Layers:
      Zre, Zim, M2 = l(Xre, Xim, Zre, Zim)

    M2re, M2im = M2

    Xhatre = tf.transpose(tf.matmul(tf.transpose(M2re)/self.p.alpha0, Zre) - tf.matmul(tf.transpose(-M2im)/self.p.alpha0, Zim), perm=(0,2,1))
    Xhatim = tf.transpose(tf.matmul(tf.transpose(-M2im)/self.p.alpha0, Zre) + tf.matmul(tf.transpose(M2re)/self.p.alpha0, Zim), perm=(0,2,1))
    
    # Xhatre = tf.transpose(tf.matmul(self.p.Phi.real, Zre) - tf.matmul(self.p.Phi.imag, Zim), perm=(0,2,1))
    # Xhatim = tf.transpose(tf.matmul(self.p.Phi.imag, Zre) + tf.matmul(self.p.Phi.real, Zim), perm=(0,2,1))

    



    # Xhat = tf.concat((Xhat[:, :, :self.p.M],Xhat[:,:,self.p.M:]), axis=1)

    return Xhatre, Xhatim, Zre, Zim

class istaStage(layers.Layer):

  def __init__(self, p, *args):
    super().__init__()

    # M1 = tf.eye(p.Phi.shape[1]) - p.alpha0*tf.matmul(tf.transpose(p.Phi), p.Phi)
    # M2 = p.alpha0*tf.transpose(p.Phi)
    self.p = p
    M1 = np.eye(p.Ng) - p.alpha*np.matmul(p.Phi.T.conj(),p.Phi)
    M2 = p.alpha*p.Phi.T.conj()
    # top = tf.concat((M1.real, -M1.imag),axis=1)
    # bot = tf.concat((M1.imag, M1.real),axis=1)
    # M1_init = np.float32(tf.concat((top,bot), axis=0))

    # top = tf.concat((M2.real, -M2.imag),axis=1)
    # bot = tf.concat((M2.imag, M2.real),axis=1)
    # M2_init = np.float32(tf.concat((top,bot), axis=0))

    self.M1re = tf.Variable(initial_value=np.float32(M1.real),
                         trainable=True, name='M1re')
    self.M1im = tf.Variable(initial_value=np.float32(M1.imag),
                         trainable=True, name='M1im')
    
    self.M2re = tf.Variable(initial_value=np.float32(M2.real),
                         trainable=True, name='M2re')
    self.M2im = tf.Variable(initial_value=np.float32(M2.imag),
                         trainable=True, name='M2im')

    self.lam = tf.Variable(initial_value=np.float32(p.alpha0*p.lam),
                         trainable=True, name='lam')
  
  def soft_threshold_complex(self, x_re, x_im, kappa):
    # x is a shape (N,2) array whose rows correspond to complex numbers
    # returns shape (N,2) array corresponding to complex numbers

    norm = tf.sqrt(x_re**2 + x_im**2)
    # norm = tf.norm(x,axis=0)
    x_re_normalized = tf.math.divide_no_nan(x_re,norm)
    x_im_normalized = tf.math.divide_no_nan(x_im,norm)
    # z = tf.math.multiply(x_normalized,tf.maximum(norm - kappa,0))
    maxterm = tf.maximum(norm - kappa,0)
    z_re = tf.math.multiply(x_re_normalized, maxterm)
    z_im = tf.math.multiply(x_im_normalized, maxterm)
    # import pdb; pdb.set_trace()
    # z = tf.reshape(z, (self.p.batch, 2*self.p.Ng, self.p.N))
    return z_re, z_im

  def call(self, Xre, Xim, Zre, Zim):
    # import pdb; pdb.set_trace()
    # A = tf.matmul(self.M1, Z)
    Are = tf.matmul(self.M1re, Zre) - tf.matmul(self.M1im, Zim)
    Aim = tf.matmul(self.M1im, Zre) + tf.matmul(self.M1re, Zim)

    # B = tf.matmul(self.M2, X)
    # Bre = tf.matmul(self.M2re1, Xre) - tf.matmul(self.M2im1, Xim)
    # Bim = tf.matmul(self.M2im2, Xre) + tf.matmul(self.M2re2, Xim)

    Bre = tf.matmul(Xre, tf.transpose(self.M2re)) - tf.matmul(Xim, tf.transpose(self.M2im))
    Bim = tf.matmul(Xre, tf.transpose(self.M2im)) + tf.matmul(Xim, tf.transpose(self.M2re))

    Bre = tf.transpose(Bre, perm=(0,2,1))
    Bim = tf.transpose(Bim, perm=(0,2,1))
    # print('ista')
    # print(Aim.numpy())
    # print(Are.numpy())
    # print(Bre.numpy())
    # print(Bim.numpy())
    # print('ista')
    # import pdb; pdb.set_trace()

    Cre = Are + Bre
    Cim = Aim + Bim
    # C = tf.reshape(A + B, (2, self.p.batch, self.p.Ng, self.p.N))

    M2 = self.M2re, self.M2im
    Zre, Zim = self.soft_threshold_complex(Cre, Cim, self.lam)
    return Zre, Zim, M2

class ampLayer1(layers.Layer):

  def __init__(self, p, *args):
    super().__init__()
    # M3_init = p.A
    # self.M3 = tf.Variable(initial_value=M3_init,
    #                      trainable=True, name='M3_l1')

    

  def call(self, Xre, Xim, Rre, Rim, M3re, M3im):
    # import pdb; pdb.set_trace()
    # Xtilde = tf.transpose(tf.matmul(tf.transpose(R, perm=(0,2,1)), M3), perm=(0,2,1)) + X
    Xtildere = tf.matmul(M3re, Rre) - tf.matmul(M3im, Rim) + Xre
    Xtildeim = tf.matmul(M3im, Rre) + tf.matmul(M3re, Rim) + Xim

    return Xtildere, Xtildeim

class ampLayer2(layers.Layer):

  def __init__(self, p, layerid, *args):
    super().__init__()

    self.damp = tf.Variable(initial_value=np.float32(p.damp_init),
                            trainable=True, name='damp_{}'.format(layerid))

    # self.M4re1 = tf.Variable(initial_value=np.float32(np.zeros((p.M,p.M))),
    #                      trainable=True, name='M4re1')
    # self.M4re2 = tf.Variable(initial_value=np.float32(np.zeros((p.M,p.M))),
    #                      trainable=True, name='M4re2')
    # self.M4im1 = tf.Variable(initial_value=np.float32(np.zeros((p.M,p.M))),
    #                      trainable=True, name='M4im1')
    # self.M4im2 = tf.Variable(initial_value=np.float32(np.zeros((p.M,p.M))),
    #                      trainable=True, name='M4im2')

    # self.M4re = tf.Variable(initial_value=np.float32(np.random.randn(p.M,p.M)/10),
    #                      trainable=True, name='M4re')
    # self.M4im = tf.Variable(initial_value=np.float32(np.random.randn(p.M,p.M)/10),
    #                      trainable=True, name='M4im')
    
  def call(self, Yre, Yim, Xre, Xim, Rre, Rim, M3re, M3im):

    Rnewre = Yre - (tf.matmul(tf.transpose(M3re), Xre) - tf.matmul(-tf.transpose(M3im), Xim))
    Rnewim = Yim - (tf.matmul(tf.transpose(-M3im), Xre) + tf.matmul(tf.transpose(M3re), Xim))


    # Rnewre = Yre \
    #   - (tf.matmul(tf.transpose(M3re), Xre) - tf.matmul(-tf.transpose(M3im), Xim)) \
    #   + tf.matmul(Rre, self.M4re) - tf.matmul(Rim, self.M4im)
        
    # Rnewim = Yim \
    #   - (tf.matmul(tf.transpose(-M3im), Xre) + tf.matmul(tf.transpose(M3re), Xim)) \
    #   + tf.matmul(Rim, self.M4re) - tf.matmul(Rre, self.M4im)

    Rre = self.damp*Rre + (1-self.damp)*Rnewre
    Rim = self.damp*Rim + (1-self.damp)*Rnewim

    return Rre, Rim

class istaBundle_tied(layers.Layer):
  def __init__(self, p, num_stages, *args):
    super().__init__()
    self.Layers = []
    # top = tf.concat((p.Phi.real, -p.Phi.imag),axis=1)
    # bot = tf.concat((p.Phi.imag, p.Phi.real),axis=1)
    # self.Phi = tf.concat((np.float32(top),np.float32(bot)), axis=0)
    self.p = p

    M1 = np.eye(p.Ng) - p.alpha*np.matmul(p.Phi.T.conj(),p.Phi)
    M2 = p.alpha*p.Phi.T.conj()

    M1re1 = tf.Variable(initial_value=np.float32(M1.real),
                         trainable=True, name='M1re1')
    M1re2 = tf.Variable(initial_value=np.float32(M1.real),
                         trainable=True, name='M1re2')
    M1im1 = tf.Variable(initial_value=np.float32(M1.imag),
                         trainable=True, name='M1im1')
    M1im2 = tf.Variable(initial_value=np.float32(M1.imag),
                         trainable=True, name='M1im2')
    
    M2re1 = tf.Variable(initial_value=np.float32(M2.real),
                         trainable=True, name='M2re1')
    M2re2 = tf.Variable(initial_value=np.float32(M2.real),
                         trainable=True, name='M2re2')
    M2im1 = tf.Variable(initial_value=np.float32(M2.imag),
                         trainable=True, name='M2im1')
    M2im2 = tf.Variable(initial_value=np.float32(M2.imag),
                         trainable=True, name='M2im2')

    self.M = [M1re1, M1re2, M1im1, M1im2, M2re1, M2re2, M2im1, M2im2]

    for i in range(num_stages):
      self.Layers.append(istaStage_tied(p))
    
    print('ISTA-Net with', num_stages, 'stages')

  def call(self, Xre, Xim, Zre, Zim):
    
    for l in self.Layers:
      Zre, Zim = l(Xre, Xim, Zre, Zim, self.M)

    M1re1, M1re2, M1im1, M1im2, M2re1, M2re2, M2im1, M2im2 = self.M

    Xhatre = tf.transpose(tf.matmul(tf.transpose(M2re1)/self.p.alpha0, Zre) - tf.matmul(tf.transpose(-M2im1)/self.p.alpha0, Zim), perm=(0,2,1))
    Xhatim = tf.transpose(tf.matmul(tf.transpose(-M2im2)/self.p.alpha0, Zre) + tf.matmul(tf.transpose(M2re2)/self.p.alpha0, Zim), perm=(0,2,1))
    
    # Xhatre = tf.transpose(tf.matmul(self.p.Phi.real, Zre) - tf.matmul(self.p.Phi.imag, Zim), perm=(0,2,1))
    # Xhatim = tf.transpose(tf.matmul(self.p.Phi.imag, Zre) + tf.matmul(self.p.Phi.real, Zim), perm=(0,2,1))

    



    # Xhat = tf.concat((Xhat[:, :, :self.p.M],Xhat[:,:,self.p.M:]), axis=1)

    return Xhatre, Xhatim, Zre, Zim

class istaStage_tied(layers.Layer):

  def __init__(self, p, *args):
    super().__init__()

    # M1 = tf.eye(p.Phi.shape[1]) - p.alpha0*tf.matmul(tf.transpose(p.Phi), p.Phi)
    # M2 = p.alpha0*tf.transpose(p.Phi)
    self.p = p
    
    # top = tf.concat((M1.real, -M1.imag),axis=1)
    # bot = tf.concat((M1.imag, M1.real),axis=1)
    # M1_init = np.float32(tf.concat((top,bot), axis=0))

    # top = tf.concat((M2.real, -M2.imag),axis=1)
    # bot = tf.concat((M2.imag, M2.real),axis=1)
    # M2_init = np.float32(tf.concat((top,bot), axis=0))

    

    self.lam = tf.Variable(initial_value=np.float32(p.alpha0*p.lam),
                         trainable=True, name='lam')
  
  def soft_threshold_complex(self, x_re, x_im, kappa):
    # x is a shape (N,2) array whose rows correspond to complex numbers
    # returns shape (N,2) array corresponding to complex numbers

    norm = tf.sqrt(x_re**2 + x_im**2)
    # norm = tf.norm(x,axis=0)
    x_re_normalized = tf.math.divide_no_nan(x_re,norm)
    x_im_normalized = tf.math.divide_no_nan(x_im,norm)
    # z = tf.math.multiply(x_normalized,tf.maximum(norm - kappa,0))
    maxterm = tf.maximum(norm - kappa,0)
    z_re = tf.math.multiply(x_re_normalized, maxterm)
    z_im = tf.math.multiply(x_im_normalized, maxterm)
    # import pdb; pdb.set_trace()
    # z = tf.reshape(z, (self.p.batch, 2*self.p.Ng, self.p.N))
    return z_re, z_im

  def call(self, Xre, Xim, Zre, Zim, M):
    M1re1, M1re2, M1im1, M1im2, M2re1, M2re2, M2im1, M2im2 = M
    # import pdb; pdb.set_trace()
    # A = tf.matmul(self.M1, Z)
    Are = tf.matmul(M1re1, Zre) - tf.matmul(M1im1, Zim)
    Aim = tf.matmul(M1im2, Zre) + tf.matmul(M1re2, Zim)

    # B = tf.matmul(self.M2, X)
    # Bre = tf.matmul(self.M2re1, Xre) - tf.matmul(self.M2im1, Xim)
    # Bim = tf.matmul(self.M2im2, Xre) + tf.matmul(self.M2re2, Xim)

    Bre = tf.matmul(Xre, tf.transpose(M2re1)) - tf.matmul(Xim, tf.transpose(M2im1))
    Bim = tf.matmul(Xre, tf.transpose(M2im2)) + tf.matmul(Xim, tf.transpose(M2re2))

    Bre = tf.transpose(Bre, perm=(0,2,1))
    Bim = tf.transpose(Bim, perm=(0,2,1))
    # print('ista')
    # print(Aim.numpy())
    # print(Are.numpy())
    # print(Bre.numpy())
    # print(Bim.numpy())
    # print('ista')
    # import pdb; pdb.set_trace()

    Cre = Are + Bre
    Cim = Aim + Bim
    # C = tf.reshape(A + B, (2, self.p.batch, self.p.Ng, self.p.N))

    Zre, Zim = self.soft_threshold_complex(Cre, Cim, self.lam)
    return Zre, Zim

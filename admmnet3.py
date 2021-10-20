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
from pdb import set_trace

class ADMMNet(tf.keras.Model):
  
  def __init__(self, p, num_stages, *args, **kwargs):
    super().__init__()

    self.N = p.N
    self.M = p.M
    self.Ng = p.Ng
    self.Phi = p.Phi
    self.Layers=[]

    for i in range(num_stages):
      # self.Layers.append(Stage(p,i))
      self.Layers.append(Stage_reduced(p,i))

    self.loss_fcn = tf.keras.losses.MeanSquaredError()

  def call(self, Y):
    Yre, Yim = Y
    Xre = tf.zeros((self.N,self.M),dtype=tf.float32)
    Xim = tf.zeros((self.N,self.M),dtype=tf.float32)
    Zre = tf.zeros((self.N,self.Ng),dtype=tf.float32)
    Zim = tf.zeros((self.N,self.Ng),dtype=tf.float32)
    Xhatre = tf.zeros((self.N,self.M),dtype=tf.float32)
    Xhatim = tf.zeros((self.N,self.M),dtype=tf.float32)
    Ure = tf.zeros_like(Xre)
    Uim = tf.zeros_like(Xim)

    for l in self.Layers:
      Xre, Xim, Zre, Zim, Xhatre, Xhatim, Ure, Uim = l(Yre, Yim, Xre, Xim, Zre, Zim, Xhatre, Xhatim, Ure, Uim)
    
    return Xhatre, Xhatim,

  def soft_threshold_real(x, lam):
    return tf.math.multiply(tf.math.sign(x), tf.maximum(tf.math.abs(x) - lam, 0))

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

class Stage(layers.Layer):

  def __init__(self, p, layerid):
    super().__init__()

    M1 = p.A
    # M2 = np.matmul(np.conj(p.A.T), p.A)
    # M3 = np.conj(p.A.T)
    # M2 = p.M2
    # M3 = p.M3

    self.A = p.A
    # self.Phi = p.Phi
    
    # self.rho = p.rho
    # self.mu = p.mu
    # self.taux = p.taux
    # self.tauz = p.tauz
    # self.sigma = p.sigma

    self.rho = tf.Variable(initial_value=p.rho, trainable=True, name='rho')
    self.mu = tf.Variable(initial_value=p.mu, trainable=True, name='mu')
    self.taux = tf.Variable(initial_value=p.taux, trainable=True, name='taux')
    self.tauz = tf.Variable(initial_value=p.tauz, trainable=True, name='tauz')
    self.sigma = tf.Variable(initial_value=p.sigma, trainable=True, name='sigma')

    # self.M1re1 = tf.Variable(initial_value=np.float32(M1.real),
    #                      trainable=True, name='M1re1')
    # self.M1re2 = tf.Variable(initial_value=np.float32(M1.real),
    #                      trainable=True, name='M1re2')
    # self.M1im1 = tf.Variable(initial_value=np.float32(M1.imag),
    #                      trainable=True, name='M1im1')
    # self.M1im2 = tf.Variable(initial_value=np.float32(M1.imag),
    #                      trainable=True, name='M1im2')
    
    self.M2re1 = tf.Variable(initial_value=np.float32(p.M2.real),
                         trainable=True, name='M2re1')
    self.M2re2 = tf.Variable(initial_value=np.float32(p.M2.real),
                         trainable=True, name='M2re2')
    self.M2im1 = tf.Variable(initial_value=np.float32(p.M2.imag),
                         trainable=True, name='M2im1')
    self.M2im2 = tf.Variable(initial_value=np.float32(p.M2.imag),
                         trainable=True, name='M2im2')

    self.M3re1 = tf.Variable(initial_value=np.float32(p.M3.real),
                         trainable=True, name='M3re1')
    self.M3re2 = tf.Variable(initial_value=np.float32(p.M3.real),
                         trainable=True, name='M3re2')
    self.M3im1 = tf.Variable(initial_value=np.float32(p.M3.imag),
                         trainable=True, name='M3im1')
    self.M3im2 = tf.Variable(initial_value=np.float32(p.M3.imag),
                         trainable=True, name='M3im2')

    self.Phire1 = tf.Variable(initial_value=np.float32(p.Phi.real),
                         trainable=True, name='Phire1')
    self.Phire2 = tf.Variable(initial_value=np.float32(p.Phi.real),
                         trainable=True, name='Phire2')
    self.Phiim1 = tf.Variable(initial_value=np.float32(p.Phi.imag),
                         trainable=True, name='Phiim1')
    self.Phiim2 = tf.Variable(initial_value=np.float32(p.Phi.imag),
                         trainable=True, name='Phiim2')

    # self.M3 = [tf.Variable(initial_value=np.float32(M3.real),
    #                      trainable=True, name='M3re1'),\
    #            tf.Variable(initial_value=np.float32(M3.real),
    #                      trainable=True, name='M3re2'), \
    #            tf.Variable(initial_value=np.float32(M3.imag),
    #                      trainable=True, name='M3im1'), \
    #            tf.Variable(initial_value=np.float32(M3.imag),
    #                      trainable=True, name='M3im2')]

    # self.M2 = [tf.Variable(initial_value=np.float32(M2.real),
    #                      trainable=True, name='M2re1'),\
    #            tf.Variable(initial_value=np.float32(M2.real),
    #                      trainable=True, name='M2re2'),\
    #            tf.Variable(initial_value=np.float32(M2.imag),
    #                      trainable=True, name='M2im1'),\
    #            tf.Variable(initial_value=np.float32(M2.imag),
    #                      trainable=True, name='M2im2')]

    # self.Phi = [tf.Variable(initial_value=np.float32(Phi.real),
    #                      trainable=True, name='Phire1'),\
    #             tf.Variable(initial_value=np.float32(Phi.real),
    #                      trainable=True, name='Phire2'),\
    #             tf.Variable(initial_value=np.float32(Phi.imag),
    #                      trainable=True, name='Phiim1'),\
    #             tf.Variable(initial_value=np.float32(Phi.imag),
    #                      trainable=True, name='Phiim2')]

    # self.M3 = [self.M3re1, self.M3re2, self.M3im1, self.M3im2]
    # self.M2 = [self.M2re1, self.M2re2, self.M2im1, self.M2im2]
    # self.Phi = [self.Phire1, self.Phire2, self.Phiim1, self.Phiim2]
    
    # self.c = tf.Variable(initial_value=self.c0, trainable=True, name='c')
    # self.d = tf.Variable(initial_value=self.d0, trainable=True, name='d')

  def prox_l2_norm(self, lamb, vre, vim):
    # import pdb; pdb.set_trace()
    v = tf.sqrt(vre**2 + vim**2)
    const = tf.maximum(0., 1-lamb/tf.norm(v,axis=2))[:,:,None]
    Sre = const * vre
    Sim = const * vim
    return Sre, Sim

  def prox_l2_norm_w_regularization(self, lamb, rho, cre, cim, vre, vim):
    lamb_tilde = lamb/(1+lamb*rho)
    return self.prox_l2_norm(lamb_tilde, lamb_tilde/lamb * vre + rho*lamb_tilde*cre, lamb_tilde/lamb * vim + rho*lamb_tilde*cim)

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

  def call(self, Yre, Yim, Xre, Xim, Zre, Zim, Xhatre, Xhatim, Ure, Uim):
    # X update
    Dre = Xhatre - 1/self.rho * Ure
    Dim = Xhatim - 1/self.rho * Uim

    Gre = 2*(tf.matmul(self.M2re1, Xre) - tf.matmul(self.M2im1, Xim) - (tf.matmul(self.M3re1, Yre) - tf.matmul(self.M3im1, Yim)))
    Gim = 2*(tf.matmul(self.M2im2, Xre) + tf.matmul(self.M2re2, Xim) - (tf.matmul(self.M3im2, Yre) + tf.matmul(self.M3re2, Yim)))

    Cre = Xre - self.taux/2 * Gre
    Cim = Xim - self.taux/2 * Gim

    Xre, Xim = self.prox_l2_norm_w_regularization(self.mu, self.rho, Dre, Dim, Cre, Cim)

    # Z update
    Fre = (Xre - Xhatre + 1/self.rho * Ure)
    Fim = (Xim - Xhatim + 1/self.rho * Uim)

    Gre = -2 * (tf.matmul(Fre, self.Phire1) - tf.matmul(Fim, -self.Phiim1))
    Gim = -2 * (tf.matmul(Fre, -self.Phiim2) + tf.matmul(Fim, self.Phire2))

    Zre, Zim = self.soft_threshold_complex(Zre - self.tauz/2 * Gre, Zim - self.tauz/2 * Gim, self.sigma)

    Xhatre = tf.matmul(Zre, tf.transpose(self.Phire1)) - tf.matmul(Zim, tf.transpose(self.Phiim1))
    Xhatim = tf.matmul(Zre, tf.transpose(self.Phiim2)) + tf.matmul(Zim, tf.transpose(self.Phire2))

    # U update
    Ure = Ure + self.rho*(Xre - Xhatre)
    Uim = Uim + self.rho*(Xim - Xhatim)

    return Xre, Xim, Zre, Zim, Xhatre, Xhatim, Ure, Uim


    # print(Ere.numpy())
    # print(Eim.numpy())
    # print(Gre.numpy())
    # print(Gim.numpy())
    # print(Xre.numpy())
    # print(Xim.numpy())
    # print(Tre.numpy())
    # print(Tim.numpy())

class Stage_reduced(layers.Layer):
  def __init__(self, p, layerid):
    super().__init__()

    M1 = p.A

    self.A = p.A

    self.rho = tf.Variable(initial_value=p.rho, trainable=True, name='rho')
    self.mu = tf.Variable(initial_value=p.mu, trainable=True, name='mu')
    self.taux = tf.Variable(initial_value=p.taux, trainable=True, name='taux')
    self.tauz = tf.Variable(initial_value=p.tauz, trainable=True, name='tauz')
    self.sigma = tf.Variable(initial_value=p.sigma, trainable=True, name='sigma')
    
    self.M2re = tf.Variable(initial_value=np.float32(p.M2.real),
                         trainable=True, name='M2re1')
    self.M2im = tf.Variable(initial_value=np.float32(p.M2.imag),
                         trainable=True, name='M2im1')

    self.M3re = tf.Variable(initial_value=np.float32(p.M3.real),
                         trainable=True, name='M3re1')
    self.M3im = tf.Variable(initial_value=np.float32(p.M3.imag),
                         trainable=True, name='M3im1')

    self.Phire = tf.Variable(initial_value=np.float32(p.Phi.real),
                         trainable=True, name='Phire1')
    self.Phiim = tf.Variable(initial_value=np.float32(p.Phi.imag),
                         trainable=True, name='Phiim1')

  def prox_l2_norm(self, lamb, vre, vim):
    # import pdb; pdb.set_trace()
    v = tf.sqrt(vre**2 + vim**2)
    const = tf.maximum(0., 1-lamb/tf.norm(v,axis=2))[:,:,None]
    Sre = const * vre
    Sim = const * vim
    return Sre, Sim

  def prox_l2_norm_w_regularization(self, lamb, rho, cre, cim, vre, vim):
    lamb_tilde = lamb/(1+lamb*rho)
    return self.prox_l2_norm(lamb_tilde, lamb_tilde/lamb * vre + rho*lamb_tilde*cre, lamb_tilde/lamb * vim + rho*lamb_tilde*cim)

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

  def call(self, Yre, Yim, Xre, Xim, Zre, Zim, Xhatre, Xhatim, Ure, Uim):
    # X update
    Dre = Xhatre - 1/self.rho * Ure
    Dim = Xhatim - 1/self.rho * Uim

    Gre = 2*(tf.matmul(self.M2re, Xre) - tf.matmul(self.M2im, Xim) - (tf.matmul(self.M3re, Yre) - tf.matmul(self.M3im, Yim)))
    Gim = 2*(tf.matmul(self.M2im, Xre) + tf.matmul(self.M2re, Xim) - (tf.matmul(self.M3im, Yre) + tf.matmul(self.M3re, Yim)))

    Cre = Xre - self.taux/2 * Gre
    Cim = Xim - self.taux/2 * Gim

    Xre, Xim = self.prox_l2_norm_w_regularization(self.mu, self.rho, Dre, Dim, Cre, Cim)

    # Z update
    Fre = (Xre - Xhatre + 1/self.rho * Ure)
    Fim = (Xim - Xhatim + 1/self.rho * Uim)

    Gre = -2 * (tf.matmul(Fre, self.Phire) - tf.matmul(Fim, -self.Phiim))
    Gim = -2 * (tf.matmul(Fre, -self.Phiim) + tf.matmul(Fim, self.Phire))

    Zre, Zim = self.soft_threshold_complex(Zre - self.tauz/2 * Gre, Zim - self.tauz/2 * Gim, self.sigma)

    Xhatre = tf.matmul(Zre, tf.transpose(self.Phire)) - tf.matmul(Zim, tf.transpose(self.Phiim))
    Xhatim = tf.matmul(Zre, tf.transpose(self.Phiim)) + tf.matmul(Zim, tf.transpose(self.Phire))

    # U update
    Ure = Ure + self.rho*(Xre - Xhatre)
    Uim = Uim + self.rho*(Xim - Xhatim)

    return Xre, Xim, Zre, Zim, Xhatre, Xhatim, Ure, Uim
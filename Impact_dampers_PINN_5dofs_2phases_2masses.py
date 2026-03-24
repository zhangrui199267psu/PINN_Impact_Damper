"""
@author: Rui Zhang
"""

import tensorflow as tf
import numpy as np
import time
import math
#from Compute_Jacobian import jacobian
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

class PIPNNs:
    # Initialize the class
    def __init__(self, lb, ub, t0, t, x0_total, xt0_total, y03, yt03, y04, yt04, M, K, D, n_dof, phi, phi1, phi2,
                 layers, hyp_ini_weight_loss, hyp_ini_para, optimizer_LB1, optimizer_LB2):

        self.lb = lb
        self.ub = ub
        
        self.t = t
        self.t0= t0
        self.x0 = x0_total
        self.xt0 = xt0_total
        self.y03 = y03
        self.yt03 = yt03
        self.y04 = y04
        self.yt04 = yt04

        self.M = tf.constant(M, dtype=tf.float32)
        self.K = tf.constant(K, dtype=tf.float32)
        self.phi = phi
        self.phi1 = phi1
        self.phi2 = phi2
    
        self.D = tf.constant(D, dtype=tf.float32)  
        
        # Initialize weights for losses
        self.beta_icx = hyp_ini_weight_loss[0]
        self.beta_fx = hyp_ini_weight_loss[1]
        self.beta_f = hyp_ini_weight_loss[2]

        self.n_dof = n_dof
        
        # Initialize impact times for each DOF
        #self.lambda_impacts = [tf.Variable(initial_value, dtype=tf.float32) 
        #                       for _ in range(n_dofs)]
        
        self.lambda_1 = tf.Variable(hyp_ini_para[0], dtype=tf.float32)
        self.lambda_2 = tf.Variable(hyp_ini_para[0], dtype=tf.float32)


        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf Placeholders

        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.xt0_tf = tf.placeholder(tf.float32, shape=[None, self.xt0.shape[1]])

        # tf Graphs
        self.x_pred, self.xt_pred, self.xtt_pred = self.net_u(self.t_tf)
        self.x0_pred, self.xt0_pred, _ = self.net_u(self.t0_tf)
        self.fx_pred, _,_= self.net_f(self.t_tf)
        _, self.f1_pred,_= self.net_f(tf.reshape(self.lambda_1, [1, 1]))
        _, _, self.f2_pred= self.net_f(tf.reshape(self.lambda_2, [1, 1]))
                
        # Loss# Loss
        self.loss_icx = tf.reduce_mean(tf.square(self.x0_pred-self.x0_tf)) + tf.reduce_mean(tf.square(self.xt0_pred-self.xt0_tf))
        self.loss_fx = tf.reduce_mean(tf.square(self.fx_pred))
        self.loss_f = tf.reduce_mean(tf.square(self.f1_pred)) +  tf.reduce_mean(tf.square(self.f2_pred)) 

        self.loss = self.beta_icx * self.loss_icx + self.beta_fx * self.loss_fx

        # Optimizers
        if optimizer_LB1:
            self.optimizer1 = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
            
        if optimizer_LB2:
            self.optimizer2 = tf.contrib.opt.ScipyOptimizerInterface(self.loss_f,
                                                                var_list=[self.lambda_1,self.lambda_2],
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam1 = tf.train.AdamOptimizer()
        self.train_op_Adam1 = self.optimizer_Adam1.minimize(self.loss)
        
        self.optimizer_Adam2 = tf.train.AdamOptimizer()
        self.train_op_Adam2 = self.optimizer_Adam2.minimize(self.loss_f, var_list=[self.lambda_1,self.lambda_2])

        # Loss logger
        self.loss_log = []
        self.loss_icx_log = []
        self.loss_fx_log = []
        self.loss_f_log = []

        # parameters logger
        self.lambda_1_log = []
        self.lambda_2_log = []

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)

        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)

        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, t):
        x = self.neural_net(t, self.weights, self.biases)
                
        # Computing gradient for each component
        x_t = [tf.gradients(x[:, i], t)[0] for i in range(self.n_dof)]
        x_tt = [tf.gradients(x_t[i], t)[0] for i in range(self.n_dof)]
    
        x_t = tf.concat(x_t, axis=1)
        x_tt = tf.concat(x_tt, axis=1)
        
        return x, x_t, x_tt
        
    def net_f(self, t):
        x = self.neural_net(t, self.weights, self.biases)
                
        # Computing gradient for each component
        x_t = [tf.gradients(x[:, i], t)[0] for i in range(self.n_dof)]
        x_tt = [tf.gradients(x_t[i], t)[0] for i in range(self.n_dof)]
    
        x_t = tf.concat(x_t, axis=1)
        x_tt = tf.concat(x_tt, axis=1)
        
        F = tf.transpose(tf.concat([tf.zeros_like(t), tf.zeros_like(t),tf.zeros_like(t),tf.zeros_like(t), self.phi1 * tf.sin(self.phi2*np.pi*(t+self.phi))], axis=1))
        
        # Compute MX'' + CX' + KX - F
        fx = tf.matmul(self.M, x_tt, transpose_b=True) + tf.matmul(self.K, x, transpose_b=True) - F
            
        y3 = self.yt03 * t + self.y03
        y4 = self.yt04 * t + self.y04
        
        f1 = tf.math.abs(x[:, 2:3]-y3)-self.D
        f2 = tf.math.abs(x[:, 3:4]-y4)-self.D
        
        return fx, f1, f2
        
    def callback1(self, loss, loss_icx, loss_fx):

        # Store loss
        self.loss_log.append(loss)
        self.loss_icx_log.append(loss_icx)

        print('Loss: %e, Loss_icx: %.5e, Loss_fx: %.5e'
              % (loss, loss_icx, loss_fx))

    def train1(self, nIter, optimizer_LB1):

        tf_dict = {self.t_tf: self.t,
                   self.t0_tf: self.t0,
                   self.x0_tf: self.x0,
                   self.xt0_tf: self.xt0}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam1, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time

                loss_value = self.sess.run(self.loss, tf_dict)
                loss_icx_value = self.sess.run(self.loss_icx, tf_dict)
                loss_fx_value = self.sess.run(self.loss_fx, tf_dict)

                # Store loss
                self.loss_log.append(loss_value)
                self.loss_icx_log.append(loss_icx_value)
                self.loss_fx_log.append(loss_fx_value)

                print('It: %d, Loss: %.3e, Loss_icx: %.3e,  Loss_fx: %.3e, Time: %.2f' %
                    (it, loss_value, loss_icx_value, loss_fx_value,  elapsed))
                start_time = time.time()

        if optimizer_LB1:
            self.optimizer1.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.loss_icx, self.loss_fx],
                                loss_callback=self.callback1)
        
    def callback2(self, loss_f, lambda_1, lambda_2):

        # Store loss
        self.loss_f_log.append(loss_f)

        # Store lambda
        self.lambda_1_log.append(lambda_1)
        self.lambda_2_log.append(lambda_2)

        print('Loss_f: %.5e, l1: %.5f, l2: %.5f'
              % (loss_f, lambda_1, lambda_2))

    def train2(self, nIter, optimizer_LB2):

        tf_dict = {self.t_tf: self.t,
                   self.t0_tf: self.t0,
                   self.x0_tf: self.x0,
                   self.xt0_tf: self.xt0}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam2, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time

                loss_f_value = self.sess.run(self.loss_f, tf_dict)

                # Store loss
                self.loss_f_log.append(loss_f_value)

                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)

                # Store lambda
                self.lambda_1_log.append(lambda_1_value)
                self.lambda_2_log.append(lambda_2_value)

                print('It: %d,  Loss_f: %.3e, Lambda_1: %.3f,  Lambda_2: %.3f, Time: %.2f' %
                    (it, loss_f_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()

        if optimizer_LB2:
            self.optimizer2.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss_f,
                                         self.lambda_1,
                                         self.lambda_2],
                                loss_callback=self.callback2)

    def predict(self, t):

        tf_dict = {self.t_tf: t}

        x_star = self.sess.run(self.x_pred, tf_dict)
        xt_star = self.sess.run(self.xt_pred, tf_dict)
        xtt_star = self.sess.run(self.xtt_pred, tf_dict)

        return x_star, xt_star, xtt_star
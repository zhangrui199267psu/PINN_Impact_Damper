"""
@author: Rui Zhang
"""

import os
import time
import math
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # CPU:-1; GPU0: 0; GPU1: 1

np.random.seed(1234)
tf.set_random_seed(1234)


class PIPNNs:
    # Initialize the class
    def __init__(self, lb, ub, t0, t,
                 x0_total, xt0_total, y0, yt0,
                 M, K, D, n_dof,
                 phi, phi1, phi2,
                 layers, hyp_ini_weight_loss, hyp_ini_para,
                 optimizer_LB1=True, optimizer_LB2=True):

        self.lb  = lb.astype(np.float32)
        self.ub  = ub.astype(np.float32)

        self.t   = t.astype(np.float32)
        self.t0  = t0.astype(np.float32)
        self.x0  = x0_total.astype(np.float32)
        self.xt0 = xt0_total.astype(np.float32)

        # scalars or length-1 arrays
        self.y0  = float(y0)
        self.yt0 = float(yt0)

        self.M = tf.constant(M, dtype=tf.float32)
        self.K = tf.constant(K, dtype=tf.float32)
        self.D = tf.constant(D, dtype=tf.float32)

        # excitation parameters as tensors
        self.phi  = tf.constant(phi,  dtype=tf.float32)
        self.phi1 = tf.constant(phi1, dtype=tf.float32)
        self.phi2 = tf.constant(phi2, dtype=tf.float32)

        self.n_dof = n_dof

        # loss weights
        self.beta_icx = float(hyp_ini_weight_loss[0])
        self.beta_fx  = float(hyp_ini_weight_loss[1])
        self.beta_f   = float(hyp_ini_weight_loss[2])

        # impact time parameter
        self.lambda_1 = tf.Variable(hyp_ini_para[0], dtype=tf.float32, name="lambda_1")

        # Initialize NN
        self.layers  = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # placeholders
        self.t_tf   = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.t0_tf  = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.x0_tf  = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.xt0_tf = tf.placeholder(tf.float32, shape=[None, self.xt0.shape[1]])

        # forward passes
        self.x_pred, self.xt_pred, self.xtt_pred = self.net_u(self.t_tf)
        self.x0_pred, self.xt0_pred, _ = self.net_u(self.t0_tf)

        self.fx_pred, _ = self.net_f(self.t_tf)
        _, self.f1_pred = self.net_f(tf.reshape(self.lambda_1, [1, 1]))

        # losses
        self.loss_icx = (tf.reduce_mean(tf.square(self.x0_pred - self.x0_tf)) +
                         tf.reduce_mean(tf.square(self.xt0_pred - self.xt0_tf)))
        self.loss_fx  = tf.reduce_mean(tf.square(self.fx_pred))
        self.loss_f   = tf.reduce_mean(tf.square(self.f1_pred))

        # main loss (option: include or exclude beta_f * loss_f)
        self.loss = (self.beta_icx * self.loss_icx +
                     self.beta_fx  * self.loss_fx
                     # + self.beta_f   * self.loss_f   # enable if desired
                     )

        # Optimizers
        if optimizer_LB1:
            self.optimizer1 = tf.contrib.opt.ScipyOptimizerInterface(
                self.loss,
                method="L-BFGS-B",
                options={
                    "maxiter": 50000,
                    "maxfun": 50000,
                    "maxcor": 50,
                    "maxls": 50,
                    "ftol": 1.0 * np.finfo(float).eps
                })

        if optimizer_LB2:
            self.optimizer2 = tf.contrib.opt.ScipyOptimizerInterface(
                self.loss_f,
                var_list=[self.lambda_1],
                method="L-BFGS-B",
                options={
                    "maxiter": 50000,
                    "maxfun": 50000,
                    "maxcor": 50,
                    "maxls": 50,
                    "ftol": 1.0 * np.finfo(float).eps
                })

        self.optimizer_Adam1 = tf.train.AdamOptimizer()
        self.train_op_Adam1  = self.optimizer_Adam1.minimize(self.loss)

        self.optimizer_Adam2 = tf.train.AdamOptimizer()
        self.train_op_Adam2  = self.optimizer_Adam2.minimize(self.loss_f,
                                                             var_list=[self.lambda_1])

        # logs
        self.loss_log      = []
        self.loss_icx_log  = []
        self.loss_fx_log   = []
        self.loss_f_log    = []
        self.lambda_1_log  = []

        # session
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    # ---------------- NN utils ----------------
    def initialize_NN(self, layers):
        weights, biases = [], []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim, out_dim = size[0], size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                                      stddev=xavier_stddev),
                           dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            H = tf.tanh(tf.matmul(H, weights[l]) + biases[l])
        Y = tf.matmul(H, weights[-1]) + biases[-1]
        return Y

    # ---------------- forward models ----------------
    def net_u(self, t):
        x = self.neural_net(t, self.weights, self.biases)
        x_t_list  = [tf.gradients(x[:, i], t)[0] for i in range(self.n_dof)]
        x_tt_list = [tf.gradients(x_t_list[i], t)[0] for i in range(self.n_dof)]
        x_t  = tf.concat(x_t_list,  axis=1)
        x_tt = tf.concat(x_tt_list, axis=1)
        return x, x_t, x_tt

    def net_f(self, t):
        x = self.neural_net(t, self.weights, self.biases)
        x_t_list  = [tf.gradients(x[:, i], t)[0] for i in range(self.n_dof)]
        x_tt_list = [tf.gradients(x_t_list[i], t)[0] for i in range(self.n_dof)]
        x_t  = tf.concat(x_t_list,  axis=1)
        x_tt = tf.concat(x_tt_list, axis=1)

        # external force, only 5th DOF excited
        zero = tf.zeros_like(t)
        F = tf.concat(
            [zero, zero, zero, zero,
             self.phi1 * tf.sin(self.phi2 * math.pi * (t + self.phi))],
            axis=1
        )
        F = tf.transpose(F)  # shape (n_dof, N)

        fx = tf.matmul(self.M, x_tt, transpose_b=True) + \
             tf.matmul(self.K, x,    transpose_b=True) - F

        # contact condition at DOF index 3
        y = self.yt0 * t + self.y0  # same shape as t
        f = tf.abs(x[:, 3:4] - y) - self.D
        return fx, f

    # ---------------- training ----------------
    def callback1(self, loss, loss_icx, loss_fx):
        self.loss_log.append(loss)
        self.loss_icx_log.append(loss_icx)
        self.loss_fx_log.append(loss_fx)
        print("Loss: %e, Loss_icx: %.5e, Loss_fx: %.5e"
              % (loss, loss_icx, loss_fx))

    def train(self, nIter, optimizer_LB1=True):
        tf_dict = {
            self.t_tf:   self.t,
            self.t0_tf:  self.t0,
            self.x0_tf:  self.x0,
            self.xt0_tf: self.xt0
        }
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam1, tf_dict)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value, loss_icx_value, loss_fx_value = \
                    self.sess.run([self.loss, self.loss_icx, self.loss_fx], tf_dict)
                self.loss_log.append(loss_value)
                self.loss_icx_log.append(loss_icx_value)
                self.loss_fx_log.append(loss_fx_value)
                print("It: %d, Loss: %.3e, Loss_icx: %.3e, Loss_fx: %.3e, Time: %.2f"
                      % (it, loss_value, loss_icx_value, loss_fx_value, elapsed))
                start_time = time.time()

        if optimizer_LB1:
            self.optimizer1.minimize(
                self.sess,
                feed_dict=tf_dict,
                fetches=[self.loss, self.loss_icx, self.loss_fx],
                loss_callback=self.callback1
            )

    def callback2(self, loss_f, lambda_1):
        self.loss_f_log.append(loss_f)
        self.lambda_1_log.append(lambda_1)
        print("Loss_f: %.5e, lambda_1: %.5f" % (loss_f, lambda_1))

    def train2(self, nIter, optimizer_LB2=True):
        tf_dict = {
            self.t_tf:   self.t,
            self.t0_tf:  self.t0,
            self.x0_tf:  self.x0,
            self.xt0_tf: self.xt0
        }
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam2, tf_dict)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_f_value, lambda_1_value = \
                    self.sess.run([self.loss_f, self.lambda_1], tf_dict)
                self.loss_f_log.append(loss_f_value)
                self.lambda_1_log.append(lambda_1_value)
                print("It: %d, Loss_f: %.3e, lambda_1: %.3f, Time: %.2f"
                      % (it, loss_f_value, lambda_1_value, elapsed))
                start_time = time.time()

        if optimizer_LB2:
            self.optimizer2.minimize(
                self.sess,
                feed_dict=tf_dict,
                fetches=[self.loss_f, self.lambda_1],
                loss_callback=self.callback2
            )

    # ---------------- prediction ----------------
    def predict(self, t):
        t = t.astype(np.float32)
        tf_dict = {self.t_tf: t}
        x_star, xt_star, xtt_star = self.sess.run(
            [self.x_pred, self.xt_pred, self.xtt_pred], tf_dict
        )
        return x_star, xt_star, xtt_star

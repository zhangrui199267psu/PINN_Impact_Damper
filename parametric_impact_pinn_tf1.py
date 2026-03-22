
"""
Parametric PINN for a single-unit impact damper / meta-impactor
---------------------------------------------------------------
This extends the original TF1-style PINN so that the network learns a
parametric solution operator:

    (t, mx, my, k, c, D, y0, yt0, x0, xt0) -> x(t)

The impact time is treated as an additional learnable variable lambda_1
for each training sample/case.

Notes
-----
1) This is a direct parametric extension of the original attached code.
2) It is still a single-segment / single-impact PINN, which is the cleanest
   starting point for building the full sequential parametric framework.
3) For the lattice version, replace the scalar ODE residual with the vector
   residual M X_tt + C X_t + K X = F and expand the outputs accordingly.
4) Written in TF1-compatible style to stay close to the original code.

Author: OpenAI for Rui Zhang
"""

import os
import time
import numpy as np
import tensorflow as tf

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # CPU:-1; GPU0:0/1 depending on machine

np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

# TF1 compatibility
tf.compat.v1.disable_eager_execution()

class SequentialParametricImpactPINN(object):
    """
    Sequential wrapper around ParametricImpactPINN for repeated impacts.
    One parameter case at a time.
    """

    def __init__(
        self,
        mx, my, k, c, D, r,
        layers,
        hyp_ini_weight_loss=(1.0, 1.0, 10.0),
        optimizer_LB=True,
    ):
        self.mx = float(mx)
        self.my = float(my)
        self.k = float(k)
        self.c = float(c)
        self.D = float(D)
        self.r = float(r)

        self.layers = layers
        self.hyp_ini_weight_loss = hyp_ini_weight_loss
        self.optimizer_LB = optimizer_LB

        self.impact_times = []
        self.models = []
        self.segment_data = []

    def impact_update(self, xt_minus, yt_minus):
        """
        Velocity update from restitution + momentum conservation.
        """
        mx = self.mx
        my = self.my
        r = self.r

        A = np.array([[mx, my],
                      [1.0, -1.0]], dtype=np.float32)

        B = np.array([[mx, my],
                      [-r,  r]], dtype=np.float32)

        V_minus = np.array([[float(xt_minus)],
                            [float(yt_minus)]], dtype=np.float32)

        V_plus = np.linalg.solve(A, B @ V_minus)
        xt_plus = V_plus[0:1, :]
        yt_plus = V_plus[1:2, :]
        return xt_plus, yt_plus

    def run(
        self,
        x0,
        xt0,
        y0,
        yt0,
        n_impact,
        T_vector,
        nIter_vector,
        hyp_ini_para_vector,
        num_time_points=1000,
    ):
        """
        Sequentially train each impact segment.
        """
        # current state
        x0 = np.asarray(x0, dtype=np.float32).reshape(1, 1)
        xt0 = np.asarray(xt0, dtype=np.float32).reshape(1, 1)
        y0 = np.asarray(y0, dtype=np.float32).reshape(1, 1)
        yt0 = np.asarray(yt0, dtype=np.float32).reshape(1, 1)

        time_offset = 0.0

        all_t = []
        all_x = []
        all_xt = []
        all_xtt = []
        all_y = []
        all_yt = []
        all_ytt = []

        for j in range(n_impact):
            T = float(T_vector[j])
            nIter = int(nIter_vector[j])
            hyp_ini_para = float(hyp_ini_para_vector[j])

            t0 = np.array([[0.0]], dtype=np.float32)
            t_r = np.linspace(0.0, T, num_time_points).reshape(-1, 1).astype(np.float32)

            # 10 input channels:
            # [t, mx, my, k, c, D, y0, yt0, x0, xt0]
            lb = np.array([0.0,  0.1, 0.1, 0.1, 0.0, 0.1, -5.0, -5.0, -5.0, -5.0], dtype=np.float32)
            ub = np.array([T,   10.0, 10.0, 10.0, 5.0, 10.0,  5.0,  5.0,  5.0,  5.0], dtype=np.float32)

            model = ParametricImpactPINN(
                lb=lb,
                ub=ub,
                t0=t0,
                t_r=t_r,
                x0=x0,
                xt0=xt0,
                y0=y0,
                yt0=yt0,
                mx=np.array([[self.mx]], dtype=np.float32),
                my=np.array([[self.my]], dtype=np.float32),
                k=np.array([[self.k]], dtype=np.float32),
                c=np.array([[self.c]], dtype=np.float32),
                D=np.array([[self.D]], dtype=np.float32),
                layers=self.layers,
                hyp_ini_weight_loss=self.hyp_ini_weight_loss,
                hyp_ini_para=hyp_ini_para,
                optimizer_LB=self.optimizer_LB,
            )

            model.train(nIter=nIter, optimizer_LB=self.optimizer_LB, print_every=max(1, nIter // 10))

            lambda_val = model.predict_lambda()
            t_impact = float(lambda_val[0, 0])

            # state at impact
            x1, xt1, xtt1 = model.predict(
                np.array([[t_impact]], dtype=np.float32),
                x0=x0, xt0=xt0, y0=y0, yt0=yt0,
                mx=self.mx, my=self.my, k=self.k, c=self.c, D=self.D
            )

            y1 = yt0 * t_impact + y0
            yt1 = yt0

            # full response on this segment
            t_seg = np.linspace(0.0, t_impact, num_time_points + 1).reshape(-1, 1).astype(np.float32)
            x_seg, xt_seg, xtt_seg = model.predict(
                t_seg,
                x0=x0, xt0=xt0, y0=y0, yt0=yt0,
                mx=self.mx, my=self.my, k=self.k, c=self.c, D=self.D
            )
            y_seg = yt0 * t_seg + y0
            yt_seg = yt0 * np.ones_like(t_seg)
            ytt_seg = np.zeros_like(t_seg)

            all_t.append(t_seg + time_offset)
            all_x.append(x_seg)
            all_xt.append(xt_seg)
            all_xtt.append(xtt_seg)
            all_y.append(y_seg)
            all_yt.append(yt_seg)
            all_ytt.append(ytt_seg)

            self.impact_times.append(t_impact)
            self.models.append(model)
            self.segment_data.append({
                "segment": j + 1,
                "t_impact": t_impact,
                "x_minus": x1,
                "xt_minus": xt1,
                "y_minus": y1,
                "yt_minus": yt1,
            })

            # impact update
            xt_plus, yt_plus = self.impact_update(xt1, yt1)

            # new initial state for next segment
            x0 = x1
            xt0 = xt_plus
            y0 = y1
            yt0 = yt_plus

            time_offset += t_impact

        return {
            "time": np.vstack(all_t),
            "x": np.vstack(all_x),
            "xt": np.vstack(all_xt),
            "xtt": np.vstack(all_xtt),
            "y": np.vstack(all_y),
            "yt": np.vstack(all_yt),
            "ytt": np.vstack(all_ytt),
            "impact_times": np.array(self.impact_times, dtype=np.float32).reshape(-1, 1),
        }

class ParametricImpactPINN(object):
    """
    Parametric PINN for an undamped/damped SDOF impact damper segment.

    Inputs to the network:
        [t, mx, my, k, c, D, y0, yt0, x0, xt0]

    Output:
        x(t)

    Governing physics between impacts:
        mx * x_tt + c * x_t + k * x = 0

    Internal mass trajectory within a segment:
        y(t) = y0 + yt0 * t

    Impact constraint at learnable impact time lambda_1:
        |x(lambda_1) - y(lambda_1)| = D
    """

    def __init__(
        self,
        lb,
        ub,
        t0,
        t_r,
        x0,
        xt0,
        y0,
        yt0,
        mx,
        my,
        k,
        c,
        D,
        layers,
        hyp_ini_weight_loss=(1.0, 1.0, 1.0),
        hyp_ini_para=0.5,
        optimizer_LB=True,
    ):
        # -----------------------------
        # Store training data (numpy)
        # -----------------------------
        self.lb = np.asarray(lb, dtype=np.float32).reshape(1, -1)
        self.ub = np.asarray(ub, dtype=np.float32).reshape(1, -1)

        self.t0 = np.asarray(t0, dtype=np.float32).reshape(-1, 1)      # usually zeros
        self.t_r = np.asarray(t_r, dtype=np.float32).reshape(-1, 1)    # collocation times

        self.x0 = np.asarray(x0, dtype=np.float32).reshape(-1, 1)
        self.xt0 = np.asarray(xt0, dtype=np.float32).reshape(-1, 1)
        self.y0 = np.asarray(y0, dtype=np.float32).reshape(-1, 1)
        self.yt0 = np.asarray(yt0, dtype=np.float32).reshape(-1, 1)

        self.mx = np.asarray(mx, dtype=np.float32).reshape(-1, 1)
        self.my = np.asarray(my, dtype=np.float32).reshape(-1, 1)
        self.k = np.asarray(k, dtype=np.float32).reshape(-1, 1)
        self.c = np.asarray(c, dtype=np.float32).reshape(-1, 1)
        self.D = np.asarray(D, dtype=np.float32).reshape(-1, 1)

        self.n_cases = self.x0.shape[0]
        assert self.t0.shape[0] == self.n_cases, "t0 must have one row per case."
        for arr_name in ["xt0", "y0", "yt0", "mx", "my", "k", "c", "D"]:
            arr = getattr(self, arr_name)
            assert arr.shape[0] == self.n_cases, f"{arr_name} must have one row per case."

        # ---------------------------------------
        # Collocation points expanded by case id
        # ---------------------------------------
        # One set of collocation times per case
        self.n_r = self.t_r.shape[0]
        self.case_ids_r = np.repeat(np.arange(self.n_cases), self.n_r).reshape(-1, 1).astype(np.int32)
        self.t_r_full = np.tile(self.t_r, (self.n_cases, 1)).astype(np.float32)

        # Parameters repeated over collocation points
        self.mx_r = np.repeat(self.mx, self.n_r, axis=0)
        self.my_r = np.repeat(self.my, self.n_r, axis=0)
        self.k_r = np.repeat(self.k, self.n_r, axis=0)
        self.c_r = np.repeat(self.c, self.n_r, axis=0)
        self.D_r = np.repeat(self.D, self.n_r, axis=0)
        self.y0_r = np.repeat(self.y0, self.n_r, axis=0)
        self.yt0_r = np.repeat(self.yt0, self.n_r, axis=0)
        self.x0_r = np.repeat(self.x0, self.n_r, axis=0)
        self.xt0_r = np.repeat(self.xt0, self.n_r, axis=0)
       
        self.t0_r = np.repeat(self.t0, self.n_r, axis=0)
        self.x0_r = np.repeat(self.x0, self.n_r, axis=0)
        self.xt0_r = np.repeat(self.xt0, self.n_r, axis=0)

        # ---------------------------------------
        # Loss weights
        # ---------------------------------------
        self.beta_icx = float(hyp_ini_weight_loss[0])
        self.beta_fx = float(hyp_ini_weight_loss[1])
        self.beta_f = float(hyp_ini_weight_loss[2])

        # ---------------------------------------
        # Learnable impact time for each case
        # ---------------------------------------
        lambda_init = np.ones((self.n_cases, 1), dtype=np.float32) * float(hyp_ini_para)
        self.lambda_1 = tf.Variable(lambda_init, dtype=tf.float32, name="lambda_1")

        # ---------------------------------------
        # Neural network
        # ---------------------------------------
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # ---------------------------------------
        # Placeholders
        # ---------------------------------------
        self.t_r_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.case_ids_r_tf = tf.compat.v1.placeholder(tf.int32, shape=[None, 1])

        self.t0_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x0_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.xt0_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.y0_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.yt0_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.mx_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.my_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.k_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.c_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.D_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        # ---------------------------------------
        # Build graph
        # ---------------------------------------
        # Initial-condition predictions: one per case
        self.X0_in = self.build_input(
            self.t0_tf, self.mx_tf, self.my_tf, self.k_tf, self.c_tf,
            self.D_tf, self.y0_tf, self.yt0_tf, self.x0_tf, self.xt0_tf
        )
        self.x0_pred, self.xt0_pred, self.xtt0_pred = self.net_u(self.X0_in)

        # Residual points
        self.X_r_in = self.build_input(
            self.t_r_tf, self.mx_tf, self.my_tf, self.k_tf, self.c_tf,
            self.D_tf, self.y0_tf, self.yt0_tf, self.x0_tf, self.xt0_tf
        )
        self.x_r_pred, self.xt_r_pred, self.xtt_r_pred = self.net_u(self.X_r_in)

        # ODE residual at collocation points
        self.fx_r_pred = self.mx_tf * self.xtt_r_pred + self.c_tf * self.xt_r_pred + self.k_tf * self.x_r_pred

        # Gather learnable impact time for each residual sample's case
        lambda_r = tf.gather(self.lambda_1, tf.squeeze(self.case_ids_r_tf, axis=1))
        y_lambda_r = self.yt0_tf * lambda_r + self.y0_tf
        X_lambda_in = self.build_input(
            lambda_r, self.mx_tf, self.my_tf, self.k_tf, self.c_tf,
            self.D_tf, self.y0_tf, self.yt0_tf, self.x0_tf, self.xt0_tf
        )
        x_lambda_r, _, _ = self.net_u(X_lambda_in)
        self.f_impact_r_pred = tf.abs(x_lambda_r - y_lambda_r) - self.D_tf

        # ---------------------------------------
        # Losses
        # ---------------------------------------
        self.loss_icx = tf.reduce_mean(tf.square(self.x0_pred - self.x0_tf)) + \
                        tf.reduce_mean(tf.square(self.xt0_pred - self.xt0_tf))

        self.loss_fx = tf.reduce_mean(tf.square(self.fx_r_pred))
        self.loss_f = tf.reduce_mean(tf.square(self.f_impact_r_pred))

        # Optional regularization: keep lambda inside [t_min, t_max]
        t_min = tf.reduce_min(self.t_r_tf)
        t_max = tf.reduce_max(self.t_r_tf)
        self.loss_lambda_box = tf.reduce_mean(tf.square(tf.nn.relu(t_min - self.lambda_1))) + \
                               tf.reduce_mean(tf.square(tf.nn.relu(self.lambda_1 - t_max)))

        self.loss = self.beta_icx * self.loss_icx + \
                    self.beta_fx * self.loss_fx + \
                    self.beta_f * self.loss_f + \
                    1.0 * self.loss_lambda_box

        # ---------------------------------------
        # Optimizers
        # ---------------------------------------
        if optimizer_LB:
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                self.loss,
                method="L-BFGS-B",
                options={
                    "maxiter": 50000,
                    "maxfun": 50000,
                    "maxcor": 50,
                    "maxls": 50,
                    "ftol": 1.0 * np.finfo(float).eps,
                },
            )

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # ---------------------------------------
        # Logs
        # ---------------------------------------
        self.loss_log = []
        self.loss_icx_log = []
        self.loss_fx_log = []
        self.loss_f_log = []
        self.loss_lambda_box_log = []
        self.lambda_1_log = []

        # ---------------------------------------
        # Session
        # ---------------------------------------
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
            )
        )
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    # ------------------------------------------------------------------
    # Network utilities
    # ------------------------------------------------------------------
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),
            dtype=tf.float32
        )

    def build_input(self, t, mx, my, k, c, D, y0, yt0, x0, xt0):
        return tf.concat([t, mx, my, k, c, D, y0, yt0, x0, xt0], axis=1)

    def neural_net(self, X, weights, biases):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        num_layers = len(weights) + 1
        for l in range(num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, X):
        # time is the first column of X
        x = self.neural_net(X, self.weights, self.biases)
        x_t = tf.gradients(x, X)[0][:, 0:1]
        x_tt = tf.gradients(x_t, X)[0][:, 0:1]
        return x, x_t, x_tt

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def callback(self, loss, loss_icx, loss_fx, loss_f, loss_lambda_box, lambda_1):
        self.loss_log.append(loss)
        self.loss_icx_log.append(loss_icx)
        self.loss_fx_log.append(loss_fx)
        self.loss_f_log.append(loss_f)
        self.loss_lambda_box_log.append(loss_lambda_box)
        self.lambda_1_log.append(lambda_1.copy())

        print(
            "Loss: %.5e, Loss_icx: %.5e, Loss_fx: %.5e, Loss_f: %.5e, "
            "Loss_lambda_box: %.5e, lambda_1_mean: %.5f"
            % (loss, loss_icx, loss_fx, loss_f, loss_lambda_box, np.mean(lambda_1))
        )

    def _make_tf_dict_for_residual(self):
        return {
            self.t_r_tf: self.t_r_full,
            self.case_ids_r_tf: self.case_ids_r,
            self.mx_tf: self.mx_r,
            self.my_tf: self.my_r,
            self.k_tf: self.k_r,
            self.c_tf: self.c_r,
            self.D_tf: self.D_r,
            self.y0_tf: self.y0_r,
            self.yt0_tf: self.yt0_r,
            self.x0_tf: self.x0_r,
            self.xt0_tf: self.xt0_r,
        }

    def _make_tf_dict_for_initial(self):
        return {
            self.t0_tf: self.t0,
            self.mx_tf: self.mx,
            self.my_tf: self.my,
            self.k_tf: self.k,
            self.c_tf: self.c,
            self.D_tf: self.D,
            self.y0_tf: self.y0,
            self.yt0_tf: self.yt0,
            self.x0_tf: self.x0,
            self.xt0_tf: self.xt0,
        }

    def train(self, nIter=10000, optimizer_LB=True, print_every=100):

        # fully expanded feed dict for the combined graph
        tf_dict_residual = {
            self.t_r_tf: self.t_r_full,
            self.case_ids_r_tf: self.case_ids_r,

            self.t0_tf: self.t0_r,
            self.x0_tf: self.x0_r,
            self.xt0_tf: self.xt0_r,

            self.y0_tf: self.y0_r,
            self.yt0_tf: self.yt0_r,

            self.mx_tf: self.mx_r,
            self.my_tf: self.my_r,
            self.k_tf: self.k_r,
            self.c_tf: self.c_r,
            self.D_tf: self.D_r,
        }

        # case-wise feed dict only for cleaner monitoring of initial-condition loss
        tf_dict_ic = {
            self.t0_tf: self.t0,
            self.x0_tf: self.x0,
            self.xt0_tf: self.xt0,
            self.y0_tf: self.y0,
            self.yt0_tf: self.yt0,
            self.mx_tf: self.mx,
            self.my_tf: self.my,
            self.k_tf: self.k,
            self.c_tf: self.c,
            self.D_tf: self.D,
        }

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict_residual)

            if it % print_every == 0:
                elapsed = time.time() - start_time

                loss_value = self.sess.run(self.loss, tf_dict_residual)
                loss_icx_value = self.sess.run(self.loss_icx, tf_dict_ic)
                loss_fx_value = self.sess.run(self.loss_fx, tf_dict_residual)
                loss_f_value = self.sess.run(self.loss_f, tf_dict_residual)
                loss_lambda_box_value = self.sess.run(self.loss_lambda_box, tf_dict_residual)
                lambda_1_value = self.sess.run(self.lambda_1)

                self.loss_log.append(loss_value)
                self.loss_icx_log.append(loss_icx_value)
                self.loss_fx_log.append(loss_fx_value)
                self.loss_f_log.append(loss_f_value)
                self.loss_lambda_box_log.append(loss_lambda_box_value)
                self.lambda_1_log.append(lambda_1_value.copy())

                print(
                    "It: %d, Loss: %.3e, Loss_icx: %.3e, Loss_fx: %.3e, "
                    "Loss_f: %.3e, Loss_lambda_box: %.3e, Lambda_1_mean: %.3f, Time: %.2f"
                    % (
                        it, loss_value, loss_icx_value, loss_fx_value,
                        loss_f_value, loss_lambda_box_value,
                        float(np.mean(lambda_1_value)), elapsed
                    )
                )
                start_time = time.time()

        if optimizer_LB and hasattr(self, "optimizer"):
            self.optimizer.minimize(
                self.sess,
                feed_dict=tf_dict_residual,
                fetches=[self.loss, self.loss_icx, self.loss_fx, self.loss_f,
                         self.loss_lambda_box, self.lambda_1],
                loss_callback=self.callback
            )
            # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, t_star, x0, xt0, y0, yt0, mx, my, k, c, D):
        """
        Predict x, x_t, x_tt for new parameter cases.

        Parameters
        ----------
        t_star : (N,1) array
            Query times for one case.
        All other arguments : scalar or (1,1) arrays
            Parameters for the new case.

        Returns
        -------
        x_star, xt_star, xtt_star : arrays of shape (N,1)
        """
        t_star = np.asarray(t_star, dtype=np.float32).reshape(-1, 1)

        def _to_case_array(v):
            arr = np.asarray(v, dtype=np.float32).reshape(1, 1)
            return np.repeat(arr, t_star.shape[0], axis=0)

        x0_r = _to_case_array(x0)
        xt0_r = _to_case_array(xt0)
        y0_r = _to_case_array(y0)
        yt0_r = _to_case_array(yt0)
        mx_r = _to_case_array(mx)
        my_r = _to_case_array(my)
        k_r = _to_case_array(k)
        c_r = _to_case_array(c)
        D_r = _to_case_array(D)

        tf_dict = {
            self.t_r_tf: t_star,
            self.mx_tf: mx_r,
            self.my_tf: my_r,
            self.k_tf: k_r,
            self.c_tf: c_r,
            self.D_tf: D_r,
            self.y0_tf: y0_r,
            self.yt0_tf: yt0_r,
            self.x0_tf: x0_r,
            self.xt0_tf: xt0_r,
        }

        x_star = self.sess.run(self.x_r_pred, tf_dict)
        xt_star = self.sess.run(self.xt_r_pred, tf_dict)
        xtt_star = self.sess.run(self.xtt_r_pred, tf_dict)
        return x_star, xt_star, xtt_star

    def predict_lambda(self):
        """Return learned impact times lambda_1 for all training cases."""
        return self.sess.run(self.lambda_1)


if __name__ == "__main__":
    # -----------------------------------------------------------
    # Minimal example with 3 parameter cases
    # -----------------------------------------------------------
    n_cases = 3
    n_r = 200

    # Time domain per segment
    t0 = np.zeros((n_cases, 1), dtype=np.float32)
    t_r = np.linspace(0.0, 2.0, n_r).reshape(-1, 1).astype(np.float32)

    # Initial conditions and parameters for several cases
    x0 = np.zeros((n_cases, 1), dtype=np.float32)
    xt0 = np.zeros((n_cases, 1), dtype=np.float32)
    y0 = np.zeros((n_cases, 1), dtype=np.float32)
    yt0 = np.array([[-1.0], [-0.8], [-1.2]], dtype=np.float32)

    mx = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    my = np.array([[0.3], [0.35], [0.25]], dtype=np.float32)
    k = np.array([[1.0], [1.2], [0.9]], dtype=np.float32)
    c = np.array([[0.0], [0.02], [0.01]], dtype=np.float32)
    D = np.array([[1.0], [1.0], [0.9]], dtype=np.float32)

    # Input = [t, mx, my, k, c, D, y0, yt0, x0, xt0] -> 10 dims
    layers = [10, 64, 64, 64, 1]

    # Lower/upper bounds for normalization (one value per input channel)
    lb = np.array([0.0, 0.5, 0.1, 0.5, 0.0, 0.5, -1.0, -2.0, -1.0, -1.0], dtype=np.float32)
    ub = np.array([2.0, 2.0, 1.0, 2.0, 0.2, 2.0,  1.0,  2.0,  1.0,  1.0], dtype=np.float32)

    model = ParametricImpactPINN(
        lb=lb,
        ub=ub,
        t0=t0,
        t_r=t_r,
        x0=x0,
        xt0=xt0,
        y0=y0,
        yt0=yt0,
        mx=mx,
        my=my,
        k=k,
        c=c,
        D=D,
        layers=layers,
        hyp_ini_weight_loss=(1.0, 1.0, 10.0),
        hyp_ini_para=0.8,
        optimizer_LB=False,  # set True if tf.contrib is available in your TF1 env
    )

    model.train(nIter=2000, optimizer_LB=False, print_every=200)

    # Predict one new case
    t_star = np.linspace(0.0, 2.0, 300).reshape(-1, 1)
    x_star, xt_star, xtt_star = model.predict(
        t_star=t_star,
        x0=0.0, xt0=0.0, y0=0.0, yt0=-1.0,
        mx=1.0, my=0.3, k=1.0, c=0.0, D=1.0
    )

    print("Predicted x(t) shape:", x_star.shape)
    print("Learned lambda values for training cases:")
    print(model.predict_lambda())

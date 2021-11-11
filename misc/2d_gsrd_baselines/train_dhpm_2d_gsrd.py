import numpy as np
import time
import pygmsh
from pyDOE import lhs
import matplotlib
import platform
if platform.system()=='Linux':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
if platform.system()=='Windows':
    from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import pickle
import math
import scipy.io

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;
np.random.seed(1111)
tf.set_random_seed(1111)

class GS_DHPM:
    # Initialize the class
    def __init__(self, Collo, UP, LW, RT, LF, uv_layers, f_layers, lb, ub, uvDir=None, fDir=None):

        # Count for callback function
        self.count = 0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]
        self.t_c = Collo[:, 2:3]
        self.u_c = Collo[:, 3:4]
        self.v_c = Collo[:, 4:5]

        # Upper boundary condition point, free surface
        self.x_UP = UP[:, 0:1]
        self.y_UP = UP[:, 1:2]
        self.t_UP = UP[:, 2:3]

        self.x_LW = LW[:, 0:1]
        self.y_LW = LW[:, 1:2]
        self.t_LW = LW[:, 2:3]

        self.x_LF = LF[:, 0:1]
        self.y_LF = LF[:, 1:2]
        self.t_LF = LF[:, 2:3]

        self.x_RT = RT[:, 0:1]
        self.y_RT = RT[:, 1:2]
        self.t_RT = RT[:, 2:3]

        # Define layers
        self.uv_layers = uv_layers
        self.f_layers = f_layers

        # Initialize NNs
        if uvDir is not None:
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)
        else:
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)

        if fDir is not None:
            self.f_weights, self.f_biases = self.load_NN(fDir, self.f_layers)
        else:
            self.f_weights, self.f_biases = self.initialize_NN(self.f_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])    # Point for postprocessing
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t_c.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        self.t_c_tf = tf.placeholder(tf.float32, shape=[None, self.t_c.shape[1]])
        self.u_c_tf = tf.placeholder(tf.float32, shape=[None, self.u_c.shape[1]])
        self.v_c_tf = tf.placeholder(tf.float32, shape=[None, self.v_c.shape[1]])

        self.x_UP_tf = tf.placeholder(tf.float32, shape=[None, self.x_UP.shape[1]])
        self.y_UP_tf = tf.placeholder(tf.float32, shape=[None, self.y_UP.shape[1]])
        self.t_UP_tf = tf.placeholder(tf.float32, shape=[None, self.t_UP.shape[1]])

        self.x_LW_tf = tf.placeholder(tf.float32, shape=[None, self.x_LW.shape[1]])
        self.y_LW_tf = tf.placeholder(tf.float32, shape=[None, self.y_LW.shape[1]])
        self.t_LW_tf = tf.placeholder(tf.float32, shape=[None, self.t_LW.shape[1]])

        self.x_LF_tf = tf.placeholder(tf.float32, shape=[None, self.x_LF.shape[1]])
        self.y_LF_tf = tf.placeholder(tf.float32, shape=[None, self.y_LF.shape[1]])
        self.t_LF_tf = tf.placeholder(tf.float32, shape=[None, self.t_LF.shape[1]])

        self.x_RT_tf = tf.placeholder(tf.float32, shape=[None, self.x_RT.shape[1]])
        self.y_RT_tf = tf.placeholder(tf.float32, shape=[None, self.y_RT.shape[1]])
        self.t_RT_tf = tf.placeholder(tf.float32, shape=[None, self.t_RT.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred = self.net_uv(self.x_c_tf, self.y_c_tf, self.t_c_tf)
        self.f_pred_u, self.f_pred_v = self.net_f(self.x_c_tf, self.y_c_tf, self.t_c_tf)

        self.u_LF_pred, self.v_LF_pred = self.net_uv(self.x_LF_tf, self.y_LF_tf, self.t_LF_tf)
        self.u_RT_pred, self.v_RT_pred = self.net_uv(self.x_RT_tf, self.y_RT_tf, self.t_RT_tf)
        self.u_LW_pred, self.v_LW_pred = self.net_uv(self.x_LW_tf, self.y_LW_tf, self.t_LW_tf)
        self.u_UP_pred, self.v_UP_pred = self.net_uv(self.x_UP_tf, self.y_UP_tf, self.t_UP_tf)

        self.loss_f_uv = tf.reduce_mean(tf.square(self.f_pred_u)) \
                       + tf.reduce_mean(tf.square(self.f_pred_v))
        self.loss_uv = tf.reduce_mean(tf.square(self.u_pred - self.u_c_tf)) \
                     + tf.reduce_mean(tf.square(self.v_pred - self.v_c_tf))
        self.loss_PBC = tf.reduce_mean(tf.square(self.u_LF_pred - self.u_RT_pred)) \
                      + tf.reduce_mean(tf.square(self.v_LF_pred - self.v_RT_pred)) \
                      + tf.reduce_mean(tf.square(self.u_LW_pred - self.u_UP_pred)) \
                      + tf.reduce_mean(tf.square(self.v_LW_pred - self.v_UP_pred))
        self.loss = 10*self.loss_f_uv + 10*self.loss_uv + self.loss_PBC

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(100*self.loss,
                                                                var_list=self.uv_weights + self.uv_biases
                                                                        + self.f_weights + self.f_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(100*self.loss,
                                                          var_list=self.uv_weights + self.uv_biases + self.f_weights + self.f_biases)

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
            W = self.sine_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)

    def sine_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        w = tf.random_uniform([in_dim, out_dim], minval=-np.sqrt(6/in_dim)/1, maxval=np.sqrt(6/in_dim)/1, dtype=tf.float32)
        return tf.Variable(w, dtype=tf.float32)

    def save_NN_uv(self, fileDir):
        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)
        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            print("Save NN_uv parameters successfully...")

    def save_NN_f(self, fileDir):
        f_weights = self.sess.run(self.f_weights)
        f_biases = self.sess.run(self.f_biases)
        with open(fileDir, 'wb') as f:
            pickle.dump([f_weights, f_biases], f)
            print("Save NN_f parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)
            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights)+1)
            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num], dtype=tf.float32)
                b = tf.Variable(uv_biases[num], dtype=tf.float32)
                weights.append(W)
                biases.append(b)
                print("Load NN parameters successfully...")
        return weights, biases

    def neural_net_uv(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_f(self, terms, weights, biases):
        # No need for input normalization
        num_layers = len(weights) + 1
        H = terms
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, y, t):
        uv = self.neural_net_uv(tf.concat([x, y, t], 1), self.uv_weights, self.uv_biases)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v

    def net_f(self, x, y, t):
        uv = self.neural_net_uv(tf.concat([x, y, t], 1), self.uv_weights, self.uv_biases)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        ut = tf.gradients(u, t)[0]
        vt = tf.gradients(v, t)[0]

        ux = tf.gradients(u, x)[0]
        uxx = tf.gradients(ux, x)[0]
        uy = tf.gradients(u, y)[0]
        uyy = tf.gradients(uy, y)[0]
        lap_u = uxx + uyy

        vx = tf.gradients(v, x)[0]
        vxx = tf.gradients(vx, x)[0]
        vy = tf.gradients(v, y)[0]
        vyy = tf.gradients(vy, y)[0]
        lap_v = vxx + vyy

        # black-box generator of RHS
        N_uv = self.neural_net_f(tf.concat([lap_u, lap_v, u, v], 1), self.f_weights, self.f_biases)

        f_u = ut - N_uv[:, 0:1]
        f_v = vt - N_uv[:, 1:2]

        return f_u, f_v


    def callback(self, loss):
        self.count = self.count + 1
        print('{} th iterations, Loss: {}'.format(self.count, loss))

    def train(self, iter, learning_rate):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.t_c_tf: self.t_c,
                   self.u_c_tf: self.u_c, self.v_c_tf: self.v_c,
                   self.x_UP_tf: self.x_UP, self.y_UP_tf: self.y_UP, self.t_UP_tf: self.t_UP,
                   self.x_LW_tf: self.x_LW, self.y_LW_tf: self.y_LW, self.t_LW_tf: self.t_LW,
                   self.x_LF_tf: self.x_LF, self.y_LF_tf: self.y_LF, self.t_LF_tf: self.t_LF,
                   self.x_RT_tf: self.x_RT, self.y_RT_tf: self.y_RT, self.t_RT_tf: self.t_RT,
                   self.learning_rate: learning_rate}

        for it in range(iter):

            if it % 200 == 0:
                tf_dict[self.learning_rate] = tf_dict[self.learning_rate]*0.99

            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.10e, LR: %.8e' %
                      (it, loss_value, tf_dict[self.learning_rate]))

    def train_bfgs(self):
        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.t_c_tf: self.t_c,
                   self.u_c_tf: self.u_c, self.v_c_tf: self.v_c,
                   self.x_UP_tf: self.x_UP, self.y_UP_tf: self.y_UP, self.t_UP_tf: self.t_UP,
                   self.x_LW_tf: self.x_LW, self.y_LW_tf: self.y_LW, self.t_LW_tf: self.t_LW,
                   self.x_LF_tf: self.x_LF, self.y_LF_tf: self.y_LF, self.t_LF_tf: self.t_LF,
                   self.x_RT_tf: self.x_RT, self.y_RT_tf: self.y_RT, self.t_RT_tf: self.t_RT}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star, t_star):
        u_star = self.sess.run(self.u_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star, self.t_c_tf: t_star})
        v_star = self.sess.run(self.v_pred, {self.x_c_tf: x_star, self.y_c_tf: y_star, self.t_c_tf: t_star})
        return u_star, v_star

    def getloss(self):  # To be updated

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.t_c_tf: self.t_c,
                   self.u_c_tf: self.u_c, self.v_c_tf: self.v_c,
                   self.x_UP_tf: self.x_UP, self.y_UP_tf: self.y_UP, self.t_UP_tf: self.t_UP,
                   self.x_LW_tf: self.x_LW, self.y_LW_tf: self.y_LW, self.t_LW_tf: self.t_LW,
                   self.x_LF_tf: self.x_LF, self.y_LF_tf: self.y_LF, self.t_LF_tf: self.t_LF,
                   self.x_RT_tf: self.x_RT, self.y_RT_tf: self.y_RT, self.t_RT_tf: self.t_RT}

        loss_f_uv = self.sess.run(self.loss_f_uv, tf_dict)
        loss_PBC = self.sess.run(self.loss_PBC, tf_dict)
        loss_uv = self.sess.run(self.loss_uv, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)

        print('loss_f_uv:', loss_f_uv)
        print('loss_PBC:', loss_PBC)
        print('loss_UV:', loss_uv)
        print('loss:', loss)

def add_noise(truth, pec=0.05):
    np.random.seed(66)
    uv = [truth[0:1,:,:,:], truth[1:2,:,:,:]]
    uv_noi = []
    for item in uv:
        R = np.random.normal(0.0, 1.0, size=item.shape)
        std_R = np.std(R)
        std_T = np.std(item)
        noise = R * std_T / std_R * pec
        uv_noi.append(item + noise)
    return np.concatenate(uv_noi, axis=0)


def postProcess(xmin, xmax, ymin, ymax, field=[], s=12, num=0):
    ''' num: Number of time step
    '''
    [x_pred, y_pred, t_star, u_pred, v_pred, u_trth, v_trth] = field
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 9))
    fig.subplots_adjust(hspace=0.25, wspace=0.07)
    ######
    cf = ax[0, 0].scatter(x_pred, y_pred, c=u_pred, alpha=0.9, edgecolors='none', cmap='seismic', marker='s', s=s, vmin=0, vmax=1)
    ax[0, 0].axis('square')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_title(r'$u$-PINN', fontsize=22)
    cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[0, 0], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)
    ##########
    cf = ax[1, 0].scatter(x_pred, y_pred, c=u_trth, alpha=0.9, edgecolors='none', cmap='seismic', marker='s', s=s, vmin=0, vmax=1)
    ax[1, 0].axis('square')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_title(r'$u$-FD', fontsize=22)
    cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[1, 0], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)
    #######
    cf = ax[0, 1].scatter(x_pred, y_pred, c=v_pred, alpha=0.9, edgecolors='none', cmap='seismic', marker='s', s=s, vmin=0, vmax=1)
    ax[0, 1].axis('square')
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_title(r'$v$-PINN', fontsize=22)
    cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[0, 1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)
    #####
    cf = ax[1, 1].scatter(x_star, y_star, c=v_trth, alpha=0.9, edgecolors='none', cmap='seismic', marker='s', s=s, vmin=0, vmax=1)
    ax[1, 1].axis('square')
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_title(r'$v$-FD', fontsize=22)
    cbar = fig.colorbar(cf, orientation='horizontal', ax=ax[1, 1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)
    #####
    plt.savefig('./output/uv_comparison_'+str(num).zfill(3)+'.png',dpi=100)
    plt.close('all')

if __name__ == "__main__":

    # Domain bounds
    MAX_T = 400                         # dt=0.5 in PeRCNN
    lb = np.array([-0.5, -0.5,     0])
    ub = np.array([ 0.5,  0.5,   200])  # fix the scaler for pretraining and fietuning

    # Network configuration
    uv_layers = [3] + 5 * [80] + [2]
    f_layers  = [4] + 2 * [10] + [2]

    # Initial condition point for u, v
    x, y = [np.linspace(-0.5, 0.5, 101)]*2
    x, y = np.meshgrid(x[::4], y[::4])
    x, y = x.flatten()[:, None], y.flatten()[:, None]
    truth = scipy.io.loadmat('../../2d_gs_rd/data/2DRD_2x3001x100x100_[dt=05].mat')['uv']
    truth = np.concatenate((truth, truth[:, :, 0:1, :]), axis=2)          # Padding
    truth = np.concatenate((truth, truth[:, :, :, 0:1]), axis=3)
    UV = add_noise(truth, pec=0.1)                                        # Add noise
    UV = UV[:, 0:801:20, ::4, ::4]                                        # Downsampling
    XYT_c = []
    for i in range(UV.shape[1]):
        t = 400/(UV.shape[1]-1)*i
        u, v = UV[0, i, :, :].flatten()[:, None], UV[1, i, :, :].flatten()[:, None]
        xytuv = np.concatenate((x, y, x*0+t, u, v), 1)   # [x, y, t, u, v]
        XYT_c.append(xytuv)
    XYT_c = np.concatenate(XYT_c, axis=0)

    LW = [-0.5, -0.5, 0] + [1.0, 0, MAX_T] * lhs(3, 8000)
    UP = np.concatenate((LW[:, 0:1], LW[:, 1:2]+1.0, LW[:, 2:3]), axis=1)

    LF = [-0.5, -0.5, 0] + [0, 1.0, MAX_T] * lhs(3, 8000)
    RT = np.concatenate((LF[:, 0:1]+1.0, LF[:, 1:2], LF[:, 2:3]), axis=1)


    # Visualize ALL the training points
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(XYT_c[:,0:1], XYT_c[:,1:2], XYT_c[:,2:3], marker='o', alpha=0.1, s=1, color='blue')
    ax.scatter(UP[:, 0:1], UP[:, 1:2], UP[:, 2:3], marker='o', alpha=0.2, s=1)
    ax.scatter(LW[:, 0:1], LW[:, 1:2], LW[:, 2:3], marker='o', alpha=0.2, s=1)
    ax.scatter(LF[:, 0:1], LF[:, 1:2], LF[:, 2:3], marker='o', alpha=0.2, s=1)
    ax.scatter(RT[:, 0:1], RT[:, 1:2], RT[:, 2:3], marker='o', alpha=0.2, s=1)
    # ax.scatter(IC[:, 0:1], IC[:, 1:2], IC[:, 2:3], marker='o', alpha=0.2, s=1)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('T axis')
    plt.show()

    with tf.device('/device:GPU:0'):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # model = GS_DHPM(XYT_c, UP, LW, RT, LF, uv_layers, f_layers, lb, ub)
        model = GS_DHPM(XYT_c, UP, LW, RT, LF, uv_layers, f_layers, lb, ub,
                        uvDir='uv_NN_5x80_float32.pickle', fDir='f_NN_2x10_float32.pickle')

        model.getloss()

        start_time = time.time()
        # model.train(iter=5000, learning_rate=2e-3)
        model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))

        ################End of pre-train####################

        model.save_NN_f('f_NN_2x10_float32.pickle')
        model.save_NN_uv('uv_NN_5x80_float32.pickle')

        model.getloss()

        N_t = 41
        # Output result at each time step
        x_star = np.linspace(-0.5, 0.5, 101)
        y_star = np.linspace(-0.5, 0.5, 101)
        x_star, y_star = np.meshgrid(x_star, y_star)
        x_star = x_star.flatten()[:, None]
        y_star = y_star.flatten()[:, None]
        shutil.rmtree('./output', ignore_errors=True)
        os.makedirs('./output')
        for i in range(N_t):
            t_star = 0*x_star + i*MAX_T/(N_t-1)
            u_pred, v_pred = model.predict(x_star, y_star, t_star)
            u_trth, v_trth = truth[0, 20*i, :, :].flatten()[:, None],  truth[1, 20*i, :, :].flatten()[:, None]       # 2x100x100
            field = [x_star, y_star, t_star, u_pred, v_pred, u_trth, v_trth]
            postProcess(xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5, s=6, field=field, num=i)

        # save results
        UV = []
        x_star, y_star = [np.linspace(-0.5, 0.5, 101)] * 2
        x_star, y_star = np.meshgrid(x_star[:-1], y_star[:-1])
        x_star, y_star = x_star.flatten()[:, None], y_star.flatten()[:, None]
        for i in range(2501):
            print('num=', i)
            t_star = 0.5*i + 0*x_star
            u_pred, v_pred = model.predict(x_star, y_star, t_star)
            u_pred, v_pred = u_pred.reshape(1, 1, 100, 100), v_pred.reshape(1, 1, 100, 100)
            UV.append(np.concatenate((u_pred, v_pred), axis=0))

        UV = np.concatenate(UV, axis=1)
        scipy.io.savemat('uv_2x2501x100x100_[DHPM].mat', {'uv': UV[:, :, :, :]})

    pass
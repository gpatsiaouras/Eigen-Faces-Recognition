import numpy as np
import scipy.io

fea = np.array(scipy.io.loadmat('../resources/ORL_32x32.mat')['fea'])
gnd = np.array(scipy.io.loadmat('../resources/ORL_32x32.mat')['gnd'])
train_3 = np.array(scipy.io.loadmat('../resources/3Train/3.mat'))
train_5 = np.array(scipy.io.loadmat('../resources/5Train/5.mat'))
train_7 = np.array(scipy.io.loadmat('../resources/7Train/7.mat'))


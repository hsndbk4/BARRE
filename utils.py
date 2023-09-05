import math
import os
import sys
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torchvision




def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # decrease efficiency
    # torch.backends.cudnn.enabled = False
    print("==> Set seed to {:}".format(seed))

def arr_to_str(x):
    M = len(x)
    x_str = "["
    for m in range(M-1):
        x_str+="{:.4f}, ".format(x[m])

    x_str+="{:.4f}]".format(x[M-1])
    return x_str
def proj_onto_simplex(x):
    N = len(x)
    y = np.sort(x)[::-1]
    rho=-1
    for i in range(N):
        q = y[i]+(1-y[:i+1].sum())/(1+i)
        if q >0:
            rho=i
    l = (1-y[:rho+1].sum())/(rho+1)
    x_hat = np.zeros(N)
    for i in range(N):
        if x[i]+l>0:
            x_hat[i]=x[i]+l
    return x_hat

def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()

def remove_module(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_state_dict['.'.join(key.split('.')[1:])] = state_dict[key]
    return new_state_dict

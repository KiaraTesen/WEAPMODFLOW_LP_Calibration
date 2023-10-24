# -*- coding: utf-8 -*-

#---    Packages
from Functions_ADPSO_CL import *
import numpy as np
import h5py
import os
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

#---    Initial matriz
n = 35                                                 # Population size: 20 o 35

active_cells = 18948

k_shape_1 = (5,5)   #HK_1
k_shape_2 = (3,3)   #SY_1
k_shape_3 = (3,3)   #HK_2
k_shape_4 = (3,3)   #SY_2

n_var = active_cells * 2
for k in range(1,5):
    globals()['n_var_' + str(k)] = reduce(lambda x,y: x*y, globals()['k_shape_' + str(k)])
    n_var += globals()['n_var_' + str(k)]
n_var = n_var    # Number of variables
print (n_var)

#---    Create iteration register file
with h5py.File('Pre_ADPSO-CL.h5', 'w') as f:
    pob_x_h5py = f.create_dataset("pob_x", (n, n_var))

#---    Bounds
lb_kx, ub_kx = 3.25, 5
lb_sy, ub_sy = 1.175, 1.2

lb_1_kx, ub_1_kx = 0.075, 1.0
lb_1_sy, ub_1_sy = 0.1375, 0.14
lb_2_kx, ub_2_kx = 0.075, 0.50
lb_2_sy, ub_2_sy = 0.15, 0.1525	

l_bounds = np.concatenate((np.around(np.repeat(lb_kx, active_cells),4), np.around(np.repeat(lb_sy, active_cells),4), 
                           np.around(np.repeat(lb_1_kx, n_var_1),4), np.around(np.repeat(lb_1_sy, n_var_2),4), 
                           np.around(np.repeat(lb_2_kx, n_var_3),4), np.around(np.repeat(lb_2_sy, n_var_4),4)), axis = 0)
u_bounds = np.concatenate((np.around(np.repeat(ub_kx, active_cells),4), np.around(np.repeat(ub_sy, active_cells),4), 
                           np.around(np.repeat(ub_1_kx, n_var_1),4), np.around(np.repeat(ub_1_sy, n_var_2),4), 
                           np.around(np.repeat(ub_2_kx, n_var_3),4), np.around(np.repeat(ub_2_sy, n_var_4),4)), axis = 0) 

#---    Initial Sampling (Latyn Hypercube)
class Particle:
    def __init__(self,x,v,y):
        self.x = x                      # X represents the kernels
        self.v = v                      # initial velocity = zeros
        self.y = y
        self.x_best = np.copy(x)                 
        self.y_best = y

sample_scaled = get_sampling_LH(n_var, n, l_bounds, u_bounds)
print(sample_scaled)

#---    Iteration register
for i in range(n):
    with h5py.File('Pre_ADPSO-CL.h5', 'a') as f:
        f["pob_x"][i] = np.copy(sample_scaled[i])
    f.close()

#---    Read file to verify
with h5py.File('Pre_ADPSO-CL.h5', 'r') as f:
    x = f["pob_x"][:]
print(x[0])
print(len(x))
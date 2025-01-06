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
n = 20                                                # Population size: 20

active_cells = 10440

k_shape_1 = (3,3)   # HK
k_shape_2 = (3,3)   # SY

n_var = active_cells * 2
for k in range(1,3):
    globals()['n_var_' + str(k)] = reduce(lambda x,y: x*y, globals()['k_shape_' + str(k)])
    n_var += globals()['n_var_' + str(k)]
n_var = n_var       # Number of variables
print(n_var, active_cells)

#---    Create iteration register file
with h5py.File('Pre_ADPSO-CL.h5', 'w') as f:
    pob_x_h5py = f.create_dataset("pob_x", (n, int(n_var + active_cells)))

#---    Bounds
lb_kx, ub_kx = 0.001, 30
lb_sy, ub_sy = 0.015, 5

lb_1_kx, ub_1_kx = 0.0750, 1.0000
lb_1_sy, ub_1_sy = 0.0050, 0.0500

l_bounds = np.concatenate((np.around(np.repeat(lb_kx, active_cells),4), np.around(np.repeat(lb_sy, active_cells),4), 
                           np.around(np.repeat(lb_1_kx, n_var_1),4), np.around(np.repeat(lb_1_sy, n_var_2),4)), axis = 0)
u_bounds = np.concatenate((np.around(np.repeat(ub_kx, active_cells),4), np.around(np.repeat(ub_sy, active_cells),4), 
                           np.around(np.repeat(ub_1_kx, n_var_1),4), np.around(np.repeat(ub_1_sy, n_var_2),4)), axis = 0) 

#---    Initial Sampling (Latyn Hypercube)
pre_sample_scaled = get_sampling_LH(n_var, n, l_bounds, u_bounds)
pre_sample_scaled = np.around(pre_sample_scaled, 4)

#--- #---   Sub generación SY/SS
sample_scaled = np.zeros(shape=(n, int(n_var + active_cells)))
for l in range(n):
    p_values =  [100, 1000, 10000]
    sy_ss_values = np.random.choice(p_values, active_cells)

    sample_scaled[l] = np.concatenate((pre_sample_scaled[l], sy_ss_values), axis = 0)
#print(sample_scaled)

#---    Iteration register
for i in range(n):
    with h5py.File('Pre_ADPSO-CL.h5', 'a') as f:
        f["pob_x"][i] = np.copy(np.around(sample_scaled[i],4))
    f.close()

#---    Read file to verify
with h5py.File('Pre_ADPSO-CL.h5', 'r') as f:
    x = f["pob_x"][:]
print(x[0].size)
print(x[int(n-1)])
print(len(x))
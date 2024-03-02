# -*- coding: utf-8 -*-

#---    Packages
from Functions_ADPSO_CL import *
import geopandas as gpd
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from functools import reduce
import time
import sys
from request_server.request_server import send_request_py
import warnings
warnings.filterwarnings('ignore')

IP_SERVER_ADD = sys.argv[1]
TOTAL_ITERATION = int(sys.argv[2])
FINAL_ITERATION = int(sys.argv[3])
VM = int(sys.argv[4])

#---    Paths
path_WEAP = r'C:\Users\vagrant\Documents\WEAP Areas\Ligua_WEAP_MODFLOW'
path_model = os.path.join(path_WEAP, 'NWT_L_v2')
path_init_model = r'C:\Users\vagrant\Documents\WEAPMODFLOW_LP_Calibration\data\MODFLOW_model\NWT_L_initial'
path_nwt_exe = r'C:\Users\vagrant\Documents\WEAPMODFLOW_LP_Calibration\data\MODFLOW-NWT_1.2.0\bin\MODFLOW-NWT_64.exe'
path_GIS = r'C:\Users\vagrant\Documents\WEAPMODFLOW_LP_Calibration\data\GIS'
path_output = r'C:\Users\vagrant\Documents\WEAPMODFLOW_LP_Calibration\ADPSO-CL\output'
path_obs_data = r'C:\Users\vagrant\Documents\WEAPMODFLOW_LP_Calibration\data\ObservedData'

#---    Initial matriz
HP = ['kx', 'sy'] 
initial_shape_HP = gpd.read_file(path_GIS + '/SuperfitialGeology_Ligua_initial_values_v2.shp') ########33
active_matriz = initial_shape_HP['ACTIVEL1'].to_numpy().reshape((172,369))                  # Matrix of zeros and ones that allows maintaining active area

active_cells = 10440

k_shape_1 = (3,3)   # HK
k_shape_2 = (3,3)   # SY

n_var = active_cells * 2
for k in range(1,3):
    globals()['n_var_' + str(k)] = reduce(lambda x,y: x*y, globals()['k_shape_' + str(k)])
    n_var += globals()['n_var_' + str(k)]
n_var = n_var       # Number of variables
print (n_var)

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
class Particle:
    def __init__(self,x,v,y):
        self.x = x                      # X represents the kernels
        self.v = v                      # initial velocity = zeros
        self.y = y
        self.x_best = np.copy(x)                 
        self.y_best = y

pob = Particle(np.around(np.array([0]*(n_var)),4),np.around(np.array([0]*(n_var)),4),10000000000)

#---    Initial Sampling - Pob(0)
dir_recap = os.path.join(path_output + '/ADPSO_CL_register_vm' + str(VM) + '.h5')
if not os.path.isfile(dir_recap):
    ITERATION = 0
else:
    with h5py.File('ADPSO_CL_register_vm' + str(VM) + '.h5', 'r') as f:
        n_recap = len(f["pob_x"][:])
    ITERATION = n_recap
print(ITERATION)

if ITERATION == 0:

    with h5py.File('Pre_ADPSO-CL.h5', 'r') as f:
        pob.x = np.copy(f["pob_x"][VM-2])
    f.close()

    y_init = Run_WEAP_MODFLOW(path_output, str(ITERATION), initial_shape_HP, HP, active_cells, pob.x, n_var_1, n_var_2, n_var, 
                            k_shape_1, k_shape_2, active_matriz, path_init_model, path_model, path_nwt_exe, path_obs_data)
    pob.x_best = np.copy(pob.x)
    pob.y = y_init
    pob.y_best = y_init

    #---    Create iteration register file
    with h5py.File('ADPSO_CL_register_vm' + str(VM) + '.h5', 'w') as g:
        iter_h5py = g.create_dataset("iteration", (FINAL_ITERATION, 1))
        pob_x_h5py = g.create_dataset("pob_x", (FINAL_ITERATION, n_var))
        pob_y_h5py = g.create_dataset("pob_y", (FINAL_ITERATION, 1))
        pob_v_h5py = g.create_dataset("pob_v", (FINAL_ITERATION, n_var))
        pob_x_best_h5py = g.create_dataset("pob_x_best", (FINAL_ITERATION, n_var))
        pob_y_best_h5py = g.create_dataset("pob_y_best", (FINAL_ITERATION, 1))
        pob_w_h5py = g.create_dataset("w", (FINAL_ITERATION, 1))

    #---    Iteration register
        iter_h5py[0] = ITERATION
        pob_x_h5py[0] = np.copy(pob.x)
        pob_y_h5py[0] = pob.y
        pob_v_h5py[0] = np.copy(pob.v)
        pob_x_best_h5py[0] = np.copy(pob.x_best)
        pob_y_best_h5py[0] = pob.y_best
        pob_w_h5py[0] = 0.5
    g.close()

    ITERATION += 1

        #---    PSO setup
    α = 0.8                                                     # Cognitive scaling parameter  # 0.8 # 1.49
    β = 0.8                                                     # Social scaling parameter     # 0.8 # 1.49                       
    w_min = 0.4                                                 # minimum value for the inertia velocity
    w_max = 0.9                                                 # maximum value for the inertia velocity
    vMax = np.around(np.multiply(u_bounds-l_bounds,0.4),4)      # Max velocity # De 0.8 a 0.4 # E2 con 0.4
    vMin = -vMax
    w = 0.5                                                     # inertia velocity

    for it in range(ITERATION, FINAL_ITERATION):
        
        time.sleep(np.random.randint(10,20,size = 1)[0])
        gbest = send_request_py(IP_SERVER_ADD, pob.y, pob.x)           # Update global particle

        #---    Update particle velocity
        ϵ1,ϵ2 = np.around(np.random.uniform(),4), np.around(np.random.uniform(),4)            # [0, 1]

        pob.v = np.around(np.around(w*pob.v,4) + np.around(α*ϵ1*(pob.x_best - pob.x),4) + np.around(β*ϵ2*(gbest - pob.x),4),4)

        #---    Adjust particle velocity
        index_vMax = np.where(pob.v > vMax)
        index_vMin = np.where(pob.v < vMin)

        if np.array(index_vMax).size > 0:
            pob.v[index_vMax] = vMax[index_vMax]
        if np.array(index_vMin).size > 0:
            pob.v[index_vMin] = vMin[index_vMin]

        #---    Update particle position
        pob.x += pob.v

        #---    Adjust particle position
        index_pMax = np.where(pob.x > u_bounds)
        index_pMin = np.where(pob.x < l_bounds)

        if np.array(index_pMax).size > 0:
            pob.x[index_pMax] = u_bounds[index_pMax]
        if np.array(index_pMin).size > 0:
            pob.x[index_pMin] = l_bounds[index_pMin]

        #---    Evaluate the fitnness function
        y = Run_WEAP_MODFLOW(path_output, str(ITERATION), initial_shape_HP, HP, active_cells, pob.x, n_var_1, n_var_2, n_var, 
                            k_shape_1, k_shape_2, active_matriz, path_init_model, path_model, path_nwt_exe, path_obs_data)
        
        if y < pob.y_best:
            pob.x_best = np.copy(pob.x)
            pob.y_best = y
            pob.y = y
        else:
            pob.y = y

        #---    Update the inertia velocity
        w = w_max - (ITERATION) * ((w_max-w_min)/FINAL_ITERATION)

        #---    Iteration register
        with h5py.File('ADPSO_CL_register_vm' + str(VM) + '.h5', 'a') as g:
            g["iteration"][ITERATION] = ITERATION
            g["pob_x"][ITERATION] = np.copy(pob.x)
            g["pob_y"][ITERATION] = pob.y
            g["pob_v"][ITERATION] = np.copy(pob.v)
            g["pob_x_best"][ITERATION] = np.copy(pob.x_best)
            g["pob_y_best"][ITERATION] = pob.y_best

            g["w"][ITERATION] = w
        g.close()

        ITERATION += 1

#---    Siguientes iteraciones
else:
    #---    PSO setup
    α = 0.8                                                     # Cognitive scaling parameter  # 0.8 # 1.49
    β = 0.8                                                     # Social scaling parameter     # 0.8 # 1.49                       
    w_min = 0.4                                                 # minimum value for the inertia velocity
    w_max = 0.9                                                 # maximum value for the inertia velocity
    vMax = np.around(np.multiply(u_bounds-l_bounds,0.4),4)      # Max velocity # De 0.8 a 0.4 # E2 con 0.4
    vMin = -vMax
    w = 0.5                                                     # inertia velocity

    with h5py.File('ADPSO_CL_register_vm' + str(VM) + '.h5', 'r') as g:
        pob.x = np.copy(g["pob_x"][ITERATION - 1])
        pob.y = g["pob_y"][ITERATION - 1]
        pob.v = np.copy(g["pob_v"][ITERATION - 1])
        pob.x_best = np.copy(g["pob_x_best"][ITERATION - 1])
        pob.y_best = g["pob_y_best"][ITERATION - 1]

        w = g["w"][ITERATION - 1]                               # inertia velocity
    g.close()

    for it in range(ITERATION, FINAL_ITERATION):
        
        time.sleep(np.random.randint(10,20,size = 1)[0])
        gbest = send_request_py(IP_SERVER_ADD, pob.y, pob.x)           # Update global particle

        #---    Update particle velocity
        ϵ1,ϵ2 = np.around(np.random.uniform(),4), np.around(np.random.uniform(),4)            # [0, 1]

        pob.v = np.around(np.around(w*pob.v,4) + np.around(α*ϵ1*(pob.x_best - pob.x),4) + np.around(β*ϵ2*(gbest - pob.x),4),4)

        #---    Adjust particle velocity
        index_vMax = np.where(pob.v > vMax)
        index_vMin = np.where(pob.v < vMin)

        if np.array(index_vMax).size > 0:
            pob.v[index_vMax] = vMax[index_vMax]
        if np.array(index_vMin).size > 0:
            pob.v[index_vMin] = vMin[index_vMin]

        #---    Update particle position
        pob.x += pob.v

        #---    Adjust particle position
        index_pMax = np.where(pob.x > u_bounds)
        index_pMin = np.where(pob.x < l_bounds)

        if np.array(index_pMax).size > 0:
            pob.x[index_pMax] = u_bounds[index_pMax]
        if np.array(index_pMin).size > 0:
            pob.x[index_pMin] = l_bounds[index_pMin]

        #---    Evaluate the fitnness function
        y = Run_WEAP_MODFLOW(path_output, str(ITERATION), initial_shape_HP, HP, active_cells, pob.x, n_var_1, n_var_2, n_var, 
                            k_shape_1, k_shape_2, active_matriz, path_init_model, path_model, path_nwt_exe, path_obs_data)
        
        if y < pob.y_best:
            pob.x_best = np.copy(pob.x)
            pob.y_best = y
            pob.y = y
        else:
            pob.y = y

        #---    Update the inertia velocity
        w = w_max - (ITERATION) * ((w_max-w_min)/FINAL_ITERATION)

        #---    Iteration register
        with h5py.File('ADPSO_CL_register_vm' + str(VM) + '.h5', 'a') as g:
            g["iteration"][ITERATION] = ITERATION
            g["pob_x"][ITERATION] = np.copy(pob.x)
            g["pob_y"][ITERATION] = pob.y
            g["pob_v"][ITERATION] = np.copy(pob.v)
            g["pob_x_best"][ITERATION] = np.copy(pob.x_best)
            g["pob_y_best"][ITERATION] = pob.y_best

            g["w"][ITERATION] = w
        g.close()

        ITERATION += 1
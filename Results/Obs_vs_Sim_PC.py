#---    Packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hydroeval as he
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from scipy import signal
import flopy.modflow as fpm
import shutil
import win32com.client as win32
import math

#--------------------------------
#---    Observation wells    ----
#--------------------------------
obs_well = pd.read_csv(r'..\data\ObservedData\Monitoring_wells.csv', skiprows = 3)
obs_well = obs_well.set_index('Unnamed: 0')
obs_well = obs_well.transpose()
obs_well = obs_well.iloc[260:-832, :]

ow = obs_well.columns

#sim_well = pd.read_csv(r'..\data\Simulated_wells_PC.csv', skiprows = 3)
sim_well = pd.read_csv(r'..\data\Simulated_wells_RDM.csv', skiprows = 3)
sim_well = sim_well.set_index('Branch')
sim_well = sim_well.iloc[260:, :]

for p in ow[1:]:
    df_w = pd.DataFrame()
    df_w['obs'] = np.array(obs_well[p])
    df_w.loc[df_w['obs'] == ' ', 'obs'] = np.nan
    df_w['sim'] = np.array(sim_well['Sim_' + p])
    df_w = df_w.set_index(pd.to_datetime(sim_well.index))

    #---    Metrics
    df_w.loc[df_w['sim'] == ' ', 'sim'] = np.nan
    df_w = df_w.dropna()
    mod_w = df_w.loc[:, 'sim']
    obs_w = df_w.loc[:, 'obs']

    mse_w = mean_squared_error(obs_w, mod_w)
    rmse_w = math.sqrt(mse_w)                                                               # RMSE
    kge_w, r, alpha, beta = he.evaluator(he.kge, mod_w, np.array(obs_w, dtype = float))         # KGE
    mae_w = mean_absolute_error(obs_w, mod_w)                                                   # MAE
    print(p, ' RMSE: ', rmse_w, ' KGE: ', kge_w[0], ' MAE: ', mae_w)

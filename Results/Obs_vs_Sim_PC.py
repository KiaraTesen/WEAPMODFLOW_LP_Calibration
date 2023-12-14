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


best_exp = 'E1'
best_vm = 'vm15'
best_iter = 25
path_best = os.path.join(r'D:\2_PaperII_LP', best_exp, best_vm, 'iter_' + str(best_iter))

#--------------------------------
#---    Observation wells    ----
#--------------------------------
obs_well = pd.read_csv(r'..\data\ObservedData\Monitoring_wells.csv', skiprows = 3)
obs_well = obs_well.set_index('Unnamed: 0')
obs_well = obs_well.transpose()
obs_well = obs_well.iloc[260:-832, :]

ow = obs_well.columns

sim_well = pd.read_csv(os.path.join(path_best, 'iter_' + str(best_iter) + '_Simulated_wells.csv'), skiprows = 3)
sim_well = sim_well.set_index('Branch')
sim_well = sim_well.iloc[260:, :]

for p in ow[1:]:
    df_w = pd.DataFrame()
    df_w['obs'] = np.array(obs_well[p])
    df_w.loc[df_w['obs'] == ' ', 'obs'] = np.nan
    df_w['sim'] = np.array(sim_well['Sim_' + p])
    df_w = df_w.set_index(pd.to_datetime(sim_well.index))

    #---    Graphs
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(df_w.index, df_w['obs'].astype(float), "o", label = 'Obs', color = "black", linewidth = 0.1)
    ax2.plot(df_w.index, df_w['sim'], label = 'ADPSO-CL', color = "red", linewidth = 0.75)

    title = 'Wells - ' + str(p)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(bbox_to_anchor = (1.0, 1.0), loc = 'upper left')
    plt.title(title, fontsize = 21, fontweight = 'bold')
    plt.legend(loc = 'upper right', fontsize = 18)

    plt.ylabel('Groundwater table (m)', fontsize = 18)
    plt.xlabel('Years', fontsize = 18)

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
    #---    Continue graph
    """
    ymax2 = max(mod_w.max(), np.array(obs_w, dtype = float).max())
    plt.text(datetime.strptime('2001-03-24', '%Y-%m-%d'), ymax2 - ymax2/10, 'RMSE: ' + str(round(rmse_w,2)), fontsize = 18)
    plt.text(datetime.strptime('2001-03-24', '%Y-%m-%d'), ymax2 - ymax2/9, 'MAE: ' + str(round(mae_w, 2)), fontsize = 18)
    plt.text(datetime.strptime('2001-03-24', '%Y-%m-%d'), ymax2 - ymax2/8, 'KGE: ' + str(round(kge_w[0], 2)), fontsize = 18)
    """
    plt.savefig('W_obs_vs_sim_' + str(p) + '.png')
    plt.clf
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

#----------------------------------
#---    Streamflow analysis    ----
#----------------------------------
sg = ['AlicahueEnColliguay', 'EF_L02_Embalse', 'EF_L07_Confluencia', 'EF_L07_Embalse', 'EF_L09_Confluencia', 'EF_P02_Confluencia', 'EF_P07_Confluencia', 
      'EF_P07_Embalse', 'EF_P08_Ossandon', 'LiguaEnQuinquimo', 'PedernalenTejada', 'PetorcaEnLongotoma', 'PetorcaEnPenon', 'SobranteEnPinadero']

for j in sg:
    df_q = pd.read_csv(os.path.join(path_best, f"iter_{str(best_iter)}_Q_"+ j) + ".csv", skiprows = 3)
    df_q = df_q.set_index('Statistic')
    df_q = df_q.iloc[260:,:]

    DF_q = pd.DataFrame()
    DF_q['Modeled - ADPSO-CL'] = np.array(df_q['Modeled'])
    DF_q['Observed'] = np.array(df_q['Observed'])
    DF_q.loc[DF_q['Observed'] == ' ', 'Observed'] = np.nan
    DF_q = DF_q.set_index(pd.to_datetime(df_q.index))
    #print(DF_q)

    #---    Graph
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(DF_q.index, DF_q['Observed'].astype(float), "o", label = 'Obs', color = "black", linewidth = 0.1)
    ax.plot(DF_q.index, DF_q['Modeled - ADPSO-CL'], label = 'ADPSO-CL', color = "red", linewidth = 0.75)

    title = 'Streamflow gauge - ' + str(j)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(bbox_to_anchor = (1.0, 1.0), loc = 'upper left')
    plt.title(title, fontsize = 21, fontweight = 'bold')
    plt.legend(loc = 'upper right', fontsize = 18)

    plt.ylabel('Streamflow ($m^{3}/s$)', fontsize = 18)
    plt.xlabel('Years', fontsize = 18)

    #---    Metrics
    DF_q.loc[DF_q['Modeled - ADPSO-CL'] == ' ', 'Modeled - ADPSO-CL'] = np.nan
    DF_q = DF_q.dropna()
    mod = DF_q.loc[:, 'Modeled - ADPSO-CL']
    obs = DF_q.loc[:, 'Observed']

    mse_q = mean_squared_error(obs, mod)
    rmse_q = math.sqrt(mse_q)                                                               # RMSE
    kge_q, r, alpha, beta = he.evaluator(he.kge, mod, np.array(obs, dtype = float))         # KGE
    mae_q = mean_absolute_error(obs, mod)                                                   # MAE
    print(j,' RMSE: ', rmse_q, ' KGE: ', kge_q[0], ' MAE: ', mae_q)
    #---    Continue graph
    """
    ymax = max(mod.max(), np.array(obs, dtype = float).max())
    plt.text(datetime.strptime('2001-03-24', '%Y-%m-%d'), ymax - ymax/5, 'RMSE: ' + str(round(rmse_q,2)), fontsize = 18)
    plt.text(datetime.strptime('2001-03-24', '%Y-%m-%d'), ymax - ymax/4, 'MAE: ' + str(round(mae_q, 2)), fontsize = 18)
    plt.text(datetime.strptime('2001-03-24', '%Y-%m-%d'), ymax - ymax/3.4, 'KGE: ' + str(round(kge_q[0], 2)), fontsize = 18)
    """
    plt.savefig('Q_obs_vs_sim_' + str(j) + '.png')
    plt.clf

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
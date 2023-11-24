import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube,scale
from scipy import signal
import pandas as pd
import os
import flopy.modflow as fpm
import shutil
import win32com.client as win32
from sklearn.metrics import mean_squared_error
import math
from Complete_MODFLOW_Results import *
from CriteriosSustentabilidad import *
import warnings
warnings.filterwarnings('ignore')

#---    Visualization of the matriz -
def get_image_matriz(matriz, variable, path_out):
    fig=plt.figure(figsize = (16,8))
    ax = plt.axes()
    im = ax.imshow(matriz)
    plt.title(variable)
    plt.xlabel('Column')
    plt.ylabel('Row')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    plt.savefig(path_out)

#---    Sampling by Latin Hypecube
def get_sampling_LH(n_var, n, l_bounds, u_bounds):
    engine = LatinHypercube(d=n_var)
    sample = engine.random(n=n)
    sample_scaled = scale(sample, l_bounds, u_bounds)
    return np.around(sample_scaled, 4)

#---    Modification 1 of HP
def get_pre_HP(Shape_HP, new_Shape, variable, particle, begin, end):
    count = 0
    x = particle[begin : end]

    for i in range(len(new_Shape)):    
        if variable == "sy":
            if new_Shape[variable][i] == 0:
                new_Shape[variable][i] = 0.01
            else:
                #print(count, x[count], Shape_HP[variable][i])
                new_Shape[variable][i] = round(x[count] * Shape_HP[variable][i],4)
                count += 1
        else:
            if new_Shape[variable][i] == 0:
                new_Shape[variable][i] = 0.000001728
            else:
                #print(count, x[count], Shape_HP[variable][i])
                new_Shape[variable][i] = round(x[count] * Shape_HP[variable][i],4)
                count += 1
        
    rows = new_Shape["ROW"].max()
    columns = new_Shape["COLUMN"].max()

    matriz = np.zeros((rows,columns))
    for i in range(0,len(new_Shape['ROW'])):
        matriz[new_Shape['ROW'][i]-1][new_Shape['COLUMN'][i]-1] = new_Shape[variable][i] 
    #print(matriz)
    return matriz

#---    Modification 2 of HP
def get_HP(Shape_HP, variable, active_matriz, decimals, kernel):
    rows = Shape_HP["ROW"].max()
    columns = Shape_HP["COLUMN"].max()

    matriz = np.zeros((rows,columns))
    for i in range(0,len(Shape_HP['ROW'])):
        matriz[Shape_HP['ROW'][i]-1][Shape_HP['COLUMN'][i]-1] = Shape_HP[variable][i] 

    #---    Convolution
    new_matriz = signal.convolve2d(matriz, kernel, boundary = 'symm', mode = 'same')
    new_matriz = np.around(new_matriz * active_matriz, decimals = decimals)
    if variable == "sy":
        new_matriz = np.where(new_matriz == 0, 0.01, new_matriz)
    else:
        new_matriz = np.where(new_matriz == 0, 0.000001728, new_matriz)

    return new_matriz

def get_eliminate_zeros(lista):
    position = 0
    while position < len(lista):
        if lista[position] == 0.000001728:
            lista.pop(position)
        else:
            position += 1
    return lista 

#---    Evaluate "Subject to" bounds
def get_evaluate_st_bounds(min_v, max_v, vector_modif):
    if min_v > min(vector_modif):
        P_min = min_v - min(vector_modif)
    else:
        P_min = 0

    if max_v < max(vector_modif):
        P_max = max(vector_modif) - max_v
    else:
        P_max = 0
    return P_min + P_max

def Run_WEAP_MODFLOW(path_output, iteration, initial_shape_HP, HP, active_cells, sample_scaled, n_var_1, n_var_2, n_var_3, n_var, 
                     k_shape_1, k_shape_2, k_shape_3, k_shape_4, active_matriz, path_init_model, path_model, path_nwt_exe, 
                     path_obs_data):

    dir_iteration = os.path.join(path_output, "iter_" + str(iteration))
    if not os.path.isdir(dir_iteration):
        os.mkdir(dir_iteration)
    
    #--------------------------
    #---    Run MODFLOW    ----
    #--------------------------
    #---    Modified matriz
    pre_shape_HP = initial_shape_HP.copy()
    shape_k1_HP = initial_shape_HP.copy()
    new_shape_HP = initial_shape_HP.copy()

    for m in HP:
        if m == "kx":
            begin = 0
            end = active_cells
        elif m == "sy":
            begin = active_cells
            end = active_cells * 2
        globals()["matriz_pre_" + str(m)] = get_pre_HP(initial_shape_HP, pre_shape_HP, str(m), sample_scaled, begin, end)
        get_image_matriz(globals()["matriz_pre_" + str(m)], str(m), os.path.join(dir_iteration, 'Pre_' + str(m) +'.png'))
        plt.clf

        # CONVOLUTIONAL LAYERS
        decimals_kx = 4
        decimals_sy = 4

        # First kernel
        kernel_1_kx = sample_scaled[int(active_cells * 2):int(active_cells * 2 + n_var_1)].reshape(k_shape_1)
        kernel_1_sy = sample_scaled[int(active_cells * 2 + n_var_1):int(active_cells * 2 + n_var_1 + n_var_2)].reshape(k_shape_2)

        globals()["matriz_1_" + str(m)] = get_HP(pre_shape_HP, str(m), active_matriz, locals()["decimals_" + str(m)], locals()["kernel_1_" + str(m)])
        get_image_matriz(globals()["matriz_1_" + str(m)], str(m), os.path.join(dir_iteration, '1_' + str(m) +'.png'))
        plt.clf()
        globals()["vector_1_" + str(m)] = globals()["matriz_1_" + str(m)].flatten()
        shape_k1_HP[m] = globals()["vector_1_" + str(m)]

        # Second kernel
        kernel_2_kx = sample_scaled[int(active_cells * 2 + n_var_1 + n_var_2):int(active_cells * 2 + n_var_1 + n_var_2 + n_var_3)].reshape(k_shape_3)
        kernel_2_sy = sample_scaled[int(active_cells * 2 + n_var_1 + n_var_2 + n_var_3):int(n_var)].reshape(k_shape_4)
        
        globals()["matriz_" + str(m)] = get_HP(shape_k1_HP, str(m), active_matriz, locals()["decimals_" + str(m)], locals()["kernel_2_" + str(m)])
        get_image_matriz(globals()["matriz_" + str(m)], str(m), os.path.join(dir_iteration, 'Final_' + str(m) +'.png'))
        plt.clf()
        globals()["vector_" + str(m)] = globals()["matriz_" + str(m)].flatten()
        new_shape_HP[m] = globals()["vector_" + str(m)]

    #---    Other variables that MODFLOW require
    new_shape_HP['kz'] = vector_kx / new_shape_HP['HK/VK']

    rows_ = new_shape_HP["ROW"].max()
    columns_ = new_shape_HP["COLUMN"].max()
    matriz_kz = np.zeros((rows_,columns_))
    for i in range(0,len(new_shape_HP['ROW'])):
        matriz_kz[new_shape_HP['ROW'][i]-1][new_shape_HP['COLUMN'][i]-1] = new_shape_HP['kz'][i] 

    matriz_ss = matriz_sy/100
    new_shape_HP['ss'] = matriz_ss.flatten()

    new_shape_HP.to_file(os.path.join(dir_iteration, 'Elements_iter_' + str(iteration) + '.shp'))

    #----------------------------------------
    #---    Generate new native files    ----
    #----------------------------------------

    model = fpm.Modflow.load(path_init_model + '/con_dren_sin_aisladas_NWT.nam', version = 'mfnwt', exe_name = path_nwt_exe)
    model.write_input()
    model.remove_package("UPW")
    upw = fpm.ModflowUpw(model = model, laytyp=1, layavg=0, chani=-1.0, layvka=0, laywet=0, hdry=-888, iphdry=1, hk=matriz_kx, hani=1.0, vka=matriz_kz, ss=matriz_ss, sy=matriz_sy, extension='upw')
    upw.write_file()
    model.run_model()

    #---    Move native files to WEAP
    get_old_files = os.listdir(path_model)
    get_new_files = os.listdir(os.getcwd())

    #---    Delete old files
    for g in get_old_files:
        try:
            os.remove(os.path.join(path_model, g))
        except:
            print('No hay archivos')

    #---    Move new files
    for h in get_new_files:
        if h.endswith('.py') or h == '__pycache__' or h == 'sp' or h.endswith('.txt') or h == 'output' or h.endswith('.ps1') or h.endswith('.h5'):
            pass 
        else:
            shutil.move(os.path.join(os.getcwd(), h), os.path.join(path_model, h))

    #-------------------------------------
    #---    Run WEAP-MODFLOW model    ----
    #-------------------------------------
    WEAP = win32.Dispatch("WEAP.WEAPApplication")
    WEAP.ActiveArea = "Ligua_Petorca_WEAP_MODFLOW_RDM"
    WEAP.Calculate()
    
    #---    Export results
    favorites = pd.read_excel(r"C:\Users\vagrant\Documents\WEAPMODFLOW_LP_Calibration\data\Favorites_WEAP.xlsx")
    for i,j in zip(favorites["BranchVariable"],favorites["WEAP Export"]):
        WEAP.LoadFavorite(i)
        WEAP.ExportResults(os.path.join(dir_iteration, f"iter_{str(iteration)}_{j}.csv"), True, True, True, False, False)
    
    #------------------------------
    #---    MODFLOW Balance    ----
    #------------------------------
    MODFLOW_Balance(dir_iteration, path_model, path_obs_data)
    Processing_Balance(dir_iteration, path_obs_data)
    Volumenes_MODFLOW(path_model, dir_iteration, path_obs_data)

    #---------------------------------
    #---    Objective Function    ----
    #---------------------------------

    #---    Monitoring wells
    obs_well = pd.read_csv(os.path.join(path_obs_data, 'Monitoring_wells.csv'), skiprows = 3)
    obs_well = obs_well.set_index('Unnamed: 0')
    obs_well = obs_well.transpose()
    obs_well = obs_well.iloc[260:,:]

    ow = obs_well.columns

    sim_well = pd.read_csv(os.path.join(dir_iteration, f"iter_{str(iteration)}_Simulated_wells.csv"), skiprows = 3)
    sim_well = sim_well.set_index('Branch')
    sim_well = sim_well.iloc[260:,:]

    srmse_well = 0
    for i in ow:
        df_ = pd.DataFrame()
        df_['Obs'] = obs_well[i]
        df_['Sim'] = sim_well['Sim_'+i]
        df_ = df_.dropna()

        mse_well = mean_squared_error(df_['Obs'], df_['Sim'])
        rmse_well = math.sqrt(mse_well)
        srmse_well += rmse_well
    #print(srmse_well)

    #---    Streamflow gauges
    sg = ['AlicahueEnColliguay', 'EF_L02_Embalse', 'EF_L07_Confluencia', 'EF_L07_Embalse', 'EF_L09_Confluencia', 
          'EF_P02_Confluencia', 'EF_P07_Confluencia', 'EF_P07_Embalse', 'EF_P08_Ossandon', 'LiguaEnQuinquimo', 
          'PedernalenTejada', 'PetorcaEnLongotoma', 'PetorcaEnPenon', 'SobranteEnPinadero']

    srmse_q = 0
    for j in sg:
        df_q = pd.read_csv(os.path.join(dir_iteration, f"iter_{str(iteration)}_Q_"+ j) + ".csv", skiprows = 3)
        df_q = df_q.set_index('Statistic')
        df_q = df_q.iloc[260:,:]
        df_q.loc[df_q['Observed'] == ' ', 'Observed'] = np.nan
        df_q.loc[df_q['Modeled'] == ' ', 'Modeled'] = np.nan
        df_q = df_q.dropna()

        mse_q = mean_squared_error(df_q['Observed'], df_q['Modeled'])
        rmse_q = math.sqrt(mse_q)
        srmse_q += rmse_q
    #print(srmse_q)

    #---    Subject to
    kx_min = 0.000864
    kx_max = 86.4
    sy_min = 0.01
    sy_max = 0.14

    for i in HP:
        globals()["vector_modif_" + str(i)] = get_eliminate_zeros(globals()["vector_" + str(i)].tolist())
        globals()["P_" + str(i)] = get_evaluate_st_bounds((locals()[str(i) + "_min"]), (locals()[str(i) + "_max"]), globals()["vector_modif_" + str(i)])

    """
    #---    Streamflow penalization - Empty results
    for h in sg:
        df_q = pd.read_csv(os.path.join(dir_iteration, f"iter_{str(iteration)}_Q_"+ j + ".csv"), skiprows = 3)
        df_q = df_q.set_index('Statistic')
        df_q = df_q.iloc[260:,:]
        df_q.loc[df_q['Modeled'] == ' ', 'Modeled'] = np.nan
        df_q_null = df_q.isnull()
        freq = df_q_null.groupby(['Modeled']).count()
    #print('True: ', int(freq.iloc[1]))

    if int(freq.iloc[1]) > 0:
        P_q = 100
    else:
        P_q = 0
        """
    
    #---    Cells penalization
    Aq_zones = ['L01', 'L02', 'L05', 'L06', 'L09', 'L10', 'L12', 'P01', 'P02', 'P03', 'P07', 'P08_v1', 'P08_v2']

    error_cells_aum = 0
    error_cells_dis = 0
    for k in Aq_zones:
        df_cells = pd.read_csv(os.path.join(dir_iteration, f"iter_{str(iteration)}_Cells_"+ k + ".csv"), skiprows = 3)
        df_cells = df_cells.set_index('Branch')
        #df_cells = df_cells.iloc[260:,:]
        df_cells = df_cells.transpose()
        df_cells['Diference'] = df_cells['Dec 24 2004'] - df_cells['Jan 1 1984']
        
        df_c_ = df_cells[df_cells['Diference'] > 30]
        df_c_min = df_cells[df_cells['Diference'] < -100]

        e_aum = df_c_['Diference'].sum()
        e_dis = df_c_min['Diference'].sum()
        #print(e_aum, e_dis)
        
        error_cells_aum += e_aum
        error_cells_dis += e_dis

    #---    Total Objective Function
    g1 = 0.80
    g2 = 0.60
    g3 = 0.60
    #g4 = 0.60
    g5 = 0.05

    #of = g1*srmse_well + g2*rmse_q + g3*(P_kx + P_sy) + g4*P_q + g5*(error_cells_aum - error_cells_dis)
    of = g1*srmse_well + g2*rmse_q + g3*(P_kx + P_sy) + g5*(error_cells_aum - error_cells_dis)
    return of

def Run_WEAP_MODFLOW_temp(path_output, iteration, initial_shape_HP, HP, active_cells, sample_scaled, n_var_1, n_var_2, n_var_3, n_var, 
                     k_shape_1, k_shape_2, k_shape_3, k_shape_4, active_matriz, path_init_model, path_model, path_nwt_exe, 
                     path_obs_data):

    dir_iteration = os.path.join(path_output, "iter_" + str(iteration))
    if not os.path.isdir(dir_iteration):
        os.mkdir(dir_iteration)

    #---------------------------------
    #---    Objective Function    ----
    #---------------------------------

    #---    Monitoring wells
    obs_well = pd.read_csv(os.path.join(path_obs_data, 'Monitoring_wells.csv'), skiprows = 3)
    obs_well = obs_well.set_index('Unnamed: 0')
    obs_well = obs_well.transpose()
    obs_well = obs_well.iloc[260:-832,:]

    ow = obs_well.columns

    sim_well = pd.read_csv(os.path.join(dir_iteration, f"iter_{str(iteration)}_Simulated_wells.csv"), skiprows = 3)
    sim_well = sim_well.set_index('Branch')
    sim_well = sim_well.iloc[260:,:]

    srmse_well = 0
    for i in ow:
        df_ = pd.DataFrame()
        df_['Obs'] = np.array(obs_well[i])
        df_['Sim'] = np.array(sim_well['Sim_'+i])
        df_ = df_.dropna()
        
        mse_well = mean_squared_error(df_['Obs'], df_['Sim'])
        rmse_well = math.sqrt(mse_well)
        srmse_well += rmse_well
    print(srmse_well)

    #---    Streamflow gauges
    sg = ['AlicahueEnColliguay', 'EF_L02_Embalse', 'EF_L07_Confluencia', 'EF_L07_Embalse', 'EF_L09_Confluencia', 
          'EF_P02_Confluencia', 'EF_P07_Confluencia', 'EF_P07_Embalse', 'EF_P08_Ossandon', 'LiguaEnQuinquimo', 
          'PedernalenTejada', 'PetorcaEnLongotoma', 'PetorcaEnPenon', 'SobranteEnPinadero']

    srmse_q = 0
    for j in sg:
        df_q = pd.read_csv(os.path.join(dir_iteration, f"iter_{str(iteration)}_Q_"+ j) + ".csv", skiprows = 3)
        df_q = df_q.set_index('Statistic')
        df_q = df_q.iloc[260:,:]
        df_q.loc[df_q['Observed'] == ' ', 'Observed'] = np.nan
        df_q.loc[df_q['Modeled'] == ' ', 'Modeled'] = np.nan
        df_q = df_q.dropna()

        mse_q = mean_squared_error(df_q['Observed'], df_q['Modeled'])
        rmse_q = math.sqrt(mse_q)
        srmse_q += rmse_q
    #print(srmse_q)

    #---    Subject to
    kx_min = 0.000864
    kx_max = 86.4
    sy_min = 0.01
    sy_max = 0.14

    for i in HP:
        globals()["vector_modif_" + str(i)] = get_eliminate_zeros(globals()["vector_" + str(i)].tolist())
        globals()["P_" + str(i)] = get_evaluate_st_bounds((locals()[str(i) + "_min"]), (locals()[str(i) + "_max"]), globals()["vector_modif_" + str(i)])

    """
    #---    Streamflow penalization - Empty results
    for h in sg:
        df_q = pd.read_csv(os.path.join(dir_iteration, f"iter_{str(iteration)}_Q_"+ j + ".csv"), skiprows = 3)
        df_q = df_q.set_index('Statistic')
        df_q = df_q.iloc[260:,:]
        df_q.loc[df_q['Modeled'] == ' ', 'Modeled'] = np.nan
        df_q_null = df_q.isnull()
        freq = df_q_null.groupby(['Modeled']).count()
    #print('True: ', int(freq.iloc[1]))

    if int(freq.iloc[1]) > 0:
        P_q = 100
    else:
        P_q = 0
        """
    
    #---    Cells penalization
    Aq_zones = ['L01', 'L02', 'L05', 'L06', 'L09', 'L10', 'L12', 'P01', 'P02', 'P03', 'P07', 'P08_v1', 'P08_v2']

    error_cells_aum = 0
    error_cells_dis = 0
    for k in Aq_zones:
        df_cells = pd.read_csv(os.path.join(dir_iteration, f"iter_{str(iteration)}_Cells_"+ k + ".csv"), skiprows = 3)
        df_cells = df_cells.set_index('Branch')
        #df_cells = df_cells.iloc[260:,:]
        df_cells = df_cells.transpose()
        df_cells['Diference'] = df_cells['Dec 24 2004'] - df_cells['Jan 1 1984']
        
        df_c_ = df_cells[df_cells['Diference'] > 30]
        df_c_min = df_cells[df_cells['Diference'] < -100]

        e_aum = df_c_['Diference'].sum()
        e_dis = df_c_min['Diference'].sum()
        #print(e_aum, e_dis)
        
        error_cells_aum += e_aum
        error_cells_dis += e_dis

    #---    Total Objective Function
    g1 = 0.80
    g2 = 0.60
    g3 = 0.60
    #g4 = 0.60
    g5 = 0.05

    #of = g1*srmse_well + g2*rmse_q + g3*(P_kx + P_sy) + g4*P_q + g5*(error_cells_aum - error_cells_dis)
    of = g1*srmse_well + g2*rmse_q + g3*(P_kx + P_sy) + g5*(error_cells_aum - error_cells_dis)
    return of
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

#---    Visualization of the matriz      ## SE PUEDE ELIMINAR
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
                new_Shape[variable][i] = round(x[count] * Shape_HP[variable][i],4)
                count += 1
        elif variable == "kx":
            if new_Shape[variable][i] == 0:
                new_Shape[variable][i] = 0.000001728
            else:
                new_Shape[variable][i] = round(x[count] * Shape_HP[variable][i],4)
                count += 1
        elif variable == "sy_ss":
            if new_Shape[variable][i] == 0:
                new_Shape[variable][i] = 100
            else:
                new_Shape[variable][i] = round(x[count],4)
                count += 1

    rows = new_Shape["ROW"].max()
    columns = new_Shape["COLUMN"].max()

    matriz = np.zeros((rows,columns))
    for i in range(0,len(new_Shape['ROW'])):
        matriz[new_Shape['ROW'][i]-1][new_Shape['COLUMN'][i]-1] = new_Shape[variable][i] 
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

def Run_WEAP_MODFLOW(path_output, iteration, initial_shape_HP, HP, active_cells, sample_scaled, n_var_1, n_var_2, n_var, 
                     k_shape_1, k_shape_2, active_matriz, path_init_model, path_model, path_nwt_exe, path_obs_data):
    dir_iteration = os.path.join(path_output, "iter_" + str(iteration))
    if not os.path.isdir(dir_iteration):
        os.mkdir(dir_iteration)
    
    #--------------------------
    #---    Run MODFLOW    ----
    #--------------------------
        
    #---    Modified matriz
    pre_shape_HP = initial_shape_HP.copy()
    new_shape_HP = initial_shape_HP.copy()

    decimals = 4
    for m in HP:
        if m == "kx":
            begin = 0
            end = active_cells
        elif m == "sy":
            begin = active_cells
            end = active_cells * 2
        elif m == "sy_ss":
            begin = n_var
            end = n_var + active_cells

        globals()["matriz_pre_" + str(m)] = get_pre_HP(initial_shape_HP, pre_shape_HP, str(m), sample_scaled, begin, end)
###        get_image_matriz(globals()["matriz_pre_" + str(m)], str(m), os.path.join(dir_iteration, 'Pre_' + str(m) +'.png'))
###        plt.clf

        #---    CLs
        kernel_kx = sample_scaled[int(active_cells * 2):int(active_cells * 2 + n_var_1)].reshape(k_shape_1)
        kernel_sy = sample_scaled[int(active_cells * 2 + n_var_1):int(active_cells * 2 + n_var_1 + n_var_2)].reshape(k_shape_2)

        if m == "kx":
            globals()["matriz_" + str(m)] = get_HP(pre_shape_HP, str(m), active_matriz, decimals, locals()["kernel_" + str(m)])
            globals()["matriz_" + str(m)] = np.where(globals()["matriz_" + str(m)] < 0.0000000001, 0.0000000001, globals()["matriz_" + str(m)])
###            get_image_matriz(globals()["matriz_" + str(m)], str(m), os.path.join(dir_iteration, 'Final_' + str(m) +'.png'))
###            plt.clf()
            globals()["vector_" + str(m)] = globals()["matriz_" + str(m)].flatten()
            new_shape_HP[m] = globals()["vector_" + str(m)]

        elif m == "sy":
            globals()["matriz_" + str(m)] = get_HP(pre_shape_HP, str(m), active_matriz, decimals, locals()["kernel_" + str(m)])
            globals()["matriz_" + str(m)] = np.where(globals()["matriz_" + str(m)] < 0.01, 0.01, globals()["matriz_" + str(m)])
            globals()["matriz_" + str(m)] = np.where(globals()["matriz_" + str(m)] > 0.5, 0.5, globals()["matriz_" + str(m)])
###            get_image_matriz(globals()["matriz_" + str(m)], str(m), os.path.join(dir_iteration, 'Final_' + str(m) +'.png'))
###            plt.clf()
            globals()["vector_" + str(m)] = globals()["matriz_" + str(m)].flatten()
            new_shape_HP[m] = globals()["vector_" + str(m)]
        
        elif m == "sy_ss":
            globals()["matriz_" + str(m)] = globals()["matriz_pre_" + str(m)]
###            get_image_matriz(globals()["matriz_" + str(m)], str(m), os.path.join(dir_iteration, 'Final_' + str(m) +'.png'))
###            plt.clf()
            globals()["vector_" + str(m)] = globals()["matriz_" + str(m)].flatten()
            new_shape_HP[m] = globals()["vector_" + str(m)]
        
    #---    Other variables that MODFLOW require
    #---    #---    kz
    new_shape_HP['kz'] = vector_kx / new_shape_HP['HK/VK']

    rows_ = new_shape_HP["ROW"].max()
    columns_ = new_shape_HP["COLUMN"].max()
    matriz_kz = np.zeros((rows_,columns_))
    for i in range(0,len(new_shape_HP['ROW'])):
        matriz_kz[new_shape_HP['ROW'][i]-1][new_shape_HP['COLUMN'][i]-1] = new_shape_HP['kz'][i] 

    #---    #---    ss
    new_shape_HP['ss'] = vector_sy / new_shape_HP['sy_ss']

    rows_ = new_shape_HP["ROW"].max()
    columns_ = new_shape_HP["COLUMN"].max()
    matriz_ss = np.zeros((rows_,columns_))
    for i in range(0,len(new_shape_HP['ROW'])):
        matriz_ss[new_shape_HP['ROW'][i]-1][new_shape_HP['COLUMN'][i]-1] = new_shape_HP['ss'][i] 

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
    WEAP.ActiveArea = "Ligua_WEAP_MODFLOW"
    WEAP.Calculate()
    
    #---    Export results
    favorites = pd.read_excel(r"C:\Users\vagrant\Documents\WEAPMODFLOW_LP_Calibration\data\Favorites_WEAP.xlsx")
##    favorites = pd.read_excel(r"C:\Users\aimee\Desktop\Github\WEAPMODFLOW_LP_Calibration\data\Favorites_WEAP.xlsx")

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
    #print(srmse_well)

    #---    Streamflow gauges
    sg = ['AlicahueEnColliguay', 'EF_L02_Embalse', 'EF_L07_Confluencia', 'EF_L07_Embalse', 'EF_L09_Confluencia', 'LiguaEnQuinquimo']

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
        if i == 'sy_ss':
            pass
        else:
            globals()["vector_modif_" + str(i)] = get_eliminate_zeros(globals()["vector_" + str(i)].tolist())
            globals()["P_" + str(i)] = get_evaluate_st_bounds((locals()[str(i) + "_min"]), (locals()[str(i) + "_max"]), globals()["vector_modif_" + str(i)])

    #---    Total Objective Function
    g1 = 1.50
    g2 = 0.60
    g3 = 0.60

    of = g1*srmse_well + g2*rmse_q + g3*(P_kx + P_sy)
    return of
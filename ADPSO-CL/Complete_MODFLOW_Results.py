# -*- coding: utf-8 -*-

import pandas as pd
import os
from flopy.utils.zonbud import ZoneBudget, read_zbarray
from datetime import datetime as dt
import glob
import time
import numpy as np

################################################
####    Funciones Procesar archivos .ccf    ####
################################################

def get_scenario(temp):
    return temp[-15:-12]

def get_date(temp):
    diferencia_TS = 0
    temp1 = '1_' + temp[-11:-4]
    temp2 = dt.strptime(temp1, '%d_%Y_%W')
    return temp2 - pd.DateOffset(months=diferencia_TS)   

def get_TS(directorio, zone_analysis, output, zones):
    zones = zones
    new_df = pd.DataFrame()
    os.chdir(directorio)
    for file in glob.glob('*.csv'):
        df = pd.read_csv(file)
        melted = df.melt(id_vars=['name'], value_vars=zones)
        wk2 = melted.loc[melted['variable'] == zone_analysis]
        wk2 = wk2.drop(['variable'], axis=1)
        wk2 = wk2.T
        wk2['name_file'] = file
        new_df = pd.concat([wk2, new_df])

    new_df.columns = new_df.iloc[0]
    new_df = new_df.drop(['name'], axis=0)
    column_list = list(new_df.columns.values)
    last_name = column_list[-1]
    new_df.rename(columns={last_name: 'file'}, inplace=True)
    new_df['Scenario'] = new_df.apply(lambda x: get_scenario(x['file']), axis=1)
    new_df['date'] = new_df.apply(lambda x: get_date(x['file']), axis=1)
    new_df.set_index('date', inplace=True)
    new_df = new_df.sort_values(['date'],ascending=True)
    new_df.drop(['file'], axis=1, inplace=True)
    dir_out = output + '/' + zone_analysis + '.csv'
    new_df.to_csv(dir_out)

def get_full_balance(path_balance, path_ZB, dir_exit, temp_path, aliases, zones):
    zonefile = read_zbarray(path_ZB)

    # Leer binarios de la carpeta WEAP
    for file in os.listdir(path_balance):
        filename = os.fsdecode(file)
        if filename.endswith(".ccf"):
            t = temp_path + '/' + filename[:-4] + '.csv'
            zb = ZoneBudget(path_balance + '\\' + filename, zonefile, aliases=aliases)
            zb.to_csv(t)
            
    zones = zones
    for zone in zones:
        get_TS(temp_path, zone, dir_exit, zones)

    filelist = [ f for f in os.listdir(temp_path) if f.endswith(".csv") ]
    for f in filelist:
        os.remove(os.path.join(temp_path, f))

############################################################################
####                  PRE-PROCESSING MODFLOW RESULTS                    ####
####    COMENTARIO: La versión hace referencia al ID de la ejecución    ####
####                ruta_WEAP se especifica según la PC del usuario     ####
############################################################################

ZB = ['Zones.zbr']

def MODFLOW_Balance(dir_iteration, path_model, path_obs_data):
    # COMPLETE BALANCE
    directorio = dir_iteration + '/Zones'
    if not os.path.isdir(directorio):
        os.mkdir(directorio)

    dir_temp = directorio + '/temp'
    if not os.path.isdir(dir_temp):
        os.mkdir(dir_temp)

    # Variables
    nombre_archivo_ZB = 'Zones.zbr'
    nombre_carpeta_MF = 'NWT_L_v2'
    zones = ['L01','L02','L05','L06','L09','L10','L12']  # Zone Budget Zones
    aliases = {1:'L01', 2:'L02', 3:'L05', 4:'L06', 5:'L09', 6:'L10', 7:'L12'} # Alias Zone Budget Zone
        
    path_salida = directorio
    path_balance = path_model
    path_ZB = path_obs_data + '/' + nombre_archivo_ZB
    temp_path = dir_temp
        
    # Ejecución funciones de Procesamiento
    get_full_balance(path_balance, path_ZB, path_salida, temp_path, aliases, zones)

    # Elimina carpeta temporal
    try:
        os.rmdir(temp_path)
    except OSError as e:
        print("Error: %s: %s" % (temp_path, e.strerror))

###################################################
####    POST - PROCESSING - MODFLOW RESULTS    ####
###################################################

def get_df_ls(df, fecha):
    df_ls = pd.DataFrame()
    for i in df.columns.values[1:-2]:
        df_ls[i] = pd.DataFrame((df[i].to_numpy())/86400)
    df_ls.set_index(fecha['Fecha'],inplace = True)
    df_temp = df_ls.iloc[260:,:]
    return df_temp

def get_balance_cuenca(ruta_export_BALANCE, inicio, fin, zones, variables, años, cuenca):
    Res = (pd.read_excel(ruta_export_BALANCE + '/Resumen_balance_' + str(zones[inicio]) + '.xlsx').iloc[:,1:12]).to_numpy()
    for q in range (inicio + 1,fin):
        dato = (pd.read_excel(ruta_export_BALANCE + '/Resumen_balance_' + str(zones[q]) + '.xlsx').iloc[:,1:12]).to_numpy()
        Res = Res + dato
    Res_cuenca = pd.DataFrame(Res, columns = variables)
    Res_cuenca.set_index(años['Fecha'],inplace = True)
    return Res_cuenca.to_excel(ruta_export_BALANCE + '/Resumen_balance_' + str(cuenca) + '.xlsx')

def Processing_Balance(dir_iteration, path_obs_data):
    ruta_BALANCE_ZB = dir_iteration + '/Zones'
    
    ruta_export_BALANCE = dir_iteration + '/BALANCE'
    if not os.path.isdir(ruta_export_BALANCE):
        os.mkdir(ruta_export_BALANCE)

    fecha = pd.read_csv(path_obs_data + '/Fechas.csv')
    años = pd.read_csv(path_obs_data + '/Años.csv')

    variables = ['Variacion Neta Flujo Interacuifero', 'Recarga desde río', 'Recarga Lateral', 
                 'Recarga distribuida', 'Recarga', 'Variacion Neta Flujo Mar', 'Afloramiento - DRAIN', 
                 'Afloramiento - RIVER', 'Afloramiento total', 'Bombeos', 'Almacenamiento']

    zones = ['L01','L02','L05','L06','L09','L10','L12']  # Zone Budget Zones
    # SERIES ANUALES - AÑO HIDROLÓGICO
    for j in zones:
        Resumen = pd.DataFrame(columns = variables)

        df = pd.read_csv(ruta_BALANCE_ZB + '/' + j + '.csv')
    
        df_temp = get_df_ls(df, fecha)
           
        # ANALISIS
        FI_in = (df_temp['FROM_ZONE_0'].to_numpy() + df_temp['FROM_L01'].to_numpy() + df_temp['FROM_L02'].to_numpy() + df_temp['FROM_L05'].to_numpy() + df_temp['FROM_L06'].to_numpy() + 
                 df_temp['FROM_L09'].to_numpy() + df_temp['FROM_L10'].to_numpy() + df_temp['FROM_L12'].to_numpy())
        FI_out = (df_temp['TO_ZONE_0'].to_numpy() + df_temp['TO_L01'].to_numpy() + df_temp['TO_L02'].to_numpy() + df_temp['TO_L05'].to_numpy() + df_temp['TO_L06'].to_numpy() + 
                  df_temp['TO_L09'].to_numpy() + df_temp['TO_L10'].to_numpy() + df_temp['TO_L12'].to_numpy())
        Resumen.loc[:,'Variacion Neta Flujo Interacuifero'] = FI_in - FI_out

        Rch_rio = (df_temp['FROM_RIVER_LEAKAGE'].to_numpy())
        Resumen.loc[:,'Recarga desde río'] = Rch_rio
    
        Rch_lat = (df_temp['FROM_WELLS'].to_numpy())
        Resumen.loc[:,'Recarga Lateral'] = Rch_lat

        Rch_dist = (df_temp['FROM_RECHARGE'].to_numpy())
        Resumen.loc[:,'Recarga distribuida'] = Rch_dist

        Resumen.loc[:, 'Recarga'] = Rch_rio + Rch_lat + Rch_dist
    
        Resumen.loc[:,'Variacion Neta Flujo Mar'] = (df_temp['FROM_CONSTANT_HEAD'].to_numpy() - df_temp['TO_CONSTANT_HEAD'].to_numpy())
        
        Af_Drain = -(df_temp['TO_DRAINS'].to_numpy())
        Resumen.loc[:,'Afloramiento - DRAIN'] = Af_Drain

        Af_RIVER = -(df_temp['TO_RIVER_LEAKAGE'].to_numpy())
        Resumen.loc[:,'Afloramiento - RIVER'] = Af_RIVER

        Resumen.loc[:,'Afloramiento total'] = Af_Drain + Af_RIVER
        
        Resumen.loc[:,'Bombeos'] = -(df_temp['TO_WELLS'].to_numpy())
    
        Resumen.loc[:,'Almacenamiento'] = -(df_temp['FROM_STORAGE'].to_numpy() - df_temp['TO_STORAGE'].to_numpy())

        Resumen.to_excel(ruta_export_BALANCE + '/Resumen_mensual_balance_' + str(j) + '.xlsx')

        Resumen = Resumen.to_numpy()
        data_prom = np.zeros((21,11))
        for n in range(0,11):
            for m in range(0,21):
                data_prom[m,n] = np.mean(Resumen[:,n][52*m:52*m+52])   

        Res_anual = pd.DataFrame(data_prom, columns = variables)
        Res_anual.set_index(años['Fecha'],inplace = True)
        Res_anual.to_excel(ruta_export_BALANCE + '/Resumen_balance_' + str(j) + '.xlsx')

    Ligua = get_balance_cuenca(ruta_export_BALANCE, 0, 7, zones, variables, años, 'Ligua')
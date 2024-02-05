# -*- coding: utf-8 -*-
import glob
import os
import numpy as np
import flopy.utils.binaryfile as bf
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import re

def find_indices(lst, condition):
    """
    Selecciona los indices(elementos) de la lista segun una condicion particular
    Parametters --> lst :      Lista de elementos
    Returns --> retorno : Lista de indices que se asocian a elementos que cumplen una condicion
    """
    retorno = [x for x, elem in enumerate(lst) if condition(elem)]
    return retorno

#----------------------------------
#----    Volumenes por Shac    ----
#----------------------------------
def Volumenes_MODFLOW(path_model, dir_iteration, path_obs_data):
    # Leer archivo .hed para extraer flujo en las celdas y volumenes
    prj = 'con_dren_sin_aisladas_NWT'
    Path_out = dir_iteration

    # Lectura de resultados y archivos nativos
    path_balance = path_model
    Pth_DIS = path_balance+'/'+prj+'.dis'
    Pth_UPW = path_balance+'/'+prj+'.upw'
    Pth_ZB = path_obs_data + '/Zones.zbr' 
    filelisthed = glob.glob(path_balance+'\\*.hed')

    key_hed = lambda s: s.rsplit('_')[-3]    # Con esto selecciono el escenario con el cual generare grupos
    Scenario = []

    zones = ['L01','L02','L05','L06','L09','L10','L12']  # Zone Budget Zones
    aliases = {1:'L01', 2:'L02', 3:'L05', 4:'L06', 5:'L09', 6:'L10', 7:'L12'}
    Shac = len(zones)

    cell_A = 200*200    # Tamaño celda, info que se debe extraer manualmente de la grilla MODFLOW
    rows = 174          # Cantidad de filas, info que se puede extraer de la grilla MODFLOW, el .dis, .zbr o el .zb_zones
    columns = 371      # Cantidad de columnas, info que se puede extraer de la grilla MODFLOW, el .dis, .zbr o el .zb_zones

    # Leer archivo .DIS para extraer info de TOP y BOTTOM
    f = open(Pth_DIS,'r')
    DIS_T = []
    DIS_B = []

    count = 0
    for line in f:
        if '#model_top' in line or '#botm_layer_0' in line or 'TR' in line:
            count += 1
        if '#' in line:
            continue
        if count == 1:
            line = line[1:]
            line = line.split('   ')
            if line == ['']:
                continue
            else:
                line = [float(x) for x in line]
                DIS_T.append(line)
        
        if count == 2:
            line = line[1:]
            line = re.split("   |  ", line)
            if line == ['']:
                continue
            else:
                line = [float(x) for x in line[1:]]
                DIS_B.append(line)
        if count == 3:
            break
    f.close

    # Arreglo de info row,columns
    DIS_B = np.array(list(chain(*DIS_B)))
    DIS_B.resize((rows, columns), refcheck=False)

    DIS_T = np.array(list(chain(*DIS_T)))
    DIS_T.resize((rows, columns), refcheck=False)

    # Leer archivo .ZBR para extraer info de ZoneBudgets  
    fp = open(Pth_ZB, 'r')
    lines = fp.readlines()

    data = []
    for line in lines[2:]:
        data.append([int(v) for v in line.split()])
    fp.close()
    
    Zones = np.array(list(chain(*data)))
    Zones.resize((rows,columns), refcheck=False)

    # Extraer Ss y Sy a partir de .upw
    f = open(Pth_UPW,'r')
    Values_Ss = []

    lines_to_read = list(range(359, 533))
    for position, line in enumerate(f):
        if position in lines_to_read:
            line = line[2:]
            line = line.split('  ')
            line = [float(x) for x in line]
            Values_Ss.append(line)    
    f.close

    Ss_mod = np.array(list(chain(*Values_Ss)))
    Ss_mod.resize((rows,columns), refcheck=False)
    #print(Ss_mod)

    f = open(Pth_UPW,'r')
    Values_Sy = []
    lines_to_read = list(range(534, 708))

    for position, line in enumerate(f):
        if position in lines_to_read:
            line = line[2:]
            line = line.split('  ')
            line = [float(x) for x in line]
            Values_Sy.append(line)    
    f.close

    Sy_mod = np.array(list(chain(*Values_Sy)))
    Sy_mod.resize((rows,columns), refcheck=False)
    #print(Sy_mod)
    
    for i in range(len(filelisthed)-1):
        if key_hed(filelisthed[i]) =='S00' or key_hed(filelisthed[i])=='S01':   #Descarto corrida 0 y año base
            pass
        elif key_hed(filelisthed[i]) in Scenario:
            continue
        else:
            Scenario.append(key_hed(filelisthed[i]))

    #Guarda las series de volumenes segun shac en cada escenario, VOLSZB[escenario][shac]
    VOLSZB = [[] for i in range(len(Scenario))]

    #Son los archivos .hed agrupados
    subHed = [[] for i in range(len(Scenario))]
        
    for i in range(len(Scenario)):
        # Selecciona los elementos (indices) que comparten escenario
        index_listhed = find_indices(filelisthed, lambda s: s.rsplit('_')[-3]==Scenario[i])
        subHed[i] = filelisthed[index_listhed[0]:index_listhed[len(index_listhed)-1]+1]

    year = [[] for i in range(len(Scenario))]
    anio = [[] for i in range(len(Scenario))]
    week = [[] for i in range(len(Scenario))]
    for f in range(len(Scenario)):
        volZB = [[] for i in range(Shac)]
        HEAD = []
        for file in subHed[f]:
            filename = os.fsdecode(file)
            hds = filename
            sp = filename[-11:-4]
            cut_ = sp.index("_")+1
            year_ = int(sp[0:4])
            week_ = int(sp[cut_::])
            time_ = year_+round((week_-1)/52,3)
            HedF = bf.HeadFile(hds, precision='single')
            Hd = HedF.get_data(kstpkper=(0,0))
            Hd = Hd[0]
            Hd = np.where(Hd==-999., 0, Hd) #cero para el plot
            Hd = np.where(Hd==-888., 0, Hd) #cero para el plot
            HEAD = Hd
            DIS_B[Hd==0]=0
            Vol_T = (HEAD-DIS_B)*cell_A #nf-bottom por celda tamaño
            Vol_H = np.multiply(Vol_T, Sy_mod) #Multiplica volumen por Sy
            Vol_T[DIS_T<HEAD]=(DIS_T[DIS_T<HEAD]-DIS_B[DIS_T<HEAD])*cell_A
            Vol_H[DIS_T<HEAD] = np.multiply(Vol_T[DIS_T<HEAD], Ss_mod[DIS_T<HEAD]) # Cuando TOP < HEAD, se multiplica por Ss
            Vol_T[DIS_T>=HEAD]=(HEAD[DIS_T>=HEAD]-DIS_B[DIS_T>=HEAD])*cell_A
            Vol_H[DIS_T>=HEAD] = np.multiply(Vol_T[DIS_T>=HEAD], Sy_mod[DIS_T>=HEAD]) # Cuando TOP >= HEAD, se multiplica por Sy
            Vol_T = np.where(Vol_T<0., 0, Vol_T)
            VolT = sum(sum(Vol_T))
            VolH = sum(sum(Vol_H))
            for i in range(Shac):
                ZBud = np.where(Zones != i+1,0, 0)
                ZBud = np.where(Zones == i+1,1, 0)
                Vol_z = np.multiply(Vol_H, ZBud)
                Vol_z = sum(sum(Vol_z))/(10**6)
                volZB[i].append(Vol_z)
            year[f].append(time_)    
            anio[f].append(year_)
            week[f].append(week_)
        #print('Volumen Total bajo NF :  ', VolT/10**6, 'Mm3')
        #print('Volumen Total Extraible :  ', VolH/10**6, 'Mm3')
        VOLSZB[f] = volZB

    for i in range(Shac):
        Escenarios_Volumen_Freatico = pd.DataFrame()
        for e in range(0,1):   
            Escenarios_Volumen_Freatico[Scenario[e] + 'Volumen - [Mm3]'] = VOLSZB[e][i]    #Aca es importante tener las zones ordenadas segun los ordenes de shac 
            Escenarios_Volumen_Freatico['Year'] = anio[e]
            Escenarios_Volumen_Freatico['Week'] = week[e]
            Escenarios_Volumen_Freatico.to_csv(Path_out+'/Volumen - SHAC - '+zones[i]+'.csv')
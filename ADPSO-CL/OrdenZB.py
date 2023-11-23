# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:47:36 2022
@author: Kiara Tesen
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\aimee\Desktop\Github\WEAPMODFLOW_LP_Calibration\data\ObservedData\ZB_RDM_LP.csv')
matriz = np.zeros((263,371))
for i in range(0,len(df['ROW'])):
    matriz[df['ROW'][i]-1][df['COLUMN'][i]-1] = df['ZONE'][i]

matriz = pd.DataFrame(matriz)
matriz.to_csv('MATRIZ_ZB_LP.csv')

import h5py
import os
import sys
import pandas as pd
import numpy as np

path_output = r'C:\Users\vagrant\Documents\WEAPMODFLOW_LP_Calibration\ADPSO-CL\output'
total_vms = 15

df_y = pd.DataFrame()
iter_vm_pre = 0
for i in range (2, int(total_vms + 2)):

    print('VM = ' + str(i))
    
    dir_file = os.path.join(path_output, 'ADPSO_CL_register_vm' + str(i) + '.h5')
    with h5py.File(dir_file, 'r') as f:
        x = f["pob_x"][:]
        v = f["pob_v"][:]
        y = f["pob_y"][:]
        x_best = f["pob_x_best"][:]
        y_best = f["pob_y_best"][:]
        w = f["w"][:]

        g_best_selected = f["pob_x"][3]
    f.close()

    print(g_best_selected)
    iterations = len(x)
    new_iter = iterations * (i-1)
    print('Bounds loc: ', iter_vm_pre, ' - ', new_iter - 1)
    print('Real iteration: 0 - ', iterations - 1)

    for j in range(iter_vm_pre, new_iter):
        df_y.loc[j, 'Y'] = y[int(j-new_iter), 0]
        df_y.loc[j, 'Y_BEST'] = y_best[int(j-new_iter), 0]
    iter_vm_pre += iterations

print(df_y)
min_y = min(df_y['Y'])
loc = df_y[df_y['Y'] == min_y]
print(loc)

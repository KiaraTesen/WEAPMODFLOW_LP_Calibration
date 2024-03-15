import h5py
import os
import sys
import pandas as pd
import numpy as np
from request_server.request_server import send_request_py

IP_SERVER_ADD = '10.0.0.11:8888'

path_output = r'C:\Users\vagrant\Documents\WEAPMODFLOW_LP_Calibration\ADPSO-CL\output'
#path_output = r'C:\Users\aimee\Desktop\Github\WEAPMODFLOW_LP_Calibration\ADPSO-CL\output'

best_vm = 14         # TO MODIFY
best_iter = 92       # TO MODIFY

dir_file = os.path.join(path_output, 'ADPSO_CL_register_vm' + str(best_vm) + '.h5')

with h5py.File(dir_file, 'r') as f:
    x_best = f["pob_x_best"][best_iter]
    y_best = f["pob_y_best"][best_iter]
f.close()

print(x_best)
print(y_best)
gbest = send_request_py(IP_SERVER_ADD, y_best, x_best)           # Update global particle

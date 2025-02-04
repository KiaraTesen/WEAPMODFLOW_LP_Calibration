import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')

#path_results = r'D:\2_PaperII_LP'
#path_results = r'C:\Users\aimee\OneDrive - Universidad Católica de Chile\II-Paper_SimulationOptimizationModel_LP\ResultadosPrueba'
path_results = r'C:\Users\aimee\Desktop\Github\WEAPMODFLOW_LP_Calibration\Results'

exp = ['L_E1']
vms = list(range(2, 7)) + list(range(8, 17))          # 15 workers (De la 2 a la 16)
iter = range(70)

df_r = pd.DataFrame(index = list(iter))
for i in exp:
    for j in vms:
        df = pd.read_csv(os.path.join(i, 'ADPSO_CL_register_vm' + str(j) + '.csv'))
        #print(df)

        df_r['y_' + str(i) + '_vm_' + str(j)] = df['pob_y']

df_r_lg = np.log(df_r)
df_r_lg.to_csv(os.path.join(i, str(i) + '.csv'))
"""
#---    Confidence Intervals
df_reg = pd.DataFrame(index = ['Upper CI - 95%', 'Lower CI - 95%', 'Mean'])

for n in iter:
    df_value = df_r_lg.transpose().iloc[:,n]
    df_value = df_value.dropna()

    CI = st.norm.interval(0.95, loc = np.mean(df_value), scale = st.sem(df_value))
    mean_value = np.mean(df_value)
    Lower_CI, Upper_CI = CI[0], CI[1]

    df_reg.loc['Upper CI - 95%', str(n)] = Upper_CI
    df_reg.loc['Lower CI - 95%', str(n)] = Lower_CI
    df_reg.loc['Mean', str(n)] = mean_value

df_reg_T = df_reg.transpose()
print(df_reg_T)

#---    GRAPH
L_bound = 5
U_bound = 20

fs_title = 21
fs_others = 18

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(range(len(df_reg_T)), df_reg_T.loc[:, 'Upper CI - 95%'], color = 'black', linewidth = 0.75, linestyle = 'dashed', label = 'Upper CI - 95%')
ax.plot(range(len(df_reg_T)), df_reg_T.loc[:, 'Lower CI - 95%'], color = 'black', linewidth = 0.75, linestyle = 'dotted', label = 'Lower CI - 95%')
ax.plot(range(len(df_reg_T)), df_reg_T.loc[:, 'Mean'], color = '#A52A2A', linewidth = 0.75, linestyle = 'solid', label = 'Mean')

xlim = len(iter)
plt.xticks(range(0, xlim, 1), fontsize = fs_others)
plt.xlim(0, xlim)
plt.yticks(fontsize = fs_others)
plt.ylim(L_bound, U_bound)

title = 'NP = 15, ADPSO-CL'
plt.title(str(title), fontsize = fs_title, weight = 'bold')
plt.xlabel('Iterations', fontsize = fs_others, weight = 'bold')
plt.ylabel('log E', fontsize = fs_others, weight = 'bold')
plt.legend(loc = 'upper right', fontsize = fs_others)

plt.savefig('LogE_vs_iter_E3.png', dpi = 1200)
plt.savefig('LogE_vs_iter_E3.eps', format = 'eps', dpi = 1200)
plt.clf
"""
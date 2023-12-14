import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')

path_results = r'D:\2_PaperII_LP'

exp = ['E1']
vms = range(2, 22)
iter = range(27)

df_r = pd.DataFrame(index = list(iter))
for i in exp:
    for j in vms:
        df = pd.read_csv(os.path.join(path_results, i, 'vm' + str(j), 'ADPSO_CL_register_vm' + str(j) + '.csv'))
        #print(df)

        df_r['y_' + str(i) + '_vm_' + str(j)] = df['pob_y']

df_r_lg = np.log(df_r)
df_r_lg.to_csv('E1.csv')
#print(df_r)
#print(df_r_lg)

#plt.plot(df_r_lg)
#plt.show()
#plt.clf()

#---    Confidence Intervals
df_reg = pd.DataFrame(index = ['Upper CI - 95%', 'Lower CI - 95%', 'Mean'])

for n in iter:
    df_value = df_r_lg.transpose().iloc[:,n]
    df_value = df_value.dropna()
    #print(df_value)

    CI = st.norm.interval(alpha = 0.95, loc = np.mean(df_value), scale = st.sem(df_value))
    mean_value = np.mean(df_value)
    Lower_CI, Upper_CI = CI[0], CI[1]

    df_reg.loc['Upper CI - 95%', str(n)] = Upper_CI
    df_reg.loc['Lower CI - 95%', str(n)] = Lower_CI
    df_reg.loc['Mean', str(n)] = mean_value

df_reg_T = df_reg.transpose()
print(df_reg_T)

#---    GRAPH
L_bound = 5
U_bound = 8

fs_title = 21
fs_others = 18

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(range(len(df_reg_T)), df_reg_T.loc[:, 'Upper CI - 95%'], color = 'black', linewidth = 0.75, linestyle = 'dashed', label = 'Upper CI - 95%')
ax.plot(range(len(df_reg_T)), df_reg_T.loc[:, 'Lower CI - 95%'], color = 'black', linewidth = 0.75, linestyle = 'dotted', label = 'Lower CI - 95%')
ax.plot(range(len(df_reg_T)), df_reg_T.loc[:, 'Mean'], color = '#A52A2A', linewidth = 0.75, linestyle = 'solid', label = 'Mean')

xlim = len(iter)
plt.xticks(range(0, xlim, 20), fontsize = fs_others)
plt.xlim(0, xlim)
plt.yticks(fontsize = fs_others)
plt.ylim(L_bound, U_bound)

title = 'NP = 20, ADPSO-CL'
plt.title(str(title), fontsize = fs_title, weight = 'bold')
plt.xlabel('Iterations', fontsize = fs_others, weight = 'bold')
plt.ylabel('log E', fontsize = fs_others, weight = 'bold')
plt.legend(loc = 'upper right', fontsize = fs_others)

plt.savefig('LogE_vs_iter.png', dpi = 1200)
plt.savefig('LogE_vs_iter,eps', format = 'eps', dpi = 1200)
plt.clf
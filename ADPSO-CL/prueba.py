import h5py

VM = 2
ITERATION = 0
FINAL_ITERATION = 200
n_var = 30000

with h5py.File('ADPSO_CL_register_vm' + str(VM) + '.h5', 'w') as f:
    iter_h5py = f.create_dataset("iteration", (FINAL_ITERATION, 1))
    pob_x_h5py = f.create_dataset("pob_x", (FINAL_ITERATION, n_var))
    pob_y_h5py = f.create_dataset("pob_y", (FINAL_ITERATION, 1))
    pob_v_h5py = f.create_dataset("pob_v", (FINAL_ITERATION, n_var))
    pob_x_best_h5py = f.create_dataset("pob_x_best", (FINAL_ITERATION, n_var))
    pob_y_best_h5py = f.create_dataset("pob_y_best", (FINAL_ITERATION, 1))
    pob_w_h5py = f.create_dataset("w", (FINAL_ITERATION, 1))

#---    Iteration register
    iter_h5py[0] = ITERATION
    pob_w_h5py[0] = 0.5
f.close()

with h5py.File('ADPSO_CL_register_vm' + str(VM) + '.h5', 'r') as f:
    x = f["w"][:]
print(x)
import torch
import os
import os.path as op
import numpy as np
import hnn_core
from hnn_core import simulate_dipole, Network, read_params, JoblibBackend
import matplotlib as mpl
import matplotlib.pyplot as plt
import sbi.utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
import multiprocessing
import datetime
import dill
import joblib
from joblib import Parallel, delayed
from dask.distributed import Client
time_stamp = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")

client = Client(processes=False)

save_suffix = 'beta_event' + '_' + time_stamp
save_path = '../../data/beta/prerun_simulations/' + time_stamp + '/'
os.mkdir(save_path)

params_fname = '../../data/beta/params/beta_param.param'

num_simulations = 10

prior_dict = {'dipole_scalefctr': (60000, 200000),
 't_evprox_1': (225, 255),
 'sigma_t_evprox_1': (10, 50),
 'numspikes_evprox_1': (1, 20),
 'gbar_evprox_1_L2Pyr_ampa': (1e-06, 0.0005),
 'gbar_evprox_1_L5Pyr_ampa': (1e-06, 0.0005),
 't_evdist_1': (235, 255),
 'sigma_t_evdist_1': (5, 30),
 'numspikes_evdist_1': (1, 20),
 'gbar_evdist_1_L2Pyr_ampa': (1e-06, 0.0005),
 'gbar_evdist_1_L5Pyr_ampa': (1e-06, 0.0005)}

param_low = [float(item[0]) for key, item in prior_dict.items()]
param_high = [float(item[1]) for key, item in prior_dict.items()]
prior = utils.BoxUniform(low=torch.tensor(param_low), high=torch.tensor(param_high))

theta_samples = prior.sample_n(num_simulations)

def dill_save(save_object, save_prefix, save_suffix, save_path, extension='.pkl'):
    save_file = open(save_path + save_prefix + '_' + save_suffix + extension, 'wb')
    dill.dump(save_object, save_file)
    save_file.close()

dill_save(params_fname, 'params_fname', save_suffix, save_path)
dill_save(prior, 'prior', save_suffix, save_path)
dill_save(theta_samples, 'theta_samples', save_suffix, save_path)

class HNNSimulator:
    def __init__(self, params_fname, prior_dict):
        self.params = read_params(params_fname)
        self.params['tstop'] = 30
        self.param_names = list(prior_dict.keys())

    def __call__(self, new_param_values):
        new_params = dict(zip(self.param_names, new_param_values.detach().cpu().numpy()))
        self.params.update(new_params)

        net = Network(self.params)
        with JoblibBackend(n_jobs=1):
            dpl = simulate_dipole(net, n_trials=1)

        summstats = torch.as_tensor(dpl[0].data['agg'])
        return summstats


hnn_simulator = HNNSimulator(params_fname,prior_dict)
sbi_simulator, sbi_prior = prepare_for_sbi(hnn_simulator, prior)

# def run_simulator(simulator, theta, sim_idx):
#     new_param_values = theta_samples[sim_idx,:]
#     dpl = simulator(new_param_values)

#     dpl_name = save_path + save_prefix + '_' + save_suffix + 'dpl_sim{}'.format(sim_idx) + '.txt'
#     param_name = save_path + save_prefix + '_' + save_suffix + 'theta_sim{}'.format(sim_idx) + '.txt'

#     np.savetxt(dpl_name, dpl, delimiter=',')
#     np.savetxt(param_name, new_param_values, delimiter=',')

with joblib.parallel_backend('dask'):    
        # Parallel(verbose=100,n_jobs=12)(delayed(run_simulator)(sbi_simulator, theta_samples[sim_idx,:], sim_idx) for sim_idx in range(num_simulations))
        theta, x = simulate_for_sbi(sbi_simulator, proposal=sbi_prior, num_simulations=num_simulations, num_workers=12)


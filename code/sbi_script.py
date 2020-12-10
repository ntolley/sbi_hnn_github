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
time_stamp = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")

save_suffix = 'beta_event_test_t10000_500ms' + '_' + time_stamp
save_path = '/users/ntolley/scratch/sbi/beta_event_test/' + time_stamp + '/'
os.mkdir(save_path)


param_names = ['dipole_scalefctr', 't_evprox_1', 'sigma_t_evprox_1', 'numspikes_evprox_1', 'gbar_evprox_1_L2Pyr_ampa', 'gbar_evprox_1_L5Pyr_ampa', 't_evdist_1', 'sigma_t_evdist_1', 'numspikes_evdist_1', 'gbar_evdist_1_L2Pyr_ampa', 'gbar_evdist_1_L5Pyr_ampa']
default_values = [3000, 26.61, 2.47, 1, 0.01525, 0.00865, 63.53, 3.85, 1, 7e-06, 0.1423]
param_low = [60000, 125, 10, 1, 1e-06, 1e-06, 135, 5, 1, 1e-06, 1e-06]
param_high = [200000, 155, 50, 20, 0.0005, 0.0005, 155, 30, 20, 0.0005, 0.0005]
prior = utils.BoxUniform(low=torch.tensor(param_low), high=torch.tensor(param_high))

def dill_save(save_object, save_prefix, save_suffix, save_path):
    save_file = open(save_path + save_prefix + '_' + save_suffix, 'wb')
    dill.dump(save_object, save_file)
    save_file.close()


class HNNSimulator:
    def __init__(self):
        hnn_core_root = op.dirname(hnn_core.__file__)
        params_fname = '/users/ntolley/Jones_Lab/sbi_hnn/051220_peribeta_basedonpostbeta16_6_opt_0.05_timestep_opt2_10ms_smoothing_opt_1trials_opt.param'
        self.params = read_params(params_fname)

        self.param_names = ['dipole_scalefctr', 't_evprox_1', 'sigma_t_evprox_1', 'numspikes_evprox_1', 'gbar_evprox_1_L2Pyr_ampa', 'gbar_evprox_1_L5Pyr_ampa', 't_evdist_1', 'sigma_t_evdist_1', 'numspikes_evdist_1', 'gbar_evdist_1_L2Pyr_ampa', 'gbar_evdist_1_L5Pyr_ampa']

    def __call__(self, new_param_values):
        new_params = dict(zip(self.param_names, new_param_values.detach().cpu().numpy()))
        self.params.update(new_params)

        net = Network(self.params)
        with JoblibBackend(n_jobs=1):
            dpl = simulate_dipole(net, n_trials=1)

        summstats = torch.as_tensor(dpl[0].data['agg'])
        return summstats


hnn_simulator = HNNSimulator()
simulator, prior = prepare_for_sbi(hnn_simulator, prior)
inference = SNPE(prior)

dill_save(simulator, 'simulator', save_suffix, save_path)
dill_save(prior, 'prior', save_suffix, save_path)
dill_save(inference, 'inference', save_suffix, save_path)

theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=10000, num_workers=48)
dill_save(theta, 'theta', save_suffix, save_path)
dill_save(x, 'x', save_suffix, save_path)

density_estimator = inference.append_simulations(theta, x).train()
dill_save(density_estimator, 'density_estimator', save_suffix, save_path)

posterior = inference.build_posterior(density_estimator)
dill_save(posterior, 'posterior', save_suffix, save_path)
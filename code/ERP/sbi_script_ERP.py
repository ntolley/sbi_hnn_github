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

save_suffix = 'ERPYes_t1000' + '_' + time_stamp
save_path = '/users/ntolley/scratch/sbi/ERP/' + time_stamp + '/'
os.mkdir(save_path)


param_names = []
default_values = []
param_low = []
param_high = []
prior = utils.BoxUniform(low=torch.tensor(param_low), high=torch.tensor(param_high))

def dill_save(save_object, save_prefix, save_suffix, save_path):
    save_file = open(save_path + save_prefix + '_' + save_suffix, 'wb')
    dill.dump(save_object, save_file)
    save_file.close()


class HNNSimulator:
    def __init__(self):
        hnn_core_root = op.dirname(hnn_core.__file__)
        params_fname = '/users/ntolley/Jones_Lab/sbi_hnn/'
        self.params = read_params(params_fname)

        self.param_names = []

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

theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=1000, num_workers=48)
dill_save(theta, 'theta', save_suffix, save_path)
dill_save(x, 'x', save_suffix, save_path)

density_estimator = inference.append_simulations(theta, x).train()
dill_save(density_estimator, 'density_estimator', save_suffix, save_path)

posterior = inference.build_posterior(density_estimator)
dill_save(posterior, 'posterior', save_suffix, save_path)
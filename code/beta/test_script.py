# from joblib.externals.loky import set_loky_pickler
# set_loky_pickler("dill")
import torch
import os
import os.path as op
import numpy as np
import hnn_core
from hnn_core import simulate_dipole, Network, read_params, JoblibBackend
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import sbi.utils as utils
#from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
import multiprocessing
import datetime
import dill
import joblib
# from distributed import Client
from joblib import Parallel, delayed

params_fname = '/home/ntolley/Jones_Lab/sbi_hnn/data/beta/params/beta_param.param'

num_simulations = 1

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

theta_samples = prior.sample((num_simulations,))


class HNNSimulator:
    def __init__(self, params_fname, prior_dict):
        self.params = read_params(params_fname)
        #self.params['tstop'] = 30
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
#sbi_simulator, sbi_prior = prepare_for_sbi(hnn_simulator, prior)

def run_simulator(simulator, theta, sim_idx):
    dpl = simulator(theta)


hnn_simulator(theta_samples[0,:])


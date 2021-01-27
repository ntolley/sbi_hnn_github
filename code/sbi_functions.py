import numpy as np

def format_spikes(spike_times, spike_gids):
    unique_gids = np.unique([gid for l in spike_gids for gid in l[0]])
    num_sims = len(spike_gids)
    spike_times_list = []
    for gid in unique_gids:
        unit_trains = []
        for sim_idx in range(num_sims):
            gid_mask = np.in1d(spike_gids[sim_idx][0], gid)
            unit_trains.append(np.array(spike_times[sim_idx][0])[gid_mask])
        spike_times_list.append(unit_trains)

class HNNSimulator:
    def __init__(self, params_fname, prior_dict):
        if 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']
            
        import hnn_core
        from hnn_core import simulate_dipole, Network, read_params, JoblibBackend, MPIBackend
        self.params = read_params(params_fname)
        #self.params['tstop'] = 100
        self.param_names = list(prior_dict.keys())

    def __call__(self, new_param_values):
        new_params = dict(zip(self.param_names, new_param_values.detach().cpu().numpy()))
        self.params.update(new_params)

        net = Network(self.params)
        with JoblibBackend(n_jobs=1):
            dpl = simulate_dipole(net, n_trials=1)

        summstats = dpl[0].data['agg']
        spike_times = net.cell_response.spike_times
        spike_gids = net.cell_response.spike_gids
        spike_types = net.cell_response.spike_types
        return summstats, spike_times, spike_gids, spike_types


#sbi_simulator, sbi_prior = prepare_for_sbi(hnn_simulator, prior)
#params = read_params(params_fname)
def run_simulator(theta, params_fname, prior_dict, sim_idx):
    hnn_simulator = HNNSimulator(params_fname,prior_dict)
    dpl, spike_times, spike_gids, spike_types = hnn_simulator(theta)
    return (dpl, spike_times, spike_gids, spike_types)
    
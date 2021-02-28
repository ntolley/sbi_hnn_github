import numpy as np
import os
import hnn_core
from hnn_core import simulate_dipole, Network, read_params, JoblibBackend, MPIBackend
import torch
from torch import nn
import torch.nn.functional as F

def format_spikes(spike_times, spike_gids, trial_form = True):
    num_sims = len(spike_gids)
    spike_times_list = []
    for gid in unique_gids:
        unit_trains = []
        for sim_idx in range(num_sims):
            gid_mask = np.in1d(spike_gids[sim_idx][0], gid)
            unit_trains.append(np.array(spike_times[sim_idx][0])[gid_mask])
        spike_times_list.append(unit_trains)
    return spike_times_list

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
        self.params['numspikes_evprox_1'] = self.params['numspikes_evprox_1'].astype(int)
        self.params['numspikes_evdist_1'] = self.params['numspikes_evdist_1'].astype(int)  

        net = Network(self.params, add_drives_from_params=True)
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

#Dataset class used to load simulated dipoles
class HNNDataset(torch.utils.data.Dataset):
    def __init__(self, dpl, theta, device):
        self.dpl = torch.as_tensor(dpl).float()
        self.theta = torch.as_tensor(theta).float()

    def __len__(self):
        return self.dpl.shape[0]

    def __getitem__(self, idx):
        return self.dpl[idx,:], self.theta[idx, :]
    
#Bottlenecked autoencoder with user defined layers and sizes
class autoencoder_linear(nn.Module):
    def __init__(self, input_size, layer_size, device):
        super(autoencoder_linear, self).__init__()
        self.input_size,  self.layer_size = input_size, layer_size
        self.latent_dim = layer_size[-1]
        self.device = device

        #List layer sizes
        self.encoder_hidden = np.concatenate([[input_size], layer_size])
        self.decoder_hidden = self.encoder_hidden[::-1]
        
        #Compile layers into lists
        self.encoder_list = nn.ModuleList(
            [nn.Linear(in_features=self.encoder_hidden[idx], out_features=self.encoder_hidden[idx+1]).to(self.device) for idx in range(len(self.encoder_hidden)-1)] )

        self.decoder_list = nn.ModuleList(
            [nn.Linear(in_features=self.decoder_hidden[idx], out_features=self.decoder_hidden[idx+1]).to(self.device) for idx in range(len(self.decoder_hidden)-1)] )
        
 
    def forward(self, x):
        #Encoding step
        for idx in range(len(self.encoder_list)):
            x = F.tanh(self.encoder_list[idx](x))
        latent = x.clone() #Store final output of encoding stack as latent variable
        #Decoding step
        for idx in range(len(self.decoder_list)-1):
            x = F.tanh(self.decoder_list[idx](x))
        x = self.decoder_list[-1](x) #Only use linear activation for final output

        return x, latent

#LSTM/GRU autoencoder with attention
class autoencoder_gru(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, window_size, step_size, dropout, device):
        super(autoencoder_gru, self).__init__()
        self.input_size,  self.hidden_size, self.layer_size = input_size, hidden_size, layer_size
        self.window_size, self.step_size = window_size, step_size
        self.dropout, self.device = dropout, device
        self.num_gru_layers = 2
        self.latent_dim = layer_size[-1]

        #Layer size list for linear layers
        self.encoder_hidden = np.concatenate([[np.round((hidden_size*window_size)/step_size).astype(int)], layer_size])
        self.decoder_hidden = self.encoder_hidden[::-1]

        
        #Final input/output layers
        self.gru_encoder = nn.GRU(input_size, hidden_size, num_layers=self.num_gru_layers, batch_first=True, dropout=dropout)   
        self.gru_decoder = nn.GRU(hidden_size, hidden_size, num_layers=self.num_gru_layers, batch_first=True, dropout=dropout)   
        self.linear_decoder = nn.Linear(in_features=np.round((hidden_size*window_size)/step_size).astype(int), out_features=input_size)

        #Compile layers into lists
        self.encoder_list = nn.ModuleList(
            [nn.Linear(in_features=self.encoder_hidden[idx], out_features=self.encoder_hidden[idx+1]) for idx in range(len(self.encoder_hidden)-1)] )

        self.decoder_list = nn.ModuleList(
            [nn.Linear(in_features=self.decoder_hidden[idx], out_features=self.decoder_hidden[idx+1]) for idx in range(len(self.decoder_hidden)-1)] )
        
 
    def forward(self, x):
        input_shape = x.size()
        batch_size, seq_len, num_features = input_shape

        # Initializing hidden state for first input using method defined below
        hidden_encoder = self.init_hidden(batch_size)
        hidden_decoder = self.init_hidden(batch_size)

        #Encoding step
        x, hidden_encoder = self.gru_encoder(x, hidden_encoder)
        x = x.contiguous().view(batch_size, -1)
        for idx in range(len(self.encoder_list)):
            x = F.tanh(self.encoder_list[idx](x))
        latent = x.unsqueeze(1).clone() #Store final output of encoding stack as latent variable

        #Decoding step
        for idx in range(len(self.decoder_list)):
            x = F.tanh
            (self.decoder_list[idx](x))
        x = x.view(batch_size,-1,self.hidden_size)
        x, hidden_decoder = self.gru_decoder(x, hidden_decoder)

        x = x.contiguous().view(batch_size,-1)
        x = self.linear_decoder(x)
        x = x.view(batch_size,num_features).unsqueeze(1)
        return x, latent

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        weight = next(self.parameters()).data.to(self.device)
        #GRU initialization
        hidden = weight.new(self.num_gru_layers, batch_size, self.hidden_size).zero_().to(self.device)

        return hidden

#Simple feedforward ANN for decoding kinematics
class model_ann(nn.Module):
    def __init__(self, input_size, output_size, layer_size):
        super(model_ann, self).__init__()
        self.input_size,  self.layer_size, self.output_size = input_size, layer_size, output_size

        #List layer sizes
        self.layer_hidden = np.concatenate([[input_size], layer_size, [output_size]])
        
        #Compile layers into lists
        self.layer_list = nn.ModuleList(
            [nn.Linear(in_features=self.layer_hidden[idx], out_features=self.layer_hidden[idx+1]) for idx in range(len(self.layer_hidden)-1)] )        
 
    def forward(self, x):
        #Encoding step
        for idx in range(len(self.layer_list)):
            x = F.tanh(self.layer_list[idx](x))

        return x

#GRU architecture
class model_gru(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout, device, bidirectional=False):
        super(model_gru, self).__init__()

        #multiplier based on bidirectional parameter
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers * num_directions
        self.device = device
        self.dropout = dropout
        self.bidirectional = bidirectional

        #Defining the layers
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)   

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim*num_directions, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.gru(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous()
        out = self.fc(out)
        return out
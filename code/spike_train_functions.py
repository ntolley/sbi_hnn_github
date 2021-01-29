import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
import quantities as pq
from neo.core import SpikeTrain
import neo
import elephant
from elephant.spike_train_dissimilarity import victor_purpura_dist
from joblib import Parallel, delayed
import multiprocessing
import numba

#Compute Victor-Purpura Spike Train Distance
@numba.njit()
def vpSpikeTimeDist(s1,s2,q):
    s1_size, s2_size = len(s1), len(s2)
    dmat = np.zeros((s1_size+1,s2_size+1))
    dmat[:,0] = np.arange(0,s1_size+1)
    dmat[0,:] = np.arange(0,s2_size+1)

    for r in np.arange(1,s1_size+1):
        for c in np.arange(1,s2_size+1):
            d = np.abs(s1[r-1] - s2[c-1])
            dmat[r,c] = np.nanmin( [ dmat[r,c-1]+1, dmat[r-1,c]+1, dmat[r-1,c-1] + (q*d)] )

    d = dmat[s1_size,s2_size]
    return d, dmat

@numba.njit()
def vpTrialSimilarityMatrix(unit_data,q=0.5):
    num_trials = len(unit_data)
    trial_similarity_matrix = np.zeros((num_trials,num_trials))

    for t1 in range(num_trials):
        for t2 in range(num_trials):
            s1 = unit_data[t1]
            s2 = unit_data[t2]
            d, dmat = vpSpikeTimeDist(s1,s2,q)

            trial_similarity_matrix[t1,t2] = d

    return trial_similarity_matrix

@numba.njit()
def simnets_dist(u1, u2):
    return np.corrcoef(u1, u2)[0,1]

@numba.njit()
def simnets_mat(trial_sim):
    num_gids = len(trial_sim)
    unit_sim = np.zeros((num_gids, num_gids))
    for idx2 in range(num_gids):
        for idx1 in range(num_gids):
            unit_sim[idx1,idx2] = simnets_dist(trial_sim[idx1], trial_sim[idx2])
    return unit_sim

#Format timestamps columns into list of size num_trials, empty lists inserted on trials where no spikes fired
def grouped_ts_spikes_df(spikes_df, q=0.5):
    spikes_df = spikes_df_all.groupby('gid').agg(lambda x: tuple(x)).applymap(list).reset_index()
    spikes_df['type'] = spikes_df['type'].map(lambda x: x[0])
    spikes_df['detected'] = spikes_df['detected'].map(lambda x: x[0])

    #Format timestamps columns into list of size num_trials, empty lists inserted on trials where no spikes fired
    grouped_ts = []
    trial_sim = []
    for gid in np.unique(spikes_df['gid'].values):
        gid_ts = []
        gid_trial_idx = 0 
        for trial_idx in range(num_samples):
            if trial_idx in spikes_df['trial'][spikes_df['gid'] == gid].values[0]:
                gid_ts.append(np.array(spikes_df['timestamps'][spikes_df['gid'] == gid].values[0][gid_trial_idx]))
                gid_trial_idx = gid_trial_idx + 1
            else:
                gid_ts.append(np.array([]))
        grouped_ts.append(np.array(gid_ts))
        trial_sim.append(vpTrialSimilarityMatrix(gid_ts,q).reshape((1,-1)).squeeze())
    spikes_df['grouped_ts'] = grouped_ts
    spikes_df['trial_sim'] = trial_sim
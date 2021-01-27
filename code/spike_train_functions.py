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

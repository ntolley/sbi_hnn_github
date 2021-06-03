import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spike_train_functions
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import argrelextrema
import quantities as pq
from neo.core import SpikeTrain
import neo
import elephant
from elephant.spike_train_dissimilarity import victor_purpura_dist
from joblib import Parallel, delayed
import multiprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
num_cores = multiprocessing.cpu_count()

#Used to manually spike information to work with other functions in this module
class tmpSpikes:
    def __init__(self, spike_times, spike_gids, spike_types):
        self.spike_times = spike_times
        self.spike_gids = spike_gids
        self.spike_types = spike_types


def spikes_df_simnets(spikes_df, event, start, stduration, q, perplexity=40, learning_rate=5, filter_nans=False):
    ts = spikes_df.timestamps.values
    units = np.unique(spikes_df['gid'].values)

    st_dict = spike_train_functions.spikes_df_train_trials(spikes_df, event, start, stduration, shift_idx=1)
    unit_similarity_matrix, trial_similarity_stack = spike_train_functions.vpUnitSimilarityMatrix(st_dict,q,num_cores,'ms')
    if filter_nans:
        unit_similarity_matrix = np.nan_to_num(unit_similarity_matrix)

    sim_embedded = TSNE(n_components=2, perplexity=perplexity, random_state=3, learning_rate=learning_rate).fit_transform(unit_similarity_matrix)

    TSNE_col = np.array([sim_embedded[np.where(units==gid),:] for gid in spikes_df['gid'].values]).squeeze()
    print(TSNE_col.shape)
    spikes_df['TSNE1'] = TSNE_col[:,0] 
    spikes_df['TSNE2'] = TSNE_col[:,1] 

    return unit_similarity_matrix, trial_similarity_stack

#Compute simnets over a sliding window defined by sweep
def sliding_spikes_df_simnets(spikes_df, event, start, stduration, q, sweep, filter_nans):

    st_dict_all = {}
    for shift_idx, shift in enumerate(sweep):
        #Calculate simnets for each shift
        event_shift = event + shift
        st_dict = spike_train_functions.spikes_df_train_trials(spikes_df, event_shift, start, stduration, shift_idx)
        st_dict_all.update(st_dict)

    unit_similarity_matrix, trial_similarity_stack = spike_train_functions.vpUnitSimilarityMatrix(st_dict_all,q,num_cores)
    if filter_nans:
        unit_similarity_matrix = np.nan_to_num(unit_similarity_matrix)
    
    return  unit_similarity_matrix, trial_similarity_stack, st_dict_all
    

#Note that gids that don't spike are currently ignored (don't show up in spikes.spike_gids)
def make_spikes_df(spikes, num_trials=None):
    if num_trials is None:
        num_trials = len(spikes.times)
        
    df_list = []
    for trial in range(num_trials):
        trial_df = pd.DataFrame({'timestamps': spikes.spike_times[trial], 'gid': spikes.spike_gids[trial], 'type': spikes.spike_types[trial]})
        ts_col = trial_df.groupby('gid').timestamps.apply(np.array).reset_index()
        types_col = trial_df.groupby('gid').type.unique().apply(lambda x: x[0]).reset_index()
        trial_df = pd.concat([types_col.set_index('gid'),ts_col.set_index('gid')], axis=1).reset_index()
        trial_df = trial_df[np.in1d(trial_df['type'], ['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal'])]
        trial_df['trial'] = np.repeat(trial, len(trial_df))

        df_list.append(trial_df)
    spikes_df = pd.concat(df_list)
    return spikes_df

def plot_raster(spikes, cell_types=['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal'], colors=['C0', 'C1', 'C2', 'C3'], cmap = None, trial=None):
    #Plot spikes raster
    if trial==None:
        spike_times = np.array(sum(spikes.spike_times, []))
        spike_types = np.array(sum(spikes.spike_types, []))
        spike_gids = np.array(sum(spikes.spike_gids, []))
    else:
        spike_times = np.array(spikes.spike_times[trial])
        spike_types = np.array(spikes.spike_types[trial])
        spike_gids = np.array(spikes.spike_gids[trial])

    ypos = 0
    for idx, cell_type in enumerate(cell_types):
        gid_list = spike_gids[spike_types == cell_type]
        gid_times = spike_times[np.in1d(spike_gids, gid_list)]
        
        unique_gids = np.unique(gid_list)
        gid_ypos = gid_list.copy()
        #Map y coordinate to specific gids
        for gid in unique_gids:
            gid_ypos[gid_list==gid] = np.repeat(np.where(unique_gids==gid)[0], gid_ypos[gid_list==gid].size)
        gid_ypos = gid_ypos - ypos

        if cmap is None:
            plt.scatter(gid_times,gid_ypos, color=colors[idx], s=1)
        else:
            cmap_type = cmap[np.in1d(spike_gids, gid_list)]
            plt.scatter(gid_times,gid_ypos, c=cmap_type, s=1)

        ypos = ypos - len(unique_gids)


    plt.xlabel('Time (ms)')
    plt.ylabel('Single Units')

    ax = plt.gca()
    return ax

def plot_raster_cluster(spikes, gids, cluster_labels, colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']):
    cluster_map = {gids[idx] : cluster_labels[idx] for idx in range(len(gids))}
    cell_types = ['L2_basket', 'L2_pyramidal', 'L5_basket', 'L5_pyramidal']

    #Filter out feeds and evoked inputs
    spike_gids = np.array(sum(spikes._spike_gids, []))
    gid_mask = np.in1d(spike_gids, gids)
    spike_gids = spike_gids[gid_mask]
    spike_times = np.array(sum(spikes._spike_times, []))[gid_mask]
    spike_types = np.array(sum(spikes._spike_types, []))[gid_mask]

    spike_clusters = np.array([cluster_map[gid] for gid in spike_gids])
    
    ypos = 0
    #Plot with gids sorted by cluster label
    for idx, label in enumerate(np.unique(cluster_labels)):
        gid_list = spike_gids[spike_clusters == label]
        gid_times = spike_times[np.in1d(spike_gids, gid_list)]
        unique_gids = np.unique(gid_list)
        gid_ypos = gid_list.copy()
        #Map y coordinate to specific gids
        for gid in unique_gids:
            gid_ypos[gid_list==gid] = np.repeat(np.where(unique_gids==gid)[0], gid_ypos[gid_list==gid].size)
        gid_ypos = gid_ypos - ypos

        plt.scatter(gid_times,gid_ypos, color=colors[idx],s=1)
        ypos = ypos - len(unique_gids)


    plt.xlabel('Time (ms)')
    plt.ylabel('Single Units')

    ax = plt.gca()
    return ax

#Create list with same shape as spike_gids but replaced with cluster labels
# **Todo, update to retain trial structure
def spike_clusters(spikes, gids, cluster_labels):
    cluster_map = {gids[idx] : cluster_labels[idx] for idx in range(len(gids))}
    spike_gids = np.array(sum(spikes._spike_gids, []))

    spike_clusters = []
    for gid in spike_gids:
        if gid in gids:
            spike_clusters.append(cluster_map[gid])
        else:
            spike_clusters.append(-1)

    return np.array(spike_clusters)

def plot_hist(spikes, ax1):
    spike_times = np.array(sum(spikes._times, []))
    spike_types_data = np.array(sum(spikes._types, []))
    unique_types = np.unique(spike_types_data)
    spike_types_mask = {s_type: np.in1d(spike_types_data, s_type)
                            for s_type in unique_types}
    #Plot input histogram
    bins = np.linspace(0, spike_times[-1], 50)

    ax1.hist(spike_times[spike_types_mask['evprox1']], bins, label='Proximal', color='C7')
    ax1.hist(spike_times[spike_types_mask['evprox2']], bins, label='Proximal', color='C7')
    ax1.set_xlim([0,170])
    ax1.set_ylabel('Current Dipole (a.u.)')
    ax1.yaxis.set_ticks([])
    # axs[0,1].annotate('B', (0.025, 0.83), xycoords='axes fraction', va='center', fontweight='bold', fontsize=25)

    #ax1.set_ylim([0,180])

    ax2 = ax1.twinx()
    ax2.hist([], bins, label='Proximal', color='C7')
    ax2.hist(spike_times[spike_types_mask['evdist1']], bins, label='Distal', color='C9')
    ax2.yaxis.set_ticks([])
    #ax2.set_ylim([0,380])
    ax2.invert_yaxis()
    # ax2.legend()

    #Overlay current dipole 
    # ax3 = ax1.twinx()
    # ax3.plot(dpls[0].times, dpls[0].data['agg'], color='k')
    # ax3.hist([], bins, label='Distal Input', color='C9')
    # ax3.hist([], bins, label='Proximal Input', color='C7')

    # ax3.yaxis.set_ticks([])
    return ax1

def kmeans_silhouette(X, range_n_clusters):  
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')

    plt.show()
import numpy as np
import sys
from scipy.ndimage import label
from typing import Union, Tuple, List, Iterable

from run_main import STRUCTURING_ELEMENT

def rank_cluster_map(R0_map:np.ndarray, get_ranks: Union[int, Iterable]) -> Tuple[np.ndarray, List, List]:
    """
    Find connected clusters and return rank-ordered size along with corresponding  id.
    If get ranks is an int, the rank upto and included the value `get_ranks' is returned.
    If a tuple is supplied, just those ranks will be returned.
    """

    R0_clusters = label_connected(R0_map)[0]
    cluster_sizes, cluster_ids = cluster_freq_count(labeled=R0_clusters)

    if isinstance(get_ranks, int):
        cluster_ids = cluster_ids[:get_ranks]
        cluster_sizes = cluster_sizes[:get_ranks]

    else:
        try:
            cluster_ids = [cluster_ids[rank-1] for rank in get_ranks]
            cluster_sizes = [cluster_sizes[rank-1] for rank in get_ranks]
        except Exception:
            sys.exit(f'Error type {type(get_ranks)} is not iterable')

    R0_clusters_ = np.zeros_like(R0_clusters)
    R0_clusters = R0_clusters * np.isin(R0_clusters, cluster_ids)  # select top n clusters
    for rank, id in enumerate(cluster_ids):
        R0_clusters_[np.where(R0_clusters == id)] = rank + 1
        cluster_ids[rank] = rank + 1

    return R0_clusters_, cluster_sizes, cluster_ids


def label_connected(R0_map:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labeled, num_comp = label(R0_map, STRUCTURING_ELEMENT)
    return labeled, num_comp

def cluster_freq_count(labeled:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Count the number of sites per cluster and rank-order for each unique cluster, find the corresponding number of
    elements.
    """
    number_of_clusters = np.unique(labeled).shape[0]  # The number of individual clusters
    cluster_ids = np.arange(1, number_of_clusters, 1)
    cluster_counts = np.zeros_like(cluster_ids)
    for index, id in enumerate(cluster_ids):
        cluster_counts[index] = len(np.where(labeled == id)[0])  # the number of sites inside a given cluster

    idx = np.argsort(cluster_counts)  # sort index by cluster size
    return cluster_counts[idx][::-1], cluster_ids[idx][::-1]  # return sorted

class Cluster_sturct():
    def __init__(self, R0_map, neighbourhood=None):
        NN_element = np.ones(shape=(3, 3))  # Nearest neighbour element
        if neighbourhood is None:
            self.NN_element = NN_element
            self.neighbourhood_type = 'Moore'
        else:
            NN_element[0, 0] = 0
            NN_element[0, 2] = 0
            NN_element[2, 0] = 0
            NN_element[0, 2] = 0
            self.neighbourhood = neighbourhood
            self.neighbourhood_type = 'Von-N'

        self.R0_map = R0_map
        self.labeled = None
        self.ranked_map = None
        self.cluster_ids = None
        self.cluster_count = None
        self.R0_applied_threshold = None


    def apply_R0_threshold(self, R0_threshold):
        self.R0_applied_threshold = np.where(self.R0_map > R0_threshold, self.R0_map, 0)
        return self.R0_applied_threshold

    def label_connected(self):
        assert self.R0_applied_threshold is not None # /ERROR, apply R0 threshold
        self.labeled = label(self.R0_applied_threshold)[0]
        self.cluster_count, self.cluster_ids = cluster_freq_count(self.labeled)

    def rank_R0_cluster_map(self, rank_N:Union[int, list]):
        """
        If called, rank the top `rank_N' clusters.
        """
        try:  # If iterable, get chosen indicies, get:  c_1, c_2, ...., c_n
            iter(rank_N)
            rank_N = [index - 1 for index in rank_N]  # start from 0th place
            plot_clusters = self.cluster_ids[rank_N]
        except:  # If int type, get up to N indices
            plot_clusters = self.cluster_ids[:rank_N]

        shape = self.labeled.shape
        ranked_map = np.zeros(shape)
        for index, cluster_id in enumerate(plot_clusters):
            # find where in UK land regions are clusters and set value
            ranked_map[np.where(self.labeled == cluster_id)] = index + 1

        self.ranked_map = ranked_map  # add ranked_map definition to class
        return self.ranked_map


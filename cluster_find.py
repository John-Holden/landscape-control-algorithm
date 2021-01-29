from scipy.ndimage import label
import numpy as np
from typing import Union


def cluster_freq_count(labeled):
    """
    Count the number of sites per cluster and rank-order for each unique cluster,
    find the corresponding number of elements and sort
    :param labeled:
    :return: arr: cluster_count 'num of lattice points in ith cluster'
             arr: cluster_id  'ith clusters numerical id'
    """
    clusters = np.sort(labeled.reshape(labeled.shape[0] * labeled.shape[1]))  # flatten
    sz = np.unique(clusters).shape[0]  # number of individual clusters (scalar)
    cluster_id, cluster_count = np.arange(1, sz, 1), np.zeros(
        sz - 1)  # define cluster id and arr to store count
    for i in clusters:  # iterate through every site of each cluster
        if i in cluster_id:  # count number of times repeated
            cluster_count[i - 1] += 1  # i - 1 --> cluster id starts from 0
        else:
            pass

    idx = np.argsort(cluster_count)  # sort index by cluster size
    cluster_count, cluster_id = cluster_count[idx][::-1], cluster_id[idx][::-1]  # sort cluster count and cluster ID
    return cluster_count, cluster_id



class Cluster_sturct():
    def __init__(self, R0_map, neighbourhood=None):
        if neighbourhood is None:
            neighbourhood = np.ones(shape=(3, 3))
            self.neighbourhood = 'Moore'
        else:
            self.neighbourhood = 'Von-N'

        self.labeled = label(np.where(R0_map > 1, R0_map, 0), neighbourhood)[0]
        self.cluster_count, self.cluster_ids = cluster_freq_count(self.labeled)
        self.ranked_map = None

    def rank_cluster_sizes(self, rank_N:Union[int, list]):
        """
        If called, rank the top `rank_N' clusters.
        """
        if isinstance(rank_N, list):  # If list type, get chosen indicies, get:  c_1, c_2, ...., c_n
            iter_ = self.cluster_ids[rank_N]
        elif isinstance(rank_N, int):
            iter_ = self.cluster_ids[:rank_N]  # If int type, get up to N indices

        shape = self.labeled.shape
        ranked_map = np.zeros(shape)
        for i, idx in enumerate(iter_):  # color top 3 (or less) clusters
            # find where in UK land regions are clusters and color
            ranked_map[np.where(self.labeled == idx)] = i + 1

        self.ranked_map = ranked_map  # add ranked_map definition to class
        return self.ranked_map


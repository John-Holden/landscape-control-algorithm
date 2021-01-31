"""
Methods related to fragmenting a domain of R0 values.
"""

import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from cluster_find import get_top_cluster_sizes

def discontinuity_detect(cluster_sizes:np.ndarray) -> bool:
    """
    Find largest increase of cluster size and return
    """
    assert len(cluster_sizes) > 1  # /ERROR TOO REW CLUSTERS DETECTED
    return np.argmax(np.gradient(cluster_sizes))


def get_alpha_steps(alpha_steps: Union[iter, float, int, str], R0_min:float, R0_max:float) -> Union[list, np.ndarray]:
    """
    Find and return what values of alpha will be iterated through.
    """
    default_number_alpha = 50
    if alpha_steps == "auto":   # default 100 steps
        return np.arange(3, R0_min, -(R0_max - R0_min)/default_number_alpha)
    else:
        try:
            iter(alpha_steps)  # if alpha is iterable, return
            return alpha_steps
        except:
            return [alpha_steps]  # if alpha is single value, return in a list


def find_critically_connecting_patches(R0_pre_connect: np.ndarray, R0_post_connect: np.ndarray) -> np.ndarray:
    """
    If discontinuity is detected, find and return connecting patches of land.
    """

    # points which become non-zero in the alpha-step and belong to the post-connected cluster
    became_non_zero = np.where(np.where(R0_post_connect, 1, 0) - np.where(R0_pre_connect, 1, 0) > 0, 1, 0)
    plt.imshow(R0_pre_connect)
    plt.show()
    plt.imshow(R0_post_connect)
    plt.show()
    plt.imshow(became_non_zero)
    plt.show()
    # todo 1. for each sub-cluster in become_non_zero, find perimeter
    #         (what is the most efficient way to find the perimeter ?)
    #      2. if perimeter lies inside two or more clusters you know you have found `connecting' patches,
    #          although, this does not tell you which patches lead to the desired discontinuity-event
    #      3. find which of the connecting patches have common neighbours  e.g. two patches have the same NN C1 & C2
    #         this tells you which patches belong to the same discontinuity event
    #      4. find which of the `common-neighbour' connecting patches lead to the largest rise in cluster size
    #         from this, we can separate out the desired discontinuity event
    #  Summary : find which points connect the same clusters, find which clusters give rise to the largest discontinuity

    assert 0


def alpha_stepping_method(alpha_steps:list, R0_map:np.ndarray, find_criticals:bool) -> list:
    """
    Perform simple alpha-stepping over the R0-map, return cluster-sizes.
    """
    largest_cluster_size_vs_alpha = np.zeros_like(alpha_steps)
    R0_map = R0_map[130:180, 60:110]  # Test smaller sized domain

    # Find index to begin fragmenting cluster
    for index in range(len(alpha_steps)):
        R0_map_alpha = np.where(R0_map > alpha_steps[index], R0_map, 0)
        cluster_sizes = get_top_cluster_sizes(R0_map=R0_map_alpha, get_top_n=1)[1]  # find largest clusters
        largest_cluster_size_vs_alpha[index] = cluster_sizes[0]
        if not index:  # negate zeroth index
            continue

    R0_map_alpha = None
    if find_criticals:
        # find sharpest rise of cluster sizes
        max_index = discontinuity_detect(largest_cluster_size_vs_alpha)
        # find clusters pre connected cluster state
        R0_map_disconnected = np.where(R0_map > alpha_steps[max_index - 1], R0_map, 0)
        # assume top n sub-clusters in pre-connected
        R0_map_disconnected = get_top_cluster_sizes(R0_map=R0_map_disconnected, get_top_n=10)[0]
        # find clusters post connected cluster state
        R0_map_connected = np.where(R0_map > alpha_steps[max_index], R0_map, 0)
        # assume connected cluster is largest-ranked
        R0_map_connected = get_top_cluster_sizes(R0_map=R0_map_connected, get_top_n=1)[0]
        find_critically_connecting_patches(R0_pre_connect=R0_map_disconnected, R0_post_connect=R0_map_connected)

    return cluster_sizes






def scenario_test():
    """
    For a given epicenter, work out containment scenarios from the fragmented map. Return the payoff ratio.
    """
    print('testing fragmented domain for landscape-level control...')
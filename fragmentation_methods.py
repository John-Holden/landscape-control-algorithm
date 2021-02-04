"""
Methods related to fragmenting a domain of R0 values.
"""
import datetime
from collections import defaultdict
import numpy as np
from typing import Union, Tuple
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
import matplotlib.pyplot as plt
from cluster_find import get_top_cluster_sizes, label_connected, CONNECTIVITY_ELEMENT, CONNECTIVITY_ELEMENTS


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


def find_patch_NN(pre_connected_clusters:np.ndarray, connecting_patches:np.ndarray) -> Tuple[dict, np.ndarray]:
    """
    Find which connecting patches have nearest neighbours of two or more clusters in the pre-connected domain.
    """
    connecting_nearest_neighbours = defaultdict(list)
    connecting_nearest_neighbours_ = defaultdict(list)  # inverse dictionary keys <=> values
    connecting_patch_labelled, num_elements = label_connected(R0_map=connecting_patches)
    for connect_patch_id in range(1, num_elements+1):
        array_mask = np.where(connecting_patch_labelled == connect_patch_id, 1, 0)
        array_mask = binary_dilation(array_mask, structure=CONNECTIVITY_ELEMENT,iterations=1)
        neighbour_ids = pre_connected_clusters[np.where(array_mask)]
        neighbour_ids = set([id for id in neighbour_ids if id])
        if len(neighbour_ids) >= 2:
            # two or more neighbours suggests that element id connects them, either partially or fully
            for cluster_id in neighbour_ids:
                connecting_nearest_neighbours[connect_patch_id].append(cluster_id)
                connecting_nearest_neighbours_[cluster_id].append(connect_patch_id)

    for patch_id, cluster_ids in connecting_nearest_neighbours.items():
        print('patch id and cluster ids', patch_id, cluster_ids)
        common_NN = [patch_id]
        for cluster_id in cluster_ids:
            print('checking cluster, ', cluster_id)
            print(f'cluster {cluster_id} has elements {connecting_nearest_neighbours_[cluster_id]}')
            for patch_id_ in connecting_nearest_neighbours_[cluster_id]:
                print('checking element ', patch_id_)
                print(f'element {patch_id_} has NN {connecting_nearest_neighbours[patch_id_]}')
                if patch_id == patch_id_:
                    print(f'we already know this {patch_id_} in {cluster_id}')
                    continue
                # todo need to find groupings
                if connecting_nearest_neighbours[patch_id] == connecting_nearest_neighbours[patch_id_]:
                    print('we have duplicates')
                else:
                    print('we do NOT have duplicates')
                break



    print(connecting_nearest_neighbours)
    print(connecting_nearest_neighbours_)
    assert 0
    return connecting_nearest_neighbours, connecting_patch_labelled


def find_critically_connecting_patches(R0_pre_connect: np.ndarray, R0_post_connect: np.ndarray,
                                       cluster_ids:np.ndarray) -> np.ndarray:
    """
    If discontinuity is detected, find and return connecting patches of land.
    """
    print(type(cluster_ids), ' cluster ids ttype ')
    time = datetime.datetime.now()
    # patches which become non-zero in the alpha-step and belong to the post-connected cluster
    became_non_zero = np.where(np.where(R0_post_connect, 1, 0) - np.where(R0_pre_connect, 1, 0) > 0, 1, 0)
    # super-set of patches which become non-zero and connect the largest cluster fragment

    plt.title('disconnected')
    plt.imshow(R0_pre_connect)
    plt.show()

    plt.title('became non zero')
    plt.imshow(became_non_zero)
    plt.show()

    plt.title('Post-connected')
    plt.imshow(R0_post_connect)
    plt.show()

    R0_pre_connect_cluster_interface = np.zeros_like(R0_pre_connect)
    for id in cluster_ids:  # Find the union of all cluster interfaces,
        R0_target = np.where(R0_pre_connect == id, 1, 0)
        R0_pre_connect_cluster_interface += binary_dilation(R0_target, structure=CONNECTIVITY_ELEMENT,
                                              iterations=1)

    # The union of cluster-interfaces & the patches which became non-zero must contain the critically connecting patches
    R0_pre_connect_cluster_interface = R0_pre_connect_cluster_interface > 1
    R0_pre_connect_cluster_interface = R0_pre_connect_cluster_interface & became_non_zero
    plt.title('connecting patches')
    plt.imshow(R0_pre_connect_cluster_interface)
    plt.show()

    connecting_patch_NN, labelled_connected_patches = find_patch_NN(pre_connected_clusters=R0_pre_connect,
                                                                    connecting_patches=R0_pre_connect_cluster_interface)


    print(f'time taken {datetime.datetime.now() - time}')

    assert 0
    connecting_NN_ = {}
    connecting_NN = where_cluster_NN(pre_connected_clusters=R0_pre_connect, connecting_patches=became_non_zero)
    # TODO continue to find and group same-neighbour connecting patches...
    for index, (cluster_id, labelled_ids) in enumerate(connecting_NN.items()):
        duplicates = False
        if not index:
            connecting_NN_[cluster_id] = labelled_ids
        else:
            for cluster_id_, labelled_ids_ in connecting_NN_.items():
                labelled_ids_.append(99)
                list_ = [id for id in labelled_ids if id in labelled_ids_]
                if len(list_) > 0:
                    print(f'we have duplicates of {list_} | {cluster_id_} in {labelled_ids_}')
                    duplicates = True
                else:
                    print('we do not have duplicates appending')

            print('x', connecting_NN_)
            if not duplicates:
                connecting_NN_[cluster_id] = labelled_ids

    print('x out', connecting_NN_)





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
    R0_map = R0_map[130:175, 50:110]  # Test smaller sized domain
    # R0_map = R0_map[750:850, 400:500]  # Test smaller sized domain


    # Find index to begin fragmenting cluster
    time = datetime.datetime.now()
    for index in range(len(alpha_steps)):
        R0_map_alpha = np.where(R0_map > alpha_steps[index], R0_map, 0)
        cluster_sizes = get_top_cluster_sizes(R0_map=R0_map_alpha, get_top_n=1)[1]  # find largest clusters
        largest_cluster_size_vs_alpha[index] = cluster_sizes[0]
        if not index:  # negate zeroth index
            continue
    print(f'Time taken to alpha-step {datetime.datetime.now() - time}')
    R0_map_alpha = None
    if find_criticals:
        # find sharpest rise of cluster sizes
        max_index = discontinuity_detect(largest_cluster_size_vs_alpha)
        # find clusters pre connected cluster state
        R0_map_disconnected = np.where(R0_map > alpha_steps[max_index - 1], R0_map, 0)
        # assume top n sub-clusters in pre-connected
        R0_map_disconnected, cluster_sizes, cluster_ids  = get_top_cluster_sizes(R0_map=R0_map_disconnected, get_top_n=10)
        # find clusters post connected cluster state
        R0_map_connected = np.where(R0_map > alpha_steps[max_index], R0_map, 0)
        # assume connected cluster is largest-ranked
        R0_map_connected = get_top_cluster_sizes(R0_map=R0_map_connected, get_top_n=1)[0]
        find_critically_connecting_patches(R0_pre_connect=R0_map_disconnected, R0_post_connect=R0_map_connected,
                                           cluster_ids=cluster_ids)

    return cluster_sizes






def scenario_test():
    """
    For a given epicenter, work out containment scenarios from the fragmented map. Return the payoff ratio.
    """
    print('testing fragmented domain for landscape-level control...')
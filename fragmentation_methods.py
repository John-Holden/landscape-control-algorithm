"""
Methods related to fragmenting a domain of R0 values.
"""
import os
import sys
import datetime
import itertools

from typing import Union, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes

from plotting_methods import plot_R0_clusters, plot_fragmented_domain
from cluster_find import rank_cluster_map, label_connected
from parameters_and_setup import STRUCTURING_ELEMENT, TARGETS_C1_C2, FRAGMENT_RANK



def get_alpha_steps(alpha_steps: Union[iter, float, int, str], R0_min:float, R0_max:float,
                    number_of_steps:int =75) -> Union[list, np.ndarray]:
    """
    Find and return what values of alpha will be iterated through.
    """
    if alpha_steps == "auto":   # default 100 steps
        alpha_steps = np.linspace(R0_max, R0_min, number_of_steps)
        assert alpha_steps[-1] < 1, f'Error, expected min R0-alpha value < 1, found {alpha_steps[-1]}'
        return alpha_steps
    else:
        try:
            iter(alpha_steps)  # if alpha is iterable, return
            return alpha_steps
        except:
            return [alpha_steps]  # if alpha is single value, return in a list


def assert_targets_in_join(cluster_targets:list, cluster_target:np.ndarray,
                           R0_clusters1:np.ndarray, R0_clusters2:np.ndarray) -> bool:
    """
    Assert that the cluster-targets are a subset of the R0-clusters prior to the cluster-join. Assert too
    that the cluster targets are members of the R0-clusters after the join.
    """
    targets_connects = [None, None]
    # Test each target-cluster connects in the next step.
    for count, id in enumerate(cluster_targets):
        cluster_target_indices = np.where(cluster_target == id)
        disconnected_onto_target = np.unique(R0_clusters1[cluster_target_indices])
        connected_onto_target = np.unique(R0_clusters2[cluster_target_indices])

        assert len(connected_onto_target) == 1, f'Error, expect projection of target {id} and R0-connected to match,' \
                                               f' found {connected_onto_target}'
        assert len(disconnected_onto_target) == 1, f'Error, expect projection of target {id} and R0-disconnected to' \
                                                  f' match, found {disconnected_onto_target}'
        assert id == disconnected_onto_target[0], f'Error, expect projection of target {id} and R0-disconnected to ' \
                                                 f'match, found {disconnected_onto_target}'

        # Is the cluster-target present in the newly-connected R0 cluster ?
        targets_connects[count] = True if disconnected_onto_target[0] else False

    return all(target_present for target_present in targets_connects)


def test_removal_disconnects(R0_fragmented:np.ndarray, cluster_targets:np.ndarray) -> bool:
    """
    Test that the critically-connecting patches that have been identified break the largest cluster as expected.
    """

    R0_fragmented, sizes, ids = rank_cluster_map(R0_fragmented)
    # 1. If there is not more than 2 elements in the fragmented cluster, we have not fragmented the targets.
    # 2. If targets C1 and C2 belong to the same cluster in R0_fragmented, we have not fragmented the targets.
    target_1_in_frag = np.unique(R0_fragmented[np.where(cluster_targets==1)])
    target_2_in_frag = np.unique(R0_fragmented[np.where(cluster_targets==2)])

    try:
        assert len(target_1_in_frag) == 1, f'Error, expecting a single value in C1, found {target_1_in_frag}'
        assert len(target_2_in_frag) == 1, f'Error, expecting a single value in C2, found {target_2_in_frag}'
        assert target_1_in_frag != target_2_in_frag, f'Error, C1 and C2 should not be equal, found C1, C2 \in {target_1_in_frag}'
    except Exception as e:
        print(e)
        return False

    if len(ids) >= 2:
        return True
    if len(ids) == 1:  # removal of patches did not fragment the cluster.
        return False
    else:
        sys.exit(f'Error, something went wrong. Found cluster ids = {ids}')


def find_interface_joins(cluster_targets:np.ndarray,
                         cluster_interface:np.ndarray,
                         became_non_zero:np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Find small patches that connect the target-clusters which lay in the (close-range) cluster interface.
    To do this, we take a binary dilation (one 1 iterations) and find if this neighbours the cluster targets.
    """

    # connections may exist in a single unit-cluster or an arbitrary-sized cluster, therefore, we need to label patches
    became_non_zero = label_connected(became_non_zero)[0]

    # patches which lay in the interface and become non-zero have the chance to connect C1 and C2
    potential_connector_element_list =  np.unique(became_non_zero[np.where(cluster_interface)])[1:]

    connecting_patch_num = 0
    connecting_patches = np.zeros_like(cluster_targets)

    bd_cluster_target_1 = binary_dilation(cluster_targets == 1, STRUCTURING_ELEMENT)
    bd_cluster_target_2 = binary_dilation(cluster_targets == 2, STRUCTURING_ELEMENT)

    for connect_patch_id in potential_connector_element_list:
        target_patch = np.where(became_non_zero == connect_patch_id)
        patch_in_target_1 = any(bd_cluster_target_1[target_patch])  # is patch in a cluster-target ?
        patch_in_target_2 = any(bd_cluster_target_2[target_patch])
        if patch_in_target_1 and patch_in_target_2:
            # Two or more neighbours suggest patch_id joins C1 and C2.
            for index in range(len(target_patch[0])):
                # Find which elements of the connecting cluster directly neighbour C1 and C2 - ignore all others.
                row, col = target_patch[0][index], target_patch[1][index]
                row_coords = tuple([row + i for i in [-1,-1,-1,0,0,0,1,1,1]])
                col_coords = tuple([col + i for i in [1,0,-1,1,0,-1,1,0,-1]])
                moore_coords = tuple([row_coords, col_coords])
                if 1 in cluster_targets[moore_coords] or 2 in cluster_targets[moore_coords]:
                    connecting_patches[row, col] = 1
                    connecting_patch_num += 1

    return connecting_patches, connecting_patch_num


def find_critically_connecting_patches(R0_pre_connect: np.ndarray, R0_post_connect: np.ndarray,
                                       cluster_targets:np.ndarray) -> np.ndarray:
    """
    If discontinuity is detected, find and return a binary map of the patches of land that join the largest cluster-join.
    """
    # patches which become non-zero in the alpha-step and belong to the post-connected cluster
    became_non_zero = R0_post_connect - np.where(R0_pre_connect, 1, 0)
    became_non_zero = became_non_zero >= 1

    # fill internal holes (we only need to consider patches on the interface), then rank-order
    cluster_interface_ = binary_fill_holes(cluster_targets, STRUCTURING_ELEMENT)
    cluster_interface_ = rank_cluster_map(cluster_interface_)[0]

    # perimeter == binary dilated array - original array
    cluster_interface = binary_dilation(cluster_interface_, STRUCTURING_ELEMENT)
    cluster_interface = cluster_interface - np.where(cluster_interface_, 1, 0)

    connecting_patches, connection_number = find_interface_joins(cluster_targets, cluster_interface, became_non_zero)

    R0_fragmented = R0_post_connect * np.logical_not(connecting_patches)


    # plt.title('connecting patches')
    # plot_R0_clusters(connecting_patches)

    if connection_number and test_removal_disconnects(R0_fragmented, cluster_targets):
        # The patches found in the interface fragmented the cluster.
        return connecting_patches

    # Plot and save exception.
    #--------------------------------------------------------------#
    plt.title('Error pre-connected map')
    plot_R0_clusters(rank_cluster_map(R0_pre_connect)[0])
    if not os.path.exists('./data_store/exceptions/e_pre_connected_map.npy'):
        np.save('./data_store/exceptions/e_pre_connected_map', R0_pre_connect)

    plt.title('Error post-connected map')
    plot_R0_clusters(rank_cluster_map(R0_post_connect)[0])
    if not os.path.exists('./data_store/exceptions/e_post_connected_map.npy'):
        np.save('./data_store/exceptions/e_post_connected_map', R0_post_connect)

    plt.title('Error, domain did not fragment')
    plot_R0_clusters(rank_cluster_map(R0_fragmented)[0])
    if not os.path.exists('./data_store/exceptions/e_fragments.npy'):
        np.save('./data_store/exceptions/e_fragments', R0_fragmented)

    plt.title('Error, cluster targets')
    plot_R0_clusters(cluster_targets)
    if not os.path.exists('./data_store/exceptions/e_targets.npy'):
        np.save('./data_store/exceptions/e_targets', cluster_targets)

    plt.title(f'Error, connecting patches, number removed {connection_number}')
    plot_R0_clusters(connecting_patches)
    if not os.path.exists('./data_store/exceptions/e_patches_detected.npy'):
        np.save('./data_store/exceptions/e_patches_detected', connecting_patches)

    assert connection_number, f'Error found 0 patches to remove'
    sys.exit()


def find_alpha_discontinuities(alpha_steps, R0_map):
    """
    Find how the maximum-cluster size changes over the alpha-thresholding procedure. Return cluster-size vs alpha.
    """

    joins_at_alpha = {}
    for index in range(len(alpha_steps)-1):
        # Iterate through alpha and find where clusters join to form larger clusters.
        R0_map_alpha = np.where(R0_map > alpha_steps[index], R0_map, 0)
        R0_map_alpha, cluster_sizes, cluster_ids = rank_cluster_map(R0_map=R0_map_alpha, get_ranks=5)

        R0_map_d_alpha = np.where(R0_map > alpha_steps[index+1], R0_map, 0)
        R0_map_d_alpha, d_cluster_sizes = rank_cluster_map(R0_map=R0_map_d_alpha, get_ranks=1)[:2]

        cluster_joins = np.unique(R0_map_alpha[np.where(R0_map_d_alpha)])
        cluster_joins = [rank for rank in cluster_joins if rank not in [0]]

        if len(cluster_joins) <= 1 or 1 not in cluster_joins:
            continue

        sizes = [cluster_sizes[rank - 1] for rank in cluster_joins]
        targets = [comb for comb in itertools.combinations(cluster_joins, 2) if 1 in comb]
        cluster_size_ratios = [cluster_sizes[comb[1] - 1] / cluster_sizes[comb[0] - 1] for comb in targets]

        joins_at_alpha[index] = {'cluster_targets': targets,
                                'sizes': sizes,
                                 f'ratios' : cluster_size_ratios}

    return joins_at_alpha


def targets_join_in_step(R0_d_alpha:np.ndarray, cluster_targets:np.ndarray) -> Union[None, int]:
    """
    Test whether or not clusters-join for the alpha step. If not, return False, otherwise return the target.
    """
    for rank in range(1, 10):  # Consider whether or not a targets join in the top N
        joins = np.unique(cluster_targets[np.where(R0_d_alpha == rank)])
        targets_joined = True if 1 in joins and 2 in joins else False
        if targets_joined:
            if rank < 10:
                return np.where(R0_d_alpha == rank, 1, 0)

            print(f'rank @ join {rank}, joins {joins} ')
            raise NotImplemented

    return None


def update_targets(cluster_targets:np.ndarray, R0_d_alpha:np.ndarray) -> np.ndarray:
    """
    For each alpha-step, update the cluster-targets which monotonically increase.
    """

    cluster_1 = np.unique(R0_d_alpha[np.where(cluster_targets == 1)]) # rank of C1 in R0 map @ alpha + d_alpha
    cluster_2 = np.unique(R0_d_alpha[np.where(cluster_targets == 2)]) # rank of C2 in R0 map @ alpha + d_alpha

    if len(cluster_1) == 1 and len(cluster_2) == 1 and not np.array_equal(cluster_1, cluster_2):
        # The addition of extra patches in the step are added to the targets, and the values 1,2, preserved.
        return np.where(R0_d_alpha == cluster_1[0], 1, 0) + np.where(R0_d_alpha == cluster_2, 2, 0)

    print(f'C1 found in ranks, {cluster_1}, of R0 + d_alpha |')
    print(f'C2 found in ranks, {cluster_2}, of R0 + d_alpha |')
    raise AssertionError


def get_payoff(patches:np.ndarray, R0_map:np.ndarray) -> float:
    """
    Find the payoff, defined as the second largest fragment dived by the number of patches to fragment.
    """
    target_sizes = rank_cluster_map(R0_map)[1]

    return target_sizes[1] / len(patches[0])


def set_cluster_targets(cluster_targets: np.ndarray, cluster_join:list) -> np.ndarray:
    """
    Cluster targets, no matter their rank, are set to values 1 and 2 which are preserved throughout the iteration.
    """
    ids_to_cast = [id for id in cluster_join if id not in TARGETS_C1_C2]
    if len(ids_to_cast) == 0:  # cluster targets have ranks 1 and 2 i.e. in TARGETS_C1_C2
        return cluster_targets

    # cluster targets have value(s) other than what is defined in TARGETS_C1_C2
    targets_to_cast = [value for value in TARGETS_C1_C2 if value not in cluster_join]
    assert len(ids_to_cast) == len(targets_to_cast)
    assert len(ids_to_cast) <= 2
    for value, id in zip(targets_to_cast, ids_to_cast):  # re-evaluate target -> \in [1, 2]
        cluster_targets[np.where(cluster_targets == id)] = value

    return cluster_targets


def find_best(frag_method: Callable) -> Callable:
    """
    Find best fragmentation - iterate over different alpha-valued initial conditions
    """
    def iterator(R0_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha_steps = get_alpha_steps('auto', R0_max=7, R0_min=0.99, number_of_steps=30)
        join_history = find_alpha_discontinuities(alpha_steps, R0_map)
        best_payoff, iteration, optimal_indices = 0, 0, None

        for alpha_index, join_info in join_history.items():  # Iterate over different cluster-joins \in [1, \alpha_max]
            alpha_steps_ = alpha_steps[alpha_index:]

            if iteration < 6:
                iteration += 1
                continue

            for cluster_join, join_ratio in zip(join_info['cluster_targets'], join_info['ratios']):
                # Each join could contain N-clusters, iterate through each join to the largest cluster
                # e.g. C1-C2-C3 -> [C1, C2], [C1, C3]
                cluster_targets = rank_cluster_map(R0_map > alpha_steps_[0], get_ranks=cluster_join)[0]
                cluster_targets = set_cluster_targets(cluster_targets, cluster_join)

                connecting_patches_indices, R0_map_fragmented = frag_method(R0_map, cluster_targets, alpha_steps_)
                payoff = get_payoff(connecting_patches_indices, R0_map_fragmented)
                assert 0

                if payoff > best_payoff:
                    best_payoff = payoff
                    optimal_fragmentation = R0_map_fragmented
                    optimal_indices = connecting_patches_indices

                iteration+=1

        if optimal_indices is None:
            raise NotImplemented

        return optimal_indices, optimal_fragmentation

    return iterator


@find_best
def alpha_stepping_method(R0_map:np.ndarray, cluster_targets:np.ndarray=None,  alpha_steps:list = None) -> np.ndarray:
    """
    Perform the /alpha-stepping method over the R0-map in order to find critically-connecting patches.
    """
    critical_joins = np.zeros_like(R0_map)
    for alpha_index in range(len(alpha_steps) - 1):
        print('alpha = ', alpha_steps[alpha_index])
        # Iterate through alpha index until alpha = 0.99
        R0_alpha = rank_cluster_map(R0_map > alpha_steps[alpha_index])[0]
        R0_d_alpha = rank_cluster_map(R0_map > alpha_steps[alpha_index+1])[0]
        R0_d_alpha_target = targets_join_in_step(R0_d_alpha, cluster_targets)

        if R0_d_alpha_target is None:
            assert alpha_index, 'Error, on the first iteration, expected a discontinuity'
            cluster_targets = update_targets(cluster_targets, R0_d_alpha)
            continue

        if not np.array_equal(np.unique(cluster_targets), [0, 1, 2]):
            raise AssertionError  # Error, expected [0, 1, 2] in cluster-targets

        patches_to_remove = find_critically_connecting_patches(R0_alpha, R0_d_alpha_target, cluster_targets)
        R0_map = R0_map * np.logical_not(patches_to_remove)

        cluster_targets = update_targets(cluster_targets, rank_cluster_map(R0_map > alpha_steps[alpha_index+1])[0])
        critical_joins += patches_to_remove

    critical_joins = np.where(critical_joins)
    critical_joins = (tuple([int(i) for i in critical_joins[0]]), tuple([int(i) for i in critical_joins[1]]))

    return critical_joins, R0_map


def update_fragmentation_target(R0_map:np.ndarray, patch_indices:tuple) -> np.ndarray:
    """
    Chose the largest cluster in the fragmented domain.
    """
    R0_map[patch_indices] = 0
    return rank_cluster_map(R0_map, get_ranks=1)[0]


def fragment_R0_map(R0_map_raw: np.ndarray, fragmentation_iterations:int, plot:bool=False) -> Tuple[dict, np.ndarray]:
    """
    Iteratively fragment the largest cluster in the R0_map via targeted tree felling algorithm
    i.e. the `alpha-stepping' method. Save felled patches to file. Return fragmented domain.
    :rtype: object
    """

    if R0_map_raw.max() < 1: # Trivial map
        return None

    connecting_patches = {}
    R0_map = np.where(R0_map_raw > 1, R0_map_raw, 0)  # consider above threshold positions
    R0_map = R0_map * np.array(rank_cluster_map(R0_map, get_ranks=FRAGMENT_RANK)[0] > 0).astype(int)  # concentrate on the largest cluster
    R0_indices = np.where(R0_map)
    R0_indices = [min(R0_indices[0]), max(R0_indices[0]), min(R0_indices[1]), max(R0_indices[1])]
    R0_map = R0_map[R0_indices[0]:R0_indices[1], R0_indices[2]: R0_indices[3]]  # trim domain and save.
    R0_map_ = np.copy(R0_map)  # copy of processed domain - to save

    if plot:
        plt.title('R0-map in put:')
        plot_R0_clusters(rank_cluster_map(R0_map)[0])

    R0_target = np.copy(R0_map)
    time = datetime.datetime.now()
    for iteration in range(fragmentation_iterations):
        print(f'iteration {iteration}')
        connecting_patch_indices, R0_target_fragmented = alpha_stepping_method(R0_target)
        connecting_patches[iteration] = connecting_patch_indices
        R0_target = update_fragmentation_target(R0_map, connecting_patch_indices)
        R0_target = R0_target * R0_map

        if plot:
            plt.title(f'R0-target @ {iteration}:')
            plot_R0_clusters(R0_target, 1)

    if plot:
        plt.title(f'Fragmented to {iteration+1} iterations')
        plot_fragmented_domain(connecting_patches, R0_map)

    print(f'Time taken to fragment {fragmentation_iterations} iterations: {datetime.datetime.now() - time}')
    return connecting_patches, R0_map_


if __name__ == '__main__':
    # Load in error patches
    print('Running fragmentation for single iteration')
    e_cluster_targets = np.load('./data_store/exceptions/e_targets.npy')
    e_fragmented_R0_map = np.load('./data_store/exceptions/e_fragments.npy')
    e_connective_patches = np.load('./data_store/exceptions/e_patches_detected.npy')
    e_pre_connected_R0_map = np.load('./data_store/exceptions/e_pre_connected_map.npy')
    e_post_connected_R0_map = np.load('./data_store/exceptions/e_post_connected_map.npy')

    find_critically_connecting_patches(e_pre_connected_R0_map,
                                       e_post_connected_R0_map,
                                       e_cluster_targets)






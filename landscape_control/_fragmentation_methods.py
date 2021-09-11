import sys
import datetime
import numpy as np

import warnings
from itertools import combinations
from typing import Union, Tuple, Callable, Any
from scipy.ndimage import binary_dilation, binary_fill_holes

from ._cluster_find import rank_cluster_map
from .exceptions import ClustersDidNotFragmentSave, ClustersDidNotFragmentError
from parameters_and_setup import STRUCTURING_ELEMENT

TIMESTAMP = datetime.datetime.now().strftime("%d-%m-%Y")


def get_alpha_steps(alpha_steps: Union[iter, float, int, str], R0_min:float, R0_max:float,
                    number_of_steps:int = 75) -> Union[list, np.ndarray]:
    """
    Find and return what values of alpha will be iterated through.
    """
    if alpha_steps == "auto":   # default 100 steps
        alpha_steps = np.linspace(R0_max, R0_min, number_of_steps)
        assert alpha_steps[-1] < 1, f'Error, expected min R0-alpha value < 1, found {alpha_steps[-1]}'
        return alpha_steps

    else:
        raise NotImplementedError


def patch_tidy(R0_map: np.ndarray, cluster_targets: np.ndarray, connector_patches: np.ndarray) -> np.ndarray:
    """
    Identify and strip unnecessary critical patches - sometimes, the alpha-stepping method produces some extra
    patches, which if removed will have no influence on cluster connections. That is, a patch must lay in the interface
    between C1 and C2 (or equivalently the binary dilation of both C1 AND C2).
    """

    connector_patches1 = np.logical_and(connector_patches, binary_dilation(cluster_targets == 1, STRUCTURING_ELEMENT))
    connector_patches2 = np.logical_and(connector_patches, binary_dilation(cluster_targets == 2, STRUCTURING_ELEMENT))

    connector_patches = np.logical_and(connector_patches1, connector_patches2)
    connector_patches = np.logical_and(connector_patches, R0_map)

    if test_removal_disconnects(R0_map * np.logical_not(connector_patches), cluster_targets):
        return connector_patches
    else:
        raise ClustersDidNotFragmentError


def find_alpha_discontinuities(alpha_steps, R0_map, record_ranks: int = 3, plot_cluster_sizes_over_step: bool = False):
    """
    Find how the maximum-cluster size changes over the alpha-thresholding procedure. Return cluster-size vs alpha.
    """

    joins_at_alpha = {}
    cluster_sizes_vs_alpha = np.zeros(shape=(len(alpha_steps), record_ranks))
    for index in range(len(alpha_steps)-1):
        # Iterate through alpha and find where clusters join to form larger clusters.
        R0_map_alpha = np.where(R0_map > alpha_steps[index], 1, 0)
        if not np.any(R0_map_alpha):
            continue

        R0_map_alpha, cluster_sizes, cluster_ids = rank_cluster_map(R0_map=R0_map_alpha, get_ranks=5)
        R0_map_d_alpha = np.where(R0_map > alpha_steps[index+1], 1, 0)
        R0_map_d_alpha, d_cluster_sizes = rank_cluster_map(R0_map=R0_map_d_alpha, get_ranks=1)[:2]
        if any(cluster_sizes) and len(cluster_sizes) < record_ranks:
            cluster_sizes_vs_alpha[index][:len(cluster_sizes)] = cluster_sizes
        elif any(cluster_sizes):
            cluster_sizes_vs_alpha[index][:record_ranks] = cluster_sizes[:record_ranks]

        # clusters that join to form the largest cluster in the step alpha -> alpha + d_alpha
        cluster_joins = np.unique(R0_map_alpha[np.where(R0_map_d_alpha)])
        cluster_joins = [rank for rank in cluster_joins if not rank == 0]

        if len(cluster_joins) <= 1:  # no cluster joins
            joins_at_alpha[index] = False
            continue

        sizes = [cluster_sizes[rank - 1] for rank in cluster_joins]
        targets = [comb for comb in combinations(cluster_joins, 2) if 1 in comb]
        if not targets:  # Not implemented: target joins do not consist of the rank 1 cluster
            continue

        cluster_size_ratios = [cluster_sizes[comb[1] - 1] / cluster_sizes[comb[0] - 1] for comb in targets]

        joins_at_alpha[index] = {'cluster_targets': targets, 'sizes': sizes, f'ratios': cluster_size_ratios}

        if plot_cluster_sizes_over_step:
            from landscape_control.plotting_methods import plot_R0_clusters
            plot_R0_clusters(R0_map_alpha, rank=5)
            plot_R0_clusters(R0_map_d_alpha, rank=1)
            print(joins_at_alpha[index])

    cluster_sizes_vs_alpha[index+1][0] = d_cluster_sizes[0]  # append original size of cluster
    largest_cluster_size = cluster_sizes_vs_alpha[:, 0]
    sorted_indices = np.argsort(np.gradient(np.gradient(largest_cluster_size)))[::-1]
    if plot_cluster_sizes_over_step:
        from landscape_control.plotting_methods import plot_cluster_size_with_xi
        plot_cluster_size_with_xi(alpha_steps, record_ranks, cluster_sizes_vs_alpha)

    for c, index in enumerate(sorted_indices):
        try:
            if joins_at_alpha[index]:
                # i.e. return the largest-rise in cluster size due to a cluster-cluster join.
                return index, joins_at_alpha[index]['cluster_targets']\

        except Exception as e:
            msg = f" Warning! Something went wrong, {e}"
            warnings.warn(msg)

    sys.exit('Error something went wrong @ find_alpha_discontinuities')


def test_removal_disconnects(R0_fragmented: np.ndarray, cluster_targets: np.ndarray) -> bool:
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
        assert target_1_in_frag != target_2_in_frag, f'Error, C1 and C2 should not be equal, found C1, C2 \in ' \
                                                     f'{target_1_in_frag}'
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
                         potential_connectors:np.ndarray) -> Tuple[np.ndarray, int]:
    """ Find which clusters join the target-clusters. To do this, we take a binary dilation of each cluster-element that
    becomes non-zero in the step alpha -> d_alpha, and find if it neighbours both cluster targets."""

    # patches which lay in the interface and become non-zero have the chance to connect C1 and C2
    list_ = np.where(np.logical_and(potential_connectors, cluster_interface))
    potential_connector_ids = np.unique(potential_connectors[list_])

    connecting_patch_num = 0
    connecting_patches = np.zeros_like(cluster_targets)

    bd_cluster_target_1 = binary_dilation(cluster_targets == 1, STRUCTURING_ELEMENT)
    bd_cluster_target_2 = binary_dilation(cluster_targets == 2, STRUCTURING_ELEMENT)

    C1_join = []
    C2_join = []
    C1_C2_join = []

    for cluster_id in potential_connector_ids:
        target_patch = np.where(potential_connectors == cluster_id)
        patch_in_target_1 = any(bd_cluster_target_1[target_patch])  # is patch in a cluster-target ?
        patch_in_target_2 = any(bd_cluster_target_2[target_patch])
        if patch_in_target_1 and patch_in_target_2:
            # Two or more neighbours suggest patch_id joins C1 and C2.
            for index in range(len(target_patch[0])):
                # Find which elements of the connecting cluster directly neighbour C1 and C2 - ignore all others.
                row, col = target_patch[0][index], target_patch[1][index]
                row_coords = tuple([row + i for i in [-1, -1, -1, 0, 0, 0, 1, 1, 1]])
                col_coords = tuple([col + i for i in [1, 0, -1, 1, 0, -1, 1, 0, -1]])
                moore_coords = tuple([row_coords, col_coords])

                if 1 in cluster_targets[moore_coords] and 2 in cluster_targets[moore_coords]:
                    C1_C2_join.append((row, col))
                    connecting_patches[row, col] = 1
                    connecting_patch_num += 1
                    continue
                elif 1 in cluster_targets[moore_coords]:
                    C1_join.append((row, col))
                elif 2 in cluster_targets[moore_coords]:
                    C2_join.append((row, col))

    if len(C1_join) < len(C2_join):
        connecting_patch_num += len(C1_join)
        for patch in C1_join:
            connecting_patches[patch] = 1

    elif len(C2_join) < len(C1_join) or len(C2_join) == len(C1_join):
        connecting_patch_num += len(C2_join)
        for patch in C2_join:
            connecting_patches[patch] = 1

    return connecting_patches, connecting_patch_num


def get_payoff(patches: np.ndarray, R0_map: np.ndarray) -> float:
    """Find the payoff, defined as the second largest fragment dived by the number of patches to fragment."""
    target_sizes = rank_cluster_map(R0_map)[1]
    return target_sizes[1] / len(patches)


def targets_join_in_step(R0_d_alpha:np.ndarray, cluster_targets:np.ndarray) -> Union[bool, Any]:
    """ Test whether or not clusters-join for the alpha step. If not, return False, otherwise return the target."""
    targets_joined_rank = np.unique(R0_d_alpha[np.where(cluster_targets)])
    if len(targets_joined_rank) == 1 and targets_joined_rank[0]:
        return True, targets_joined_rank

    assert targets_joined_rank[0], 'Error, targets clusters should not be zero-valued.'
    assert len(targets_joined_rank) == 2, f'Error, expected two distinct clusters, found {targets_joined_rank}'
    assert targets_joined_rank[0] != targets_joined_rank[1], f'Error, clusters should not be equal, found {targets_joined_rank}'
    return False, targets_joined_rank  # clusters did not connect


def update_targets_after_fragmentation(cluster_targets:np.ndarray, connector_patches:np.ndarray,
                                       R0_d_alpha:np.ndarray) -> np.ndarray:
    """update teh cluster targets after critical-joins have been found and removed"""

    R0_fragmented = rank_cluster_map(R0_d_alpha * np.logical_not(connector_patches))[0]
    new_targets = np.unique(R0_fragmented[np.where(cluster_targets)])
    assert len(new_targets) == 2, f'Error, expected two elements, found {new_targets}'
    assert new_targets[0] != new_targets[1], f'Error, new targets should have different values, found {new_targets}'
    return np.where(R0_fragmented == new_targets[0], 1, 0) + np.where(R0_fragmented == new_targets[1], 2, 0)


def update_fragmentation_target(R0_map:np.ndarray, patch_indices:tuple) -> np.ndarray:
    """ Chose the largest cluster in the fragmented domain."""
    R0_map[patch_indices] = 0
    return rank_cluster_map(R0_map, get_ranks=1)[0]


def find_critically_connecting_patches(R0_pre_connect: np.ndarray, R0_post_connect: np.ndarray,
                                       cluster_targets:np.ndarray) -> np.ndarray:
    """
    If discontinuity is detected, find and return a binary map of the patches of land that join the largest cluster-join.
    """
    # patches which become non-zero in the alpha-step and belong to the post-connected cluster
    potential_connectors = rank_cluster_map(R0_post_connect * np.logical_not(cluster_targets))[0]
    # fill internal holes - we only need to consider patches on the interface
    cluster_interface_ = binary_fill_holes(cluster_targets, STRUCTURING_ELEMENT)
    cluster_interface_ = rank_cluster_map(cluster_interface_)[0]
    # computing binary dilated array - original array, gives the perimeter
    cluster_interface = binary_dilation(cluster_interface_, STRUCTURING_ELEMENT)
    cluster_interface = cluster_interface - np.where(cluster_interface_, 1, 0)
    # Find interface connections
    connecting_patches, connection_number = find_interface_joins(cluster_targets, cluster_interface, potential_connectors)

    R0_fragmented = R0_post_connect * np.logical_not(connecting_patches)

    if connection_number and test_removal_disconnects(R0_fragmented, cluster_targets):
        # The patches found in the interface fragmented the cluster.
        return connecting_patches

    # Check edge-case whereby C2 is located inside C1 or vice-versa
    C1_filled = binary_fill_holes(cluster_targets == 1, STRUCTURING_ELEMENT)
    C2_filled = binary_fill_holes(cluster_targets == 2, STRUCTURING_ELEMENT)
    if any(C1_filled[np.where(cluster_targets == 2)]):
        C2_interface = binary_dilation(cluster_targets == 2, STRUCTURING_ELEMENT)
        connecting_patches = np.where(np.logical_and(C2_interface, R0_post_connect), 1, 0) - np.where(cluster_targets
                                                                                                      == 2, 1, 0)
    elif any(C2_filled[np.where(cluster_targets == 1)]):
        C1_interface = binary_dilation(cluster_targets == 1, STRUCTURING_ELEMENT)
        connecting_patches = np.where(np.logical_and(C1_interface, R0_post_connect), 1, 0) - np.where(cluster_targets
                                                                                                      == 1, 1, 0)

    R0_fragmented = R0_post_connect * np.logical_not(connecting_patches)
    if test_removal_disconnects(R0_fragmented, cluster_targets):
        return connecting_patches

    # Plot errors, save exception and exit.
    else:
        raise ClustersDidNotFragmentSave(R0_pre_connect, R0_post_connect, R0_fragmented, cluster_targets,
                                         connecting_patches, connection_number)


def find_best(frag_method: Callable) -> Callable:
    """
    Find best fragmentation choice: for a given discontinuity (i.e. cluster join) there may be a number of associated
    clusters, C1-C2-C2. This method finds which cluster-join gives the highest payoff max(C1-C2, C1-C3) where payoff
    is defined as |Ci| / N_cull, where i != 1 and N_cull is the number of removed patches.
    """

    def iterator(R0_map: np.ndarray, debug: bool) -> tuple:
        iteration = 0
        best_payoff = 0
        optimal_indices = None
        alpha_steps = get_alpha_steps('auto', R0_max=7, R0_min=0.99, number_of_steps=30)
        alpha_index, targets = find_alpha_discontinuities(alpha_steps, R0_map, plot_cluster_sizes_over_step=debug)
        if debug:
            print(f'\t # alpha steps = {len(alpha_steps)} | index = {alpha_index} | value = {alpha_steps[alpha_index]}')
            print(f'\t {targets}')
            assert 0

        for target in targets:  # todo handle combinations of joins to find best
            if debug:
                print(f'\t fragmenting targets : {target}')

            connecting_patches_indices = frag_method(R0_map, target, alpha_steps[alpha_index:], debug)

            if connecting_patches_indices is None:  # Error occurred
                np.save(f'./data_store/exceptions/e_R0_map_{TIMESTAMP}', R0_map)
                sys.exit('Error, clusters did not fragment, something went wrong.')

            R0_map_fragmented = np.copy(R0_map)
            R0_map_fragmented[connecting_patches_indices] = 0
            payoff = get_payoff(connecting_patches_indices, R0_map_fragmented)

            if payoff > best_payoff:  # record best-performing fragment
                best_payoff = payoff
                optimal_indices = connecting_patches_indices
            iteration += 1

        if optimal_indices is None:
            sys.exit('Error, I did not find optimal fragmentation - something went wrong')

        return optimal_indices

    return iterator


@find_best
def alpha_stepping_method(R0_map: np.ndarray, targets: tuple = None,
                          alpha_steps: list = None, debug: bool = False) -> tuple:
    """
    Perform the /alpha-stepping method over the R0-map and return a tuple of  critically-connecting patch indices.
    """
    cluster_targets = None
    R0_map_ = np.copy(R0_map)
    critical_joins = np.zeros_like(R0_map)
    for alpha_index in range(len(alpha_steps) - 1):
        # Iterate through alpha index until alpha = 0.99
        R0_alpha = rank_cluster_map(R0_map > alpha_steps[alpha_index])[0]
        R0_d_alpha = rank_cluster_map(R0_map > alpha_steps[alpha_index+1])[0]
        if not alpha_index:  # set targets on first iteration
            cluster_targets = np.where(R0_alpha == targets[0], 1, 0) + np.where(R0_alpha == targets[1], 2, 0)

        targets_joined, R0_d_alpha_target = targets_join_in_step(R0_d_alpha, cluster_targets)
        if not targets_joined:
            cluster_targets = np.where(R0_d_alpha == R0_d_alpha_target[0], 1, 0) + \
                              np.where(R0_d_alpha == R0_d_alpha_target[1], 2, 0)
            continue

        R0_connected = np.where(R0_d_alpha == R0_d_alpha_target[0], 1, 0)
        patches_to_remove = find_critically_connecting_patches(R0_alpha, R0_connected, cluster_targets)
        if patches_to_remove is None:  # error, has occurred
            return None, None

        R0_map_ = R0_map_ * np.logical_not(patches_to_remove)
        cluster_targets = update_targets_after_fragmentation(cluster_targets, patches_to_remove, R0_d_alpha)
        critical_joins += patches_to_remove

    critical_joins = patch_tidy(R0_map, cluster_targets, critical_joins)
    critical_joins = np.where(critical_joins)

    return tuple([int(i) for i in critical_joins[0]]), tuple([int(i) for i in critical_joins[1]])

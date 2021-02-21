"""
Fragment the map of R0 values to N iterations.
"""
import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt
from typing import  Tuple, Callable
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes

from fragmentation_helper_methods import find_interface_joins, test_removal_disconnects, plot_save_errors, \
    update_fragmentation_target, get_alpha_steps, find_alpha_discontinuities, TIMESTAMP, get_payoff, \
    targets_join_in_step, update_targets_after_fragmentation

from cluster_find import rank_cluster_map
from plotting_methods import plot_R0_clusters, plot_fragmented_domain
from parameters_and_setup import STRUCTURING_ELEMENT, FRAGMENT_RANK


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
        connecting_patches = np.where(np.logical_and(C2_interface, R0_post_connect), 1, 0) - \
                             np.where(cluster_targets == 2, 1, 0)
    elif any(C2_filled[np.where(cluster_targets == 1)]):
        C1_interface = binary_dilation(cluster_targets == 1, STRUCTURING_ELEMENT)
        connecting_patches = np.where(np.logical_and(C1_interface, R0_post_connect), 1, 0) - \
                             np.where(cluster_targets == 1, 1, 0)

    R0_fragmented = R0_post_connect * np.logical_not(connecting_patches)
    if test_removal_disconnects(R0_fragmented, cluster_targets):
        return connecting_patches

    # Plot errors, save exception and exit.
    else:
        plot_save_errors(connection_number, R0_pre_connect, R0_post_connect,
                     R0_fragmented, cluster_targets, connecting_patches)
        return None


def find_best(frag_method: Callable) -> Callable:
    """
    Find best fragmentation - iterate over different alpha-valued initial conditions
    """
    def iterator(R0_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        iteration = 0
        best_payoff = 0

        alpha_steps = get_alpha_steps('auto', R0_max=7, R0_min=0.99, number_of_steps=30)
        alpha_index, targets = find_alpha_discontinuities(alpha_steps, R0_map)
        for target in targets:  # todo handle combinations of joins to find best
            if len(targets)>1:
                print(f'\t fragmenting targets : {target} | {iteration}/{len(targets)}')
            connecting_patches_indices, R0_map_fragmented = frag_method(R0_map, target, alpha_steps[alpha_index:])
            plt.title(f'target {target}')
            plot_R0_clusters(rank_cluster_map(R0_map_fragmented)[0])

            if connecting_patches_indices is None: # Error occurred
                np.save(f'./data_store/exceptions/e_R0_map_{TIMESTAMP}', R0_map)
                sys.exit()

            payoff = get_payoff(connecting_patches_indices, R0_map_fragmented)
            if payoff > best_payoff:  # record best-performing fragment
                best_payoff = payoff
                optimal_fragmentation = R0_map_fragmented
                optimal_indices = connecting_patches_indices

            iteration+=1

        if optimal_indices is None:
            sys.exit('Error, I did not find optimal fragmentation - something went wrong')

        return optimal_indices, optimal_fragmentation

    return iterator


@find_best
def alpha_stepping_method(R0_map:np.ndarray, targets:tuple=None,  alpha_steps:list = None) -> np.ndarray:
    """
    Perform the /alpha-stepping method over the R0-map in order to find critically-connecting patches.
    """
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

        R0_map = R0_map * np.logical_not(patches_to_remove)

        cluster_targets = update_targets_after_fragmentation(cluster_targets, patches_to_remove, R0_d_alpha)

        critical_joins += patches_to_remove

    critical_joins = np.where(critical_joins)
    critical_joins = (tuple([int(i) for i in critical_joins[0]]), tuple([int(i) for i in critical_joins[1]]))

    return critical_joins, R0_map


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
    R0_indices = [min(R0_indices[0])-2, max(R0_indices[0])+2, min(R0_indices[1])-2, max(R0_indices[1])+2]
    R0_map = R0_map[R0_indices[0]:R0_indices[1], R0_indices[2]: R0_indices[3]]  # trim domain and save.

    R0_map_ = np.copy(R0_map)  # copy of processed domain - to save
    fragmented_domain = np.zeros_like(R0_map)

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
        fragmented_domain[connecting_patch_indices] = iteration+1

    if plot :
        plt.title(f'Fragmented to {iteration+1} iterations')
        plot_fragmented_domain(fragmented_domain, R0_map)

    print(f'Time taken to fragment {fragmentation_iterations} iterations: {datetime.datetime.now() - time}')
    return connecting_patches, R0_map_


def run_single_alpha_step():
    """run find critical patches for one alpha step"""
    load_from = './test_data/C2_is_inside_C1' # ./data_store/exceptions
    cluster_targets = np.load(f'{load_from}/test_cluster_targets.npy')
    pre_connected_R0_map = np.load(f'{load_from}/test_pre_connected_map.npy')
    post_connected_R0_map = np.load(f'{load_from}/test_post_connected_map.npy')

    connector_patches = find_critically_connecting_patches(pre_connected_R0_map,
                                                           post_connected_R0_map,
                                                           cluster_targets)
    plt.title('fragmented domain')
    plot_R0_clusters(rank_cluster_map(post_connected_R0_map * np.logical_not(connector_patches))[0])
    return


def run_single_iteraion():
    """Find cluster-joins and find best payoff for a domain."""
    e_R0_map = np.load(f'test_data/optimisze_break/test_R0_domain.npy')
    critical_joins, R0_map  = alpha_stepping_method(e_R0_map)
    plot_R0_clusters(rank_cluster_map(R0_map)[0])
    return

if __name__ == '__main__':
    # Load in error patches
    run_single_iteraion()


"""
Methods related to fragmenting a domain of R0 values.
"""
import sys
import datetime
from collections import defaultdict
import numpy as np
from typing import Union, Tuple
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
from plotting_methods import plot_cluster_size_vs_alpha, plot_R0_clusters
from cluster_find import get_top_cluster_sizes, label_connected, cluster_freq_count
from run_main import STRUCTURING_ELEMENT


def find_cluster_size_vs_alpha(alpha_steps, R0_map):
    """
    Iterate the maximum-cluster size-change vs alpha and return.
    """
    time = datetime.datetime.now()
    largest_cluster_size_vs_alpha = np.zeros_like(alpha_steps)
    for index in range(len(alpha_steps)):
        R0_map_alpha = np.where(R0_map > alpha_steps[index], R0_map, 0)
        cluster_sizes = get_top_cluster_sizes(R0_map=R0_map_alpha, get_top_n=1)[1]  # find largest clusters
        largest_cluster_size_vs_alpha[index] = cluster_sizes[0]

    print(f'Time taken to work out cluster sizes vs alpha: {datetime.datetime.now() - time}')
    return largest_cluster_size_vs_alpha


def discontinuity_detect(cluster_sizes:np.ndarray, R0_map:np.ndarray,
                         alpha_steps:Union[list, np.ndarray], cluster_target: Union[None, np.ndarray],
                         test_top:int = 5) -> Tuple[Union[int, None], Union[np.ndarray, None]]:
    """
    Find largest increase due to a cluster-cluster join (i.e. discontinuity). Return the first
    alpha-index which connects 2 or more clusters ranked in the top N -- 5 by default.
    """
    assert len(cluster_sizes) > 1, f'Error, length of cluster-sizes {len(cluster_sizes)}, expected > 1.'
    # Todo fix alpha-index iteration - if alpha index < previous continue that way only the next iteration is targeted.

    arg_sorted = np.argsort(np.gradient(cluster_sizes))[::-1]
    for alpha_index in arg_sorted:
        R0_clusters1, sizes, ids = get_top_cluster_sizes(R0_map=R0_map > alpha_steps[alpha_index-1], get_top_n=test_top)
        R0_clusters2 = get_top_cluster_sizes(R0_map=R0_map > alpha_steps[alpha_index], get_top_n=1)[0]
        number_of_joins = np.unique(R0_clusters1[np.where(R0_clusters2)])[1:]  # which clusters join up ?
        # For the 1st iteration, define the cluster-target as the first largest join
        if len(number_of_joins) >= 2 and cluster_target is None:  # Expected 1 or more cluster-joins of the top-ranked clusters.
            cluster_target = np.zeros_like(R0_map)
            cluster_target[np.where(R0_clusters1 == ids[0])] = ids[0]
            cluster_target[np.where(R0_clusters1 == ids[1])] = ids[1]
            return alpha_index, cluster_target

        if len(number_of_joins) >= 2:
            # Are the cluster-targets in the join ?

            target_joins_present = all((lambda id: True if id in number_of_joins else False)(id)
                                       for id in np.unique(cluster_target)[1:])

            print('Are target-joins in this alpha-value ?', target_joins_present)
            plt.title(f'R0 disconnected alpha {alpha_steps[alpha_index-1]}')
            plot_R0_clusters(R0_map=R0_clusters1)

            plt.title(f'R0 connected alpha {alpha_steps[alpha_index]}')
            plot_R0_clusters(R0_map=R0_clusters2)

            plt.title('cluster targets for iteration')
            plot_R0_clusters(R0_map=cluster_target, rank=2)
            print('check if joins in 1-2', target_joins_present)

            if not target_joins_present:  # target-clusters are not present in the cluster-join, continue to next alpha.
                continue

            targets_connects = [None, None]  # Test each target-cluster connects in the next step.
            for count, id in enumerate(np.unique(cluster_target)[1:]):
                cluster_target_indices = np.where(cluster_target == id)
                disconnected_and_target = np.unique(R0_clusters1[cluster_target_indices])
                connected_and_target = np.unique(R0_clusters2[cluster_target_indices])

                print('projection onto disconnected ', disconnected_and_target)
                print('projection onto connected ', connected_and_target)
                # assert len(disconnected_and_target) == 1, f'Error, expect projection of target {id} and R0-disconnected to match'
                # assert id == disconnected_and_target[0],  f'Error, expect projection of target {id} and R0-disconnected to match'

                assert len(connected_and_target) == 1 , f'Error, expect projection of target {id} and R0-connected to match'

                # Is the cluster-target present in the newly-connected R0 cluster
                targets_connects[count] = True if connected_and_target[0] else False

            print('targets in connected', targets_connects)
            if all(target_present for target_present in targets_connects):
                return alpha_index, cluster_target



    return None, None  # we have not detected any more cluster-discontunities


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


def group_cluster_joins(pre_connected_clusters:np.ndarray, connecting_patches:np.ndarray) -> Tuple[dict, np.ndarray]:
    """
    Group cluster-joins and pair with list of connecting patches.
    """
    cluster_joins = defaultdict(list)
    connecting_patch_labelled, num_elements = label_connected(R0_map=connecting_patches)
    is_connection_present = False
    for connect_patch_id in range(1, num_elements+1):
        array_mask = np.where(connecting_patch_labelled == connect_patch_id, 1, 0)
        array_mask = binary_dilation(array_mask, structure=STRUCTURING_ELEMENT,iterations=1)
        neighbour_ids = pre_connected_clusters[np.where(array_mask)]
        neighbour_ids = set([id for id in neighbour_ids if id])
        if len(neighbour_ids) >= 2:
            # two or more neighbours suggests that element id connects them, either partially or fully
            join_ids = (str(id) for id in neighbour_ids)
            join_ids = '_'.join(join_ids)
            # todo think about if you want to store the patch id ?
            #      it may be better to store coordinates of elements
            cluster_joins[join_ids].append(connect_patch_id)
            is_connection_present = True

    assert is_connection_present, 'Error, no cluster-joins were found. Expecting at least one connection of len 2.'
    return cluster_joins, connecting_patch_labelled



def find_critically_connecting_patches(R0_pre_connect: np.ndarray, R0_post_connect: np.ndarray,
                                       cluster_targets:np.ndarray, cluster_ids:np.ndarray,
                                       cluster_sizes:np.ndarray) -> np.ndarray:
    """
    If discontinuity is detected, find and return the patches of land that join the largest cluster-join.
    """
    time = datetime.datetime.now()
    # patches which become non-zero in the alpha-step and belong to the post-connected cluster
    became_non_zero = R0_post_connect - np.where(R0_pre_connect, 1, 0)
    became_non_zero = became_non_zero >= 1

    R0_pre_connect_target_interface = np.zeros_like(R0_pre_connect)
    for id in cluster_ids:  # Find an approximate interface for the cluster-targets.
        R0_pre_connect_target_interface += binary_dilation(np.where(cluster_targets == id, 1, 0),
                                                            structure=STRUCTURING_ELEMENT, iterations=3)

    # The union of cluster-target interface & the patches which became non-zero must contain the critically connecting patches
    R0_pre_connect_target_interface = R0_pre_connect_target_interface > 1
    R0_pre_connect_target_interface = R0_pre_connect_target_interface & became_non_zero

    # Find dictionary-representation of cluster-joins and connecting patches
    cluster_joins, connecting_patches_ = group_cluster_joins(pre_connected_clusters=R0_pre_connect,
                                                             connecting_patches=R0_pre_connect_target_interface)

    print('cluster joins', cluster_joins)
    max_join = 0
    remove_elements = None
    for cluster_join in cluster_joins.keys():
        join_ids = cluster_join.split('_')
        join_size = 0
        for id in join_ids:
            join_size += cluster_sizes[np.where(cluster_ids == int(id))]
        if join_size > max_join:  # remove the elements which give the largest join # todo <- Re-think this, it might be redundant
            max_join = join_size
            remove_elements = cluster_joins[cluster_join]

    print('--> elements to remove ', remove_elements)
    assert remove_elements is not None, f'Error, expected non-zero remove-elements. Found {remove_elements}'
    critically_connecting = np.in1d(connecting_patches_.flatten(), remove_elements)

    print(f'Time taken to process fragmentation-splitting {datetime.datetime.now() - time}\n')
    return critically_connecting.reshape(connecting_patches_.shape), max_join


def test_removal_disconnects(to_remove:np.ndarray, join_size:int, R0_map_connected):
    """
    Test that the critically-connecting patches that have been identified break the largest cluster as expected.
    """
    R0_map_connected[np.where(to_remove)] = 0
    R0_fragmented = label_connected(R0_map=R0_map_connected)[0]
    cluster_sizes = cluster_freq_count(labeled=R0_fragmented)[0]
    assert len(cluster_sizes) > 1, f'Error, expected cluster-break. Found  only {len(cluster_sizes)} cluster-elements.'
    assert join_size <= cluster_sizes.sum(), f'Error, expected join size {join_size} <= {cluster_sizes.sum()}'
    assert len(np.where(R0_map_connected)[0]) >= 1.1 * max(cluster_sizes), f'Error, expected major fragmentation.'
    return True


def test_final_step(R0_map_disconnected: np.ndarray, cluster_ids:np.ndarray, patches_to_remove:np.ndarray) -> bool:
    """
    Test whether the final step, from R0 --> R0 < 1, has more patches to remove or not. This relies
    on the assumption that the cluster-target is the largest cluster in the set of clusters.
    """

    R0_target = R0_map_disconnected == cluster_ids[0]
    R0_pre_connect_cluster_interface = binary_dilation(R0_target, structure=STRUCTURING_ELEMENT,
                                                       iterations=1)
    relevant_boundaries = np.zeros_like(R0_pre_connect_cluster_interface)
    for id in cluster_ids[1:]:  # Find the union of all cluster interfaces,
        R0_target = np.where(R0_map_disconnected == id, 1, 0)
        R0_target = binary_dilation(R0_target, structure=STRUCTURING_ELEMENT, iterations=1)

        if len(np.where(R0_pre_connect_cluster_interface & R0_target)[0]):  # We have found a relevant boundary
            relevant_boundaries += R0_pre_connect_cluster_interface & R0_target


    if len(np.where(relevant_boundaries & patches_to_remove)[0]):
        return True  # the critically connecting patches do connect the target-cluster

    return False  # the critically connecting patches do not connect the target-cluster



def update_cluster_targets(R0_map: np.ndarray, R0_connected:np.ndarray,
                           patches:np.ndarray, alpha:float) -> np.ndarray:
    """
    Update cluster targets, both the shape and value.
    """
    R0_fragmented_ranked = R0_map * np.logical_not(patches) > alpha
    R0_fragmented_ranked = get_top_cluster_sizes(R0_fragmented_ranked, get_top_n=10)[0]
    cluster_targets = get_top_cluster_sizes(R0_connected * np.logical_not(patches), get_top_n=2)[0]
    cluster_targets_ = np.zeros_like(cluster_targets)
    cluster_targets_[:] = cluster_targets  # update the copy
    for target in np.unique(cluster_targets)[1:]:
        target_indices = np.where(cluster_targets == target)
        new_rank_value = np.unique(R0_fragmented_ranked[target_indices])
        assert len(new_rank_value) == 1, f'Error, something has gone wrong. Expected length of 1 found {new_rank_value}'
        if new_rank_value[0] == target:  # rank of target is preserved for this iteration
            continue

        print(f'previous rank of {target} now cast to {new_rank_value[0]}')
        cluster_targets_[np.where(target_indices)] = new_rank_value[0]  # cast new rank to cluster target

    return cluster_targets_


def fragmentation_iteration(alpha_steps:list, R0_map:np.ndarray, cluster_targets:np.ndarray,
                            alpha_index:int) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Access the how cluster-sizes changes vs alpha, find largest cluster-join and critically-connecting patches. Return
    the culled R0-map
    """
    # find clusters pre connected cluster state
    R0_map_disconnected = np.where(R0_map > alpha_steps[alpha_index-1], R0_map, 0)
    # assume top n sub-clusters in pre-connected
    R0_map_disconnected, cluster_sizes, cluster_ids  = get_top_cluster_sizes(R0_map=R0_map_disconnected, get_top_n=10)

    # find clusters post connected cluster state
    R0_map_connected = np.where(R0_map > alpha_steps[alpha_index], R0_map, 0)
    # assume connected cluster is largest-ranked
    R0_map_connected = get_top_cluster_sizes(R0_map=R0_map_connected, get_top_n=1)[0]

    to_remove, join_size = find_critically_connecting_patches(R0_pre_connect=R0_map_disconnected,
                                                              R0_post_connect=R0_map_connected,
                                                              cluster_targets=cluster_targets,
                                                              cluster_ids=cluster_ids, cluster_sizes=cluster_sizes)

    if alpha_steps[alpha_index] < 1 and not test_final_step(R0_map_disconnected, cluster_ids, to_remove):  # todo could remove this
        # For the last step, did the critically-connecting patches produce a cluster-join on the relevant cluster ?
        return to_remove, None


    assert test_removal_disconnects(to_remove, join_size, R0_map_connected), f'An error occurred. The ' \
                                                                             f'critically-connecting ' \
                                                                             f'patches did not fragment the ' \
                                                                             f'largest-cluster as expected.'

    # find cluster-targets for next iteration
    cluster_targets = update_cluster_targets(R0_map=R0_map, R0_connected=R0_map_connected, patches=to_remove,
                                             alpha=alpha_steps[alpha_index])

    plt.title('R0 fragmented - new cluster targets')
    plot_R0_clusters(R0_map=cluster_targets)
    return to_remove, cluster_targets

def alpha_stepping_method(alpha_steps:list, R0_map:np.ndarray):
    """
    Perform the /alpha-stepping method over the R0-map in order to find critically-connecting patches.
    """
    # R0_map = R0_map[130:175, 50:110]  # Test smaller sized domain
    # R0_map = R0_map[750:850, 400:500]  # Test smaller sized domain
    fragmentation_process = True
    iteration = 0
    critical_joins = np.zeros_like(R0_map)
    cluster_targets = None  # target a given cluster for each iteration
    while fragmentation_process:
        print(f'iteration {iteration} |')
        largest_cluster_size_vs_alpha = find_cluster_size_vs_alpha(alpha_steps, R0_map)  # get cluster size-vs-alpha
        plot_cluster_size_vs_alpha(iteration, alpha_steps, largest_cluster_size_vs_alpha)
        # Find index to begin fragmenting cluster.
        discontinuity_index, cluster_targets = discontinuity_detect(largest_cluster_size_vs_alpha, R0_map, alpha_steps,
                                                                    cluster_targets)

        if discontinuity_index is None and cluster_targets is None:
            sys.exit('break')

        # plot_cluster_size_vs_alpha(iteration, alpha_steps, largest_cluster_size_vs_alpha,
        #                            discontinuity_index=discontinuity_index)

        if not iteration:
            alpha = alpha_steps[discontinuity_index]

        if iteration and alpha_steps[discontinuity_index] > alpha:
            sys.exit('alpha increased')
            # Alpha should monotonically decrease over one iteration.
            try:
                # If no cluster-joins detected, an assertion error will be raised. Otherwise, the R0-map will be updated
                R0_map = fragmentation_iteration(alpha_steps=[alpha_, 0.99], R0_map=R0_map, alpha_index=1)[0]
                return R0_map
            except:
                # No cluster joins were found on the final step.
                return R0_map


        patches_to_remove, cluster_targets = fragmentation_iteration(alpha_steps, R0_map, cluster_targets,
                                                                     alpha_index=discontinuity_index)

        # Remove critically-connecting patches for this iteration.
        R0_map = R0_map * np.logical_not(patches_to_remove)
        critical_joins += patches_to_remove

        alpha = alpha_steps[discontinuity_index]
        iteration += 1

        print(f'alpha = {alpha}')
        plt.title(f'critical joins, alpha {alpha} iteration {iteration}')
        plot_R0_clusters(critical_joins)

        if iteration > 20:
            sys.exit('Iterations exceeded expected')
            break

    return R0_map

def scenario_test():
    """
    For a given epicenter, work out containment scenarios from the fragmented map. Return the payoff ratio.
    """
    print('testing fragmented domain for landscape-level control...')
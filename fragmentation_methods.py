"""
Methods related to fragmenting a domain of R0 values.
"""

import numpy as np
from typing import Union, Tuple, List
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
from plotting_methods import plot_cluster_size_vs_alpha, plot_R0_clusters
from cluster_find import get_top_cluster_sizes, label_connected, cluster_freq_count
from run_main import STRUCTURING_ELEMENT

MAX_ITER = 25
TEST_TOP_N = 5  # Test the top N clusters connect to from R0-connected
BELOW_THRESHOLD = 0.99
INTERFACE_MARGIN = 5
MIN_CLUSTER_INTERMEDIATE_SIZE = 2


def find_cluster_size_vs_alpha(alpha_steps, R0_map) -> np.ndarray:
    """
    Find how the maximum-cluster size changes over the alpha-thresholding procedure. Return cluster-size vs alpha.
    """
    largest_cluster_size_vs_alpha = np.zeros_like(alpha_steps)
    for index in range(len(alpha_steps)):
        R0_map_alpha = np.where(R0_map > alpha_steps[index], R0_map, 0)
        cluster_sizes = get_top_cluster_sizes(R0_map=R0_map_alpha, get_top_n=1)[1]  # find largest clusters
        largest_cluster_size_vs_alpha[index] = cluster_sizes[0]

    assert len(largest_cluster_size_vs_alpha) > 1, f'Error, length of cluster-sizes expected > 1.'
    return largest_cluster_size_vs_alpha



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



def test_removal_disconnects(to_remove:List, R0_map_connected:np.ndarray) -> bool:
    """
    Test that the critically-connecting patches that have been identified break the largest cluster as expected.
    """
    R0_fragmented = R0_map_connected * np.logical_not(to_remove)
    R0_fragmented, number_comp = label_connected(R0_map=R0_fragmented)
    connected_comp = np.unique(R0_map_connected)
    if number_comp >= 2:
        assert len(np.where(R0_map_connected)[0]) >= 1.1 * \
               len(np.where(R0_fragmented == 1)[0]), f'Error, expected major fragmentation C >= 1.1 * Max(C1, C2)'

        assert len(connected_comp) == 2, f'Error, expected two elements [0, 1] in R0-connected. Found {connected_comp}'
        assert connected_comp[1] == 1, f'Error, expected R0-connected to have value 1, found {connected_comp[1]}'
        return True


    return False # removal of patches did not fragment the cluster.


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



def find_interface_joins(pre_connected_clusters:np.ndarray, connecting_patches:np.ndarray,
                         cluster_targets: np.ndarray) -> Tuple[Tuple[tuple, tuple], int]:
    """
    Find small patches that connect the target-clusters which lay in the (close-range) cluster interface.
    To do this, we take a binary dilation (one 1 iterations) and find if this neighbours the cluster targets.
    """
    xcoords, ycoords = [], []
    connecting_patch_num = 0
    target_joins = np.unique(cluster_targets)[1:]
    connecting_patch_labelled, num_elements = label_connected(R0_map=connecting_patches)
    for connect_patch_id in range(1, num_elements+1):
        target_patch_indices = np.where(connecting_patch_labelled == connect_patch_id)
        array_mask = np.zeros_like(connecting_patch_labelled)
        array_mask[target_patch_indices] = 1
        array_mask = binary_dilation(array_mask, structure=STRUCTURING_ELEMENT,iterations=1)
        neighbour_ids = pre_connected_clusters[np.where(array_mask)]  # Which clusters do the bd lay inside ?
        neighbour_ids = set([id for id in neighbour_ids if id])
        if len(neighbour_ids) >= 2:
            # Two or more neighbours suggest patch_id joins C1 and C2.
            assert np.array_equal(target_joins, neighbour_ids) == 0, f'Error, expected cluster ids {neighbour_ids} ' \
                                                                     f'to match target interface {target_joins}'
            xcoords.extend(target_patch_indices[0])
            ycoords.extend((target_patch_indices[1]))
            connecting_patch_num += len(target_patch_indices[0])

    return (tuple(xcoords), tuple(ycoords)), connecting_patch_num



def find_intermediate_cluster_joins(R0_post_connect:np.ndarray,
                                    cluster_targets:np.ndarray) -> Tuple[Tuple[tuple, tuple], int]:
    """
     Find the cluster(s) responsible for bridging the gap between C1 and C2 and
     return the relevant edge-patches.
    """
    connecting_patch_num = 0
    xcoords, ycoords = [], []
    cluster_target_values = np.unique(cluster_targets)[1:]
    bd_targets = binary_dilation(cluster_targets, STRUCTURING_ELEMENT)
    in_post_R0_not_targers = label_connected(R0_map=R0_post_connect * np.logical_not(cluster_targets))[0]
    sizes, ids = cluster_freq_count(in_post_R0_not_targers)
    for index, connect_patch_id in enumerate(ids):
        if sizes[index] < MIN_CLUSTER_INTERMEDIATE_SIZE:  # Do not consider small clusters - should already be flagged
            break

        potential_join_indices = np.where(in_post_R0_not_targers == connect_patch_id)
        array_mask = np.zeros_like(R0_post_connect)
        array_mask[potential_join_indices] = 1
        potential_cluster_join = binary_dilation(array_mask, STRUCTURING_ELEMENT)
        neighbour_ids =  np.unique(cluster_targets[np.where(potential_cluster_join)])[1:]
        if np.array_equal(cluster_target_values, neighbour_ids):  # Where cluster joins C1 and C2
            plt.imshow(array_mask)
            plt.show()
            target_patch_indices = np.where(bd_targets & array_mask)
            xcoords.extend(target_patch_indices[0])
            ycoords.extend(target_patch_indices[1])
            connecting_patch_num += len(target_patch_indices[0])

    return (tuple(xcoords), tuple(ycoords)), connecting_patch_num


def find_critically_connecting_patches(R0_pre_connect: np.ndarray, R0_post_connect: np.ndarray,
                                       cluster_targets:np.ndarray, cluster_ids:np.ndarray) -> Tuple[tuple, tuple]:
    """
    If discontinuity is detected, find and return a binary map of the patches of land that join the largest cluster-join.
    """
    # patches which become non-zero in the alpha-step and belong to the post-connected cluster
    became_non_zero = R0_post_connect - np.where(R0_pre_connect, 1, 0)
    became_non_zero = became_non_zero >= 1

    R0_pre_connect_target_interface = np.zeros_like(R0_pre_connect)
    for id in cluster_ids:  # Find an approximate interface for the cluster-targets.
        R0_pre_connect_target_interface += binary_dilation(np.where(cluster_targets == id, 1, 0),
                                                            structure=STRUCTURING_ELEMENT, iterations=INTERFACE_MARGIN)

    # The union of cluster-target interface & the patches which became non-zero must contain the critically connecting patches
    R0_pre_connect_target_interface = R0_pre_connect_target_interface > 1
    R0_pre_connect_target_interface = R0_pre_connect_target_interface & became_non_zero

    # Find patches which lay in the interface and connect the targets
    connection_indicies, interface_remove_num = find_interface_joins(pre_connected_clusters=R0_pre_connect,
                                                           connecting_patches=R0_pre_connect_target_interface,
                                                           cluster_targets=cluster_targets)

    connecting_patches = np.zeros_like(R0_pre_connect)
    connecting_patches[connection_indicies] = 1

    if interface_remove_num and test_removal_disconnects(connecting_patches, R0_post_connect):
        # The patches found in the interface fragmented the cluster
        return connecting_patches

    # Find patches which connect the targets via intermediate cluster-cluster joins.
    connection_indicies, cluster_edge_remove_num = find_intermediate_cluster_joins(R0_post_connect, cluster_targets)
    connecting_patches[connection_indicies] = 1

    if cluster_edge_remove_num and test_removal_disconnects(connecting_patches, R0_post_connect):
        # The patches found from intermediate cluster-join edges fragmented teh cluster
        return connecting_patches

    assert cluster_edge_remove_num + interface_remove_num, f'Error found 0 patches to remove.'
    assert test_removal_disconnects(connecting_patches, R0_post_connect), f'Error, cluster did not fragment.'


def find_alpha_discontinuity(cluster_size_v_alpha:np.ndarray, R0_map:np.ndarray,
                             alpha_steps:Union[list, np.ndarray], cluster_target: Union[None, np.ndarray],
                             discontinuity_index: Union[None, int]) -> Union[int, None]:
    """
    Find the value of alpha which connects the relevant sub-clusters.
    1) On the first iteration, the alpha index responsible for largest increase of a cluster-cluster join
       is returned.
    2) On the proceeding iterations, the value of alpha which connects cluster-targets is returned.
       alpha-index which connects 2 or more clusters ranked in the top N -- 5 by default.
    """
    alpha_ind_sorted = np.argsort(np.gradient(cluster_size_v_alpha))[::-1]
    for index in alpha_ind_sorted:
        # Test what index, if any, gives rise to the relevant cluster-join
        R0_clusters1, sizes = get_top_cluster_sizes(R0_map=R0_map > alpha_steps[index-1],  get_top_n=TEST_TOP_N)[:2]
        R0_clusters2 = get_top_cluster_sizes(R0_map=R0_map > alpha_steps[index], get_top_n=1)[0]
        number_of_joins = np.unique(R0_clusters1[np.where(R0_clusters2)])[1:]  # which clusters join up ?
        cluster_targets = np.unique(cluster_target)[1:]

        # For the 1st iteration, return the index of the largest join.
        if len(number_of_joins) >= 2 and discontinuity_index is None:
            return index

        # Expected 1 or more cluster-joins of the top-ranked clusters.
        if not len(number_of_joins) >= 2:
            continue

        # alpha-stepping should not go back-wards but monotonically decrease
        if index <= discontinuity_index:
            continue

        true_if_in = lambda id: True if id in number_of_joins else False
        target_joins_present = all(true_if_in(id) for id in cluster_targets)

        # target-clusters are not present in the cluster-join, continue to next alpha.
        if not target_joins_present:
            continue

        if assert_targets_in_join(cluster_targets, cluster_target, R0_clusters1, R0_clusters2):
            return index  # cluster-targets are in R0-connected

    return None  # we have not detected any more cluster discontinuities


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

    if cluster_targets is None:  # Set the cluster targets on the first iteration.
        cluster_targets = np.where(R0_map_disconnected == cluster_ids[0], cluster_ids[0], 0)
        cluster_targets[np.where(R0_map_disconnected == cluster_ids[1])] = cluster_ids[1]

    to_remove  = find_critically_connecting_patches(R0_pre_connect=R0_map_disconnected, R0_post_connect=R0_map_connected,
                                                    cluster_targets=cluster_targets, cluster_ids=cluster_ids)


    # find cluster-targets for next iteration
    cluster_targets = update_cluster_targets(R0_map=R0_map, R0_connected=R0_map_connected, patches=to_remove,
                                             alpha=alpha_steps[alpha_index])

    plt.title(f'R0 disconnected @ {alpha_steps[alpha_index - 1]}')
    plot_R0_clusters(R0_map=R0_map_disconnected)
    plt.title(f'R0 connected @ {alpha_steps[alpha_index]}')
    plot_R0_clusters(R0_map=R0_map_connected)
    plt.title(f'R0 fragmented @ {alpha_steps[alpha_index]}')
    plot_R0_clusters(R0_map=cluster_targets)

    return to_remove, cluster_targets


def alpha_stepping_method(alpha_steps:list, R0_map:np.ndarray) -> np.ndarray:
    """
    Perform the /alpha-stepping method over the R0-map in order to find critically-connecting patches.
    """
    alpha = None
    iteration = 0
    cluster_targets = None
    discontinuity_index = None
    fragmentation_process = True
    critical_joins = np.zeros_like(R0_map)
    while fragmentation_process:
        print(f'iteration {iteration} | alpha {alpha} ')
        # get cluster size-vs-alpha
        cluster_size_v_alpha = find_cluster_size_vs_alpha(alpha_steps, R0_map)
        # Find the value of alpha to fragment
        discontinuity_index = find_alpha_discontinuity(cluster_size_v_alpha=cluster_size_v_alpha, R0_map=R0_map,
                                                      alpha_steps=alpha_steps, cluster_target=cluster_targets,
                                                      discontinuity_index=discontinuity_index)
        if alpha is not None:
            # alpha should continually decrease for each iteration
            assert alpha_steps[discontinuity_index] < alpha

        if discontinuity_index is None:
            # No more discontinuities for cluster-targets
            return critical_joins

        # plot_cluster_size_vs_alpha(iteration, alpha_steps, cluster_size_v_alpha, discontinuity_index=discontinuity_index)
        patches_to_remove, cluster_targets = fragmentation_iteration(alpha_steps, R0_map, cluster_targets,
                                                                     alpha_index=discontinuity_index)

        alpha = alpha_steps[discontinuity_index]
        R0_map = R0_map * np.logical_not(patches_to_remove)
        critical_joins += patches_to_remove

        plt.title(f'critical joins @ {iteration}')
        im = plt.imshow(critical_joins)
        plt.colorbar(im)
        plt.show()

        if alpha <= BELOW_THRESHOLD:  # At this point, we have fragmented the cluster-targets
            return critical_joins

        if iteration > MAX_ITER:
            break  # number of expected iterations exceeded

        iteration += 1

    return None


def scenario_test():
    """
    For a given epicenter, work out containment scenarios from the fragmented map. Return the payoff ratio.
    """
    print('testing fragmented domain for landscape-level control...')
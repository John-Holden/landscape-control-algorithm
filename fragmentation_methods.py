"""
Methods related to fragmenting a domain of R0 values.
"""
import sys
import numpy as np
from typing import Union, Tuple, List
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import itertools
from plotting_methods import plot_cluster_size_vs_alpha, plot_R0_clusters
from cluster_find import rank_cluster_map, label_connected, cluster_freq_count
from run_main import STRUCTURING_ELEMENT

TEST_TOP_N = 5  # Test the top N clusters connect to from R0-connected
INTERFACE_MARGIN = 5
MIN_CLUSTER_INTERMEDIATE_SIZE = 2


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
        assert len(connected_comp) == 2, f'Error, expected two elements [0, 1] in R0-connected. Found {connected_comp}'
        assert connected_comp[1] == 1, f'Error, expected R0-connected to have value 1, found {connected_comp[1]}'
        return True


    return False # removal of patches did not fragment the cluster.


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


def fragmentation_iteration(alpha_steps:list, R0_map:np.ndarray, cluster_targets:np.ndarray,
                            alpha_index:int) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Access the how cluster-sizes changes vs alpha, find largest cluster-join and critically-connecting patches. Return
    the culled R0-map
    """
    # find clusters pre connected cluster state
    R0_map_disconnected = np.where(R0_map > alpha_steps[alpha_index], R0_map, 0)
    # assume top n sub-clusters in pre-connected
    R0_map_disconnected, cluster_sizes, cluster_ids  = rank_cluster_map(R0_map=R0_map_disconnected, get_ranks=10)

    # find clusters post connected cluster state
    R0_map_connected = np.where(R0_map > alpha_steps[alpha_index+1], R0_map, 0)
    # assume connected cluster is largest-ranked
    R0_map_connected = rank_cluster_map(R0_map=R0_map_connected, get_ranks=1)[0]

    if cluster_targets is None:  # Set the cluster targets on the first iteration.
        cluster_targets = np.where(R0_map_disconnected == cluster_ids[0], cluster_ids[0], 0)
        cluster_targets[np.where(R0_map_disconnected == cluster_ids[1])] = cluster_ids[1]

    to_remove  = find_critically_connecting_patches(R0_pre_connect=R0_map_disconnected, R0_post_connect=R0_map_connected,
                                                    cluster_targets=cluster_targets, cluster_ids=cluster_ids)

    # find cluster-targets for next iteration
    cluster_targets = update_targets(cluster_targets, R0_map * np.logical_not(to_remove), alpha_steps[alpha_index+1])
    # plt.title(f'R0 disconnected @ {alpha_steps[alpha_index]}')
    # plot_R0_clusters(R0_map=R0_map_disconnected)
    # plt.title(f'R0 connected @ {alpha_steps[alpha_index+1]}')
    # plot_R0_clusters(R0_map=R0_map_connected)
    # plt.title(f'R0 fragmented @ {alpha_steps[alpha_index+1]}')
    # plot_R0_clusters(R0_map=cluster_targets)

    return to_remove, cluster_targets


def find_alpha_discontinuities(alpha_steps, R0_map, join_history=False):
    """
    Find how the maximum-cluster size changes over the alpha-thresholding procedure. Return cluster-size vs alpha.
    """

    joins_at_alpha = {}
    for index in range(len(alpha_steps)-1):
        R0_map_alpha = np.where(R0_map > alpha_steps[index], R0_map, 0)
        R0_map_alpha, cluster_sizes, cluster_ids = rank_cluster_map(R0_map=R0_map_alpha, get_ranks=5)
        R0_map_d_alpha = np.where(R0_map > alpha_steps[index+1], R0_map, 0)
        R0_map_d_alpha, d_cluster_sizes = rank_cluster_map(R0_map=R0_map_d_alpha, get_ranks=1)[:2]
        cluster_joins = np.unique(R0_map_alpha[np.where(R0_map_d_alpha)])
        cluster_joins = [rank for rank in cluster_joins if rank not in [0]]

        if len(cluster_joins) == 1:
            continue
        targets = [comb for comb in itertools.combinations(cluster_joins, 2) if 1 in comb]
        cluster_size_ratios = [cluster_sizes[comb[1]-1]/cluster_sizes[comb[0]-1] for comb in targets]
        joins_at_alpha[index]= { 'cluster_targets': targets,
                                 f'ratios' : cluster_size_ratios}

    return joins_at_alpha


def assert_discontinuity_in_step(alpha:float, d_alpha:float, R0_map:np.ndarray, cluster_targets:np.ndarray) -> bool:
    """
    Test whether or not clusters-join for the alpha step: alpha ->
    """
    alpha_sizes = rank_cluster_map(R0_map > alpha, get_ranks=1)[1]
    R0_d_alpha, d_alpha_sizes = rank_cluster_map(R0_map > d_alpha, get_ranks=1)[:2]
    joins = np.unique(cluster_targets[np.where(R0_d_alpha)])
    targets_in_d_alpha = True if 1 in joins and 2 in joins else False
    signif_rise = True if d_alpha_sizes[0] > 1.1 * alpha_sizes[0] else False
    if targets_in_d_alpha and signif_rise:
        return True

    return False


def update_targets(cluster_targets:np.ndarray, R0_map:np.ndarray, d_alpha:float) -> np.ndarray:
    """
    For each alpha-step, update the cluster-targets which monotonically increase.
    """
    R0_d_alpha = rank_cluster_map(R0_map > d_alpha, get_ranks=10)[0]
    cluster_1 = np.unique(R0_d_alpha[np.where(cluster_targets == 1)])
    cluster_2 = np.unique(R0_d_alpha[np.where(cluster_targets == 2)])
    assert len(cluster_1) == 1, f'Error, expect target 1 to consist of one element. Found {cluster_1}'
    assert len(cluster_2) == 1, f'Error, expect target 2 to consist of one element. Found {cluster_2}'
    assert cluster_1[0] != cluster_2[0], f'Error, cluster target 1 {cluster_1} should not be equal to 2 {cluster_2}'
    return np.where(R0_d_alpha == cluster_1[0], 1, 0) + np.where(R0_d_alpha == cluster_2, 2, 0)


def get_payoff(patches:np.ndarray, R0_map:np.ndarray) -> float:
    """
    Find the payoff, defined as the second largest fragment dived by the number of patches to fragment.
    """
    target_sizes = rank_cluster_map(R0_map, get_ranks=2)[1]
    return target_sizes[1] / len(patches[0])


def find_best(frag_method):
    """
    Find best fragmentation - iterate over different alpha-valued initial conditions
    """
    def iterator(R0_map):
        alpha_steps = get_alpha_steps('auto', R0_max=5, R0_min=0.99, number_of_steps=30)
        join_history = find_alpha_discontinuities(alpha_steps, R0_map)
        c = 0
        best_payoff = 0
        for alpha_index, join_info in join_history.items():
            alpha_steps_ = alpha_steps[alpha_index:]
            for cluster_join, size_ratio in zip(join_info['cluster_targets'], join_info['ratios']):
                if size_ratio < 0.10:
                    continue  # negate small joins

                cluster_targets = rank_cluster_map(R0_map > alpha_steps_[0], get_ranks=cluster_join)[0]
                connecting_patches_indices, R0_map_fragmented = frag_method(R0_map, cluster_targets, alpha_steps_)
                payoff = get_payoff(connecting_patches_indices, R0_map_fragmented)

                if payoff > best_payoff:
                    best_payoff = payoff
                    optimal_fragmentation = R0_map_fragmented
                    optimal_indices = connecting_patches_indices

                c+=1

        return optimal_indices, optimal_fragmentation

    return iterator


@find_best
def alpha_stepping_method(R0_map:np.ndarray, cluster_targets:np.ndarray=None,  alpha_steps:list = None) -> np.ndarray:
    """
    Perform the /alpha-stepping method over the R0-map in order to find critically-connecting patches.
    """
    critical_joins = np.zeros_like(R0_map)
    for alpha_index in range(len(alpha_steps) - 1):
        # Iterate through alpha index until alpha = 0.99
        if not assert_discontinuity_in_step(alpha_steps[alpha_index], alpha_steps[alpha_index+1],
                                                          R0_map, cluster_targets):

            if not alpha_index:
                sys.exit('Error, on the first iteration, expected a discontinuity')

            cluster_targets = update_targets(cluster_targets, R0_map, alpha_steps[alpha_index+1])
            continue

        patches_to_remove, cluster_targets = fragmentation_iteration(alpha_steps, R0_map, cluster_targets, alpha_index)

        R0_map = R0_map * np.logical_not(patches_to_remove)
        critical_joins += patches_to_remove

    critical_joins = np.where(critical_joins)
    critical_joins = (tuple(critical_joins[0]), tuple(critical_joins[1]))
    return critical_joins, R0_map





def update_fragmentation_target(R0_map:np.ndarray, patch_indices:tuple) -> np.ndarray:
    """
    Chose the largest cluster in the fragmented domain.
    """
    R0_map[patch_indices] = 0
    return rank_cluster_map(R0_map, get_ranks=1)[0]


def scenario_test():
    """
    For a given epicenter, work out containment scenarios from the fragmented map. Return the payoff ratio.
    """
    print('testing fragmented domain for landscape-level control...')
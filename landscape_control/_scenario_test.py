import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from typing import Union, Tuple, Iterable

from ._cluster_find import rank_cluster_map
from parameters_and_setup import STRUCTURING_ELEMENT
from .plotting_methods import plot_fragmented_domain, plot_R0_clusters


def find_inner_iterations(fragmented_domain: np.ndarray, R0_map: np.ndarray, iteration: Union[int, tuple], epi_c: tuple,
                          iterations: int, redundant_lines: dict, multi_comb_mode: bool = False):
    """
    find the relevant connected lines for a given iteration-fragmentation, take the negation of this to find the
    disconnected irrelevant lines and append to a dict
    """

    fragment_at_iter = domain_at_iteration(R0_map, fragmented_domain, iteration)[0]
    if fragment_at_iter is None:
        if multi_comb_mode:
            return

        nn_lines = np.unique(fragmented_domain[np.where(
            binary_dilation(fragmented_domain == iteration, STRUCTURING_ELEMENT)
        )])
        potential_multi_connect = tuple([i for i in nn_lines if i])
        return potential_multi_connect

    # which fragmented lines are contained on the fragment ; iter_
    connected_lines = np.unique(
        fragmented_domain[np.where(fragment_at_iter == fragment_at_iter[epi_c])]
    )

    # which lines are redundant ie. { target : [redundant fragmentation iterations] }
    non_trivial_lines = set(list(iteration)) if isinstance(iteration, tuple) else {iteration}
    non_trivial_lines = set(range(1, iterations+1)) - non_trivial_lines
    redundant_outer_iterations = non_trivial_lines - set([i for i in connected_lines if i])

    if redundant_outer_iterations:
        redundant_lines[iteration] = redundant_outer_iterations


def gen_frag_combos(redundant_combos: dict, iterations: int):
    """
    From the information of redundant fragmentation combinations, return a list of relevant combinations for given epi_C
    """
    print('redundant combinations : ', redundant_combos)
    assert 0

    frag_comb = []
    iter_ = [1 + i for i in range(iterations)]
    for i in iter_:
        comb = list(itertools.combinations(iter_, r=i))
        if len(comb) == 0:
            continue

        frag_comb.extend(comb)
    return frag_comb


def find_frag_combos(R0_map: np.ndarray, fragmented_domain: np.ndarray, iterations: int, epi_c: tuple) -> dict:
    """
    Find all combinations for given iteration
    """
    redundant_lines = {}
    potential_multi_fragment = []
    for iter_ in range(1, iterations + 1):
        potential_multi_combo = find_inner_iterations(fragmented_domain, R0_map, iter_, epi_c, iterations,
                                                      redundant_lines)

        if potential_multi_combo:
            potential_multi_fragment.append(potential_multi_combo)
            redundant_lines[iter_] = None

    if potential_multi_fragment:  # find
        potential_multi_fragment_combos = []
        for iter_combo in potential_multi_fragment:
            for len_ in range(2, len(iter_combo)+1):
                comb = list(itertools.combinations(iter_combo, r=len_))
                if comb not in potential_multi_fragment_combos:
                    potential_multi_fragment_combos.extend(comb)

        for iter_combo in potential_multi_fragment_combos:
            find_inner_iterations(fragmented_domain, R0_map, iter_combo, epi_c, iterations, redundant_lines,
                                  multi_comb_mode=True)


    return gen_frag_combos(redundant_lines, iterations)


def get_epi_c(R0_domain: np.ndarray, fragmented_domain: np.ndarray) -> Tuple:
    """
    Update list of N random epicenters
    """

    fragmented, _, ids = rank_cluster_map(R0_domain * np.logical_not(fragmented_domain))
    epicenters = []
    for id_ in ids:
        cluster_indices = np.where(fragmented == id_)
        row_epi = cluster_indices[0].sum() / len(cluster_indices[0])
        col_epi = cluster_indices[1].sum() / len(cluster_indices[0])
        potential_row_index = cluster_indices[0] - row_epi  # find closest patch (by row) to the cluster COM
        potential_col_index = cluster_indices[1] - col_epi  # find closest patch (by row) to the cluster COM
        potential_index = np.sqrt(potential_row_index ** 2 + potential_col_index ** 2)
        row_epi = cluster_indices[0][np.argmin(potential_index)]
        col_epi = cluster_indices[1][np.argmin(potential_index)]
        assert fragmented[row_epi, col_epi] == id_, f'Error, expected {id_} found {fragmented[row_epi, col_epi]}'
        epicenters.append((row_epi, col_epi))

    return tuple(epicenters)


def domain_at_iteration(R0_domain: np.ndarray, fragmented_domain: np.ndarray,
                        iterations: Union[int, Iterable]) -> Tuple[Union[None, np.ndarray], np.ndarray]:
    """
    Find fragmented domain @ given iteration
    """
    fragment_lines = np.zeros_like(R0_domain)
    if isinstance(iterations, int):
        iterations = [iterations]

    R0_fragmented = np.copy(R0_domain)
    for iteration in iterations:
        R0_fragmented = R0_fragmented * np.logical_not(fragmented_domain == iteration)
        fragment_lines += np.where(fragmented_domain == iteration, iteration, 0)

    R0_fragmented, _, _ = rank_cluster_map(R0_fragmented)

    if len(_) == len(iterations) + 1:  # lines broke the cluster as expected
        return R0_fragmented, fragment_lines

    return None, fragment_lines


def get_epicenter_payoff(epicenter: tuple, R0_fragmented: np.ndarray,
                         fragment_lines: np.ndarray) -> Tuple[np.ndarray, tuple, int, int]:
    """
    Find the payoff for a single fragmentation combination
    """
    target = R0_fragmented[epicenter[0], epicenter[1]]

    target = np.where(R0_fragmented == target)
    num_patches_removed = len(target[0])
    arr_mask = np.zeros_like(R0_fragmented)
    arr_mask[target] = 1

    bd_target = binary_dilation(arr_mask, structure=STRUCTURING_ELEMENT)

    fragment_lines = np.logical_and(fragment_lines, bd_target) * fragment_lines

    relevant_lines = np.unique(fragment_lines)
    relevant_lines = [int(line) for line in relevant_lines if line != 0]
    num_culled = len(np.where(fragment_lines)[0])

    return fragment_lines, tuple(relevant_lines), num_patches_removed, num_culled


def add_rank_to_dict(payoffs: list, epicenters: list, relevant_lines: list, scenario_store: dict):
    """
    Rank all the payoff results and append to scenario_store dictionary.
    """
    payoffs = np.array(payoffs)
    epicenters = np.array(epicenters)
    relevant_lines = np.array(relevant_lines)

    ranked_args = np.argsort(payoffs)[::-1]
    epicenters = epicenters[ranked_args]
    relevant_lines = relevant_lines[ranked_args]
    ranks = range(1, len(relevant_lines) + 1)
    for epi, line, rank in zip(epicenters, relevant_lines, ranks):
        scenario_store[tuple(epi)][line]['rank'] = rank

    return scenario_store

import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from typing import Union, Tuple, Iterable

from ._cluster_find import rank_cluster_map
from parameters_and_setup import STRUCTURING_ELEMENT
from .plotting_methods import plot_fragmented_domain, plot_R0_clusters



def fragment_combination(iterations: int) -> list:
    """
    Find all combinations for given iteration
    """
    frag_comb = []
    iter_ = [1+i for i in range(iterations)]
    for i in iter_:
        comb = list(itertools.combinations(iter_, r=i))
        if len(comb) == 0:
            continue

        frag_comb.extend(comb)

    return frag_comb


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
        potential_index = np.sqrt(potential_row_index**2 + potential_col_index**2)
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



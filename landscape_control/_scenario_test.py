import sys
import json
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from typing import Union, Tuple, Iterable

from ._cluster_find import rank_cluster_map
from ._plotting_methods import plot_fragmented_domain
from parameters_and_setup import PATH_TO_INPUT_DATA, STRUCTURING_ELEMENT


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


def get_epi_c(R0_domain:np.ndarray, all_culled_patches:np.ndarray, number: int) -> Tuple[tuple, tuple]:
    """
    Update list of N random epicenters
    """

    domain_indices = np.where(R0_domain * np.where(all_culled_patches, 0, 1))
    randomise_index = random.sample(range(0, len(domain_indices[0])), number)

    epi_c = [domain_indices[0][randomise_index],
             domain_indices[1][randomise_index]]

    return tuple((i, j) for i, j in zip(epi_c[0], epi_c[1]))


def domain_at_iteration(R0_domain:np.ndarray, connecting_patches:dict, iterations: Union[int, Iterable]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find fragmented domain @ given iteration
    """
    fragment_lines = np.zeros_like(R0_domain)
    if isinstance(iterations, int):
        iterations = [iterations]

    R0_fragmented = np.copy(R0_domain)
    for iteration in iterations:
        R0_fragmented[connecting_patches[iteration - 1]] = 0
        fragment_lines[connecting_patches[iteration - 1]] = iteration

    return rank_cluster_map(R0_fragmented)[0], fragment_lines


def get_epicenter_payoff(epicenter: tuple, R0_domain:np.ndarray, R0_fragmented: np.ndarray,
                         fragment_lines: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Find the payoff for a single fragmentation combination
    """
    target = R0_fragmented[epicenter[0], epicenter[1]]

    if not target:
        plt.title('error')
        plot_fragmented_domain(fragment_lines, np.copy(R0_domain), epicenter, show_text=True)

    assert target, 'Error, not expecting a zero-valued cluster-target.'

    target = np.where(R0_fragmented == target)
    num_patches_removed = len(target[0])
    arr_mask = np.zeros_like(R0_fragmented)
    arr_mask[target] = 1

    bd_target = binary_dilation(arr_mask, structure=STRUCTURING_ELEMENT)

    relevant_lines = fragment_lines[np.where(np.logical_and(bd_target, fragment_lines))]
    relevant_lines = np.unique(relevant_lines)
    relevant_lines = [int(line) for line in relevant_lines]

    num_culled = [len(np.where(fragment_lines == i)[0]) for i in relevant_lines]

    return tuple(relevant_lines), num_patches_removed, sum(num_culled)



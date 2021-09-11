import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .plotting_methods import plot_R0_clusters, plot_R0_map

from typing import Iterable, Union, Any
from scipy.optimize import curve_fit
from ._cluster_find import rank_cluster_map


def linear_func(xdata: Iterable, c: int):
    return c * xdata


def threshold_domain(domain:np.ndarray, density:float, rank:int=1):
    susceptible = np.where(domain > density, 1, 0)
    return rank_cluster_map(susceptible, get_ranks=rank)[0]


def get_R0_gradient_fitting(species_distribution_map: np.ndarray, rhos: np.ndarray,
                            R0_v_rho_mapping: np.ndarray, print_fitting=False) -> np.ndarray:
    """
     For an array of R0 vs rho values, fit data to linear function. Then return tree-density mapped to R0-values.
    """

    popt, pcov = curve_fit(linear_func, rhos, R0_v_rho_mapping)
    if print_fitting:
        print(f'Fitted gradients {popt[0]}, Variance {pcov[0]}')
        import matplotlib.pyplot as plt
        plt.plot(rhos, popt[0]*rhos)
        plt.plot(rhos, R0_v_rho_mapping)
        plt.title(f'grad  = {popt[0]}')
        plt.show()

    return species_distribution_map * popt[0]


def coarse_grain(domain, cg_factor) -> np.ndarray:
    """
    Re-scale original dataset to a given granularity, re-shape to:
        cg_factor km^2 x cg_factor km^2

    :return re-scaled array
    """
    if 1 in np.isnan(domain):
        domain = np.where(np.isnan(domain), 0, domain)

    if cg_factor == 1:
        return domain

    x_ind = 0
    new_xaxis = np.arange(0, domain.shape[0], cg_factor)
    new_yaxis = np.arange(0, domain.shape[1], cg_factor)
    cg_arr = np.zeros([len(new_xaxis), len(new_yaxis)])
    for row in new_xaxis:
        y_ind = 0
        for col in new_yaxis:
            patch = domain[row:row + cg_factor, col:col + cg_factor]
            av_value = np.sum(patch)
            cg_arr[x_ind][y_ind] = av_value

            y_ind += 1
        x_ind += 1
    cg_arr = cg_arr / cg_factor ** 2
    if 1 in np.isnan(domain):
        cg_arr[np.where(cg_arr == 0)] = np.nan

    return cg_arr


def get_R0_map(raw_species_data: np.ndarray, R0_vs_rho: np.ndarray,
               rhos: np.ndarray, coarse_grain_factor: int) -> Union[None, np.ndarray]:
    """
    Process a single domain, for one beta value, and return R0 map.
    """
    if max(R0_vs_rho) < 1:
        msg = 'Warning: trivial data-set, max R0 < 1'
        warnings.warn(msg)
        return None

    # raw_species_data_cg = raw_species_data if coarse_grain_factor == 1 else coarse_grain(raw_species_data, coarse_grain_factor)
    raw_species_data_cg = coarse_grain(raw_species_data, coarse_grain_factor)
    return get_R0_gradient_fitting(raw_species_data_cg, rhos, R0_vs_rho, print_fitting=False)


def trim_domain(domain: np.ndarray):
    """Find bounds around the cluster target and strip all trivial points surrounding."""
    domain_indices = np.where(domain)
    domain_indices = [min(domain_indices[0]) - 2, max(domain_indices[0]) + 2, min(domain_indices[1]) - 2,
                      max(domain_indices[1]) + 2]
    return domain[domain_indices[0]:domain_indices[1], domain_indices[2]: domain_indices[3]]  # trim domain and save.


def is_too_small(R0_map: np.ndarray, limit: int = 100) -> bool:
    """the domain under consideration is too small for processing"""
    return np.count_nonzero(R0_map) < limit


def process_R0_map(R0_map_raw: np.ndarray, get_cluster: int, threshold: Union[int, float] = 1) -> np.ndarray:
    """
    Strip patches below the threshold, 1 by default and return the cluster target. The domain is also cropped around the
    target cluster.
    """

    R0_map = np.where(R0_map_raw > threshold, R0_map_raw, 0)  # consider above threshold positions
    R0_map = R0_map * np.array(rank_cluster_map(R0_map, get_ranks=get_cluster)[0] > 0).astype(
        int)  # concentrate on the largest cluster

    R0_indices = np.where(R0_map)
    R0_indices = [min(R0_indices[0]) - 2, max(R0_indices[0]) + 2, min(R0_indices[1]) - 2, max(R0_indices[1]) + 2]
    R0_map = R0_map[R0_indices[0]:R0_indices[1], R0_indices[2]: R0_indices[3]]  # trim domain and save.
    R0_map_ = np.zeros(shape=(R0_map.shape[0]+2, R0_map.shape[1]+2))
    R0_map_[1:-1, 1:-1] = R0_map

    if any(R0_map_[0]) or any(R0_map_[-1]) or any(R0_map_[:, 0]) or any(R0_map_[:, -1]):
        raise Exception('Domain did not get correctly re-shaped')
    return R0_map_


def get_clusters_over_betas(ensemble: Any, cg_factor: int = 5, get_rank: int = 1,
                            save: bool = False, plot_clusters: bool = False, plot_R0_maps: bool = False) -> np.ndarray:
    """
    For each value of beta, find the top N ranked cluster size(s). Return an array of cluster sizes vs beta..
    """
    species_distribution_map = coarse_grain(domain=ensemble.raw_data, cg_factor=cg_factor)
    cluster_sizes = np.zeros(len(ensemble.betas)) if get_rank ==1 else np.zeros(shape=(len(ensemble.betas), get_rank))

    for beta_index in range(len(ensemble.betas)):
        print(f'beta {beta_index}/{len(ensemble.betas)}')
        R0_vs_rho = ensemble.R0_vs_rho_beta[beta_index]
        R0_map = get_R0_gradient_fitting(species_distribution_map, ensemble.rhos, R0_vs_rho)
        if R0_map.max() < 1:
            cluster_sizes[beta_index] = 0
            continue

        if plot_R0_maps:
            beta_title = rf'$\beta =$, {round(ensemble.betas[beta_index], 7)}'
            plot_R0_map(R0_map, save_name=f'R0_raw_beta_{beta_index}', title=beta_title, save=True)

        R0_map, sizes, _ = rank_cluster_map(R0_map > 1)
        cluster_sizes[beta_index] = sizes[:get_rank]
        if plot_clusters:
            flash = 7 <= beta_index <= 9
            beta_title = rf'$\beta =$, {round(ensemble.betas[beta_index], 7)}'
            plot_R0_clusters(R0_map, rank=10, save=True, save_name=f'R0_clustering_beta_{beta_index}', title=beta_title,
                             flash=flash, ext='pdf')

    cluster_sizes = cluster_sizes * cg_factor**2
    if save:  # save cluster size in units km^2
        if os.path.exists(f'{ensemble.path_to_ensemble}/cluster_size_vs_beta.npy'):
            msg = f'\n Overwriting data for : {ensemble.path_to_ensemble}/cluster_size_vs_beta'
            warnings.warn(msg), 'done'

        np.save(f'{ensemble.path_to_ensemble}/cluster_size_vs_beta', cluster_sizes)

    return cluster_sizes






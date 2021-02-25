import numpy as np
from typing import Iterable, Union
from scipy.optimize import curve_fit
from ._cluster_find import rank_cluster_map


def linear_func(xdata: Iterable, c: int):
    return c * xdata


def get_R0_gradient_fitting(species_distribution_map: np.ndarray, rhos: np.ndarray,
                            R0_v_rho_mapping: np.ndarray, print_fitting=False) -> np.ndarray:
    """
     For an array of R0 vs rho values, fit data to linear function. Then return tree-density mapped to R0-values.
    """
    popt, pcov = curve_fit(linear_func, rhos, R0_v_rho_mapping)
    if print_fitting:
        print(f'Fitted gradients {popt[0]}, Variance {pcov[0]}')
    return species_distribution_map * popt[0]


def coarse_grain(domain, cg_factor) -> 'float type, arr[n x m]':
    """
    Re-scale original dataset to a given granularity, re-shape to:
        cg_factor km^2 x cg_factor km^2
    """
    if 1 in np.isnan(domain):
        domain = np.where(np.isnan(domain), 0, domain)

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


def get_R0_map(raw_species_data:np.ndarray, R0_vs_rho:np.ndarray,
               rhos:np.ndarray, coarse_grain_factor:Union[None, int]=None) -> np.ndarray:
    """
    Process a single domain, for one beta value, and return R0 map.
    """
    if max(R0_vs_rho) < 1:
        print('Warning: trivial data-set, max R0 < 1')

    if coarse_grain_factor is not None:
        raw_species_data_cg = coarse_grain(domain=raw_species_data, cg_factor=coarse_grain_factor)

    return get_R0_gradient_fitting(raw_species_data_cg, rhos, R0_vs_rho)


def process_R0_map(R0_map_raw:np.ndarray, get_cluster:int, threshold:Union[int, float]=1) -> np.ndarray:
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

    return R0_map


# todo integrate get_clusters_over_betas with class
# def get_clusters_over_betas(ensemble_name:str, cluster_ranks:int, coarse_grain_level:int,
#                             save:bool, plot:bool) -> np.ndarray:
#     """
#     For each value of beta, find the top N ranked cluster size(s). Return an array of cluster sizes vs beta..
#     """
#     ash_ensemble = EnsembleInfo(ensemble_name=ensemble_name)  # initialise domain structures
#     species_distribution_map = coarse_grain(domain=ash_ensemble.raw_data, cg_factor=coarse_grain_level)
#     cluster_sizes = [None] * len(ash_ensemble.betas)
#     print(f'RHOS : {ash_ensemble.rhos} | {len(ash_ensemble.rhos)}')
#     for beta_index in range(len(ash_ensemble.betas)):
#         R0_vs_rho = ash_ensemble.R0_vs_rho_beta[beta_index]
#         R0_map = get_R0_gradient_fitting(species_distribution_map=species_distribution_map, rhos=ash_ensemble.rhos,
#                                              R0_v_rho_mapping=R0_vs_rho)
#
#         ash_clustering = Cluster_sturct(R0_map=R0_map)  # init cluster class
#         ash_clustering.apply_R0_threshold(R0_threshold=1)  # negate below threshold points
#         ash_clustering.label_connected()  # label connected points
#         ranked_cluster_map = ash_clustering.rank_R0_cluster_map(rank_N=cluster_ranks)  # rank top N clusters
#         cluster_sizes[beta_index] = len(np.where(ranked_cluster_map)[0]) * coarse_grain_level**2
#
#         if plot:
#             plt.title(f'Betas : {ash_ensemble.betas[beta_index]}')
#             im = plt.imshow(ranked_cluster_map)
#             plt.colorbar(im)
#             plt.show()
#
#     if save: # save cluster size in units km^2
#         np.save(f'{ash_ensemble.path_to_ensemble}/cluster_size_vs_beta', cluster_sizes)
#
#     return cluster_sizes
"""
High level functions that orchestrate methods related to cluster-finding, domain processing
and map fragmentation.
"""
import datetime
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from run_main import Ensemble_info
from plotting_methods import plot_R0_clusters
from cluster_find import Cluster_sturct, get_top_cluster_sizes
from domain_methods import coarse_grain, get_R0_gradient_fitting
from fragmentation_methods import get_alpha_steps, alpha_stepping_method


def get_single_R0_cluster_map(ensemble_name:str, coarse_grain_factor:int, beta_index:float) -> np.ndarray:
    """
    Process a single domain, for one beta value, and return R0 map.
    """
    ash_ensemble = Ensemble_info(ensemble_name=ensemble_name)  # initialise domain structures for ensemble
    species_distribution_map = coarse_grain(domain=ash_ensemble.raw_data, cg_factor=coarse_grain_factor)
    R0_vs_rho = ash_ensemble.R0_vs_rho_beta[beta_index]
    return get_R0_gradient_fitting(species_distribution_map=species_distribution_map, rhos=ash_ensemble.rhos,
                                   R0_v_rho_mapping=R0_vs_rho, print_fitting=True)


def get_clusters_over_betas(ensemble_name:str, cluster_ranks:int, coarse_grain_level:int,
                            save:bool, plot:bool) -> np.ndarray:
    """
    For each value of beta, find the top N ranked cluster size(s). Return an array of cluster sizes vs beta..
    """
    ash_ensemble = Ensemble_info(ensemble_name=ensemble_name)  # initialise domain structures
    species_distribution_map = coarse_grain(domain=ash_ensemble.raw_data, cg_factor=coarse_grain_level)
    cluster_sizes = [None] * len(ash_ensemble.betas)
    print(f'RHOS : {ash_ensemble.rhos} | {len(ash_ensemble.rhos)}')
    for beta_index in range(len(ash_ensemble.betas)):
        R0_vs_rho = ash_ensemble.R0_vs_rho_beta[beta_index]
        R0_map = get_R0_gradient_fitting(species_distribution_map=species_distribution_map, rhos=ash_ensemble.rhos,
                                             R0_v_rho_mapping=R0_vs_rho)

        ash_clustering = Cluster_sturct(R0_map=R0_map)  # init cluster class
        ash_clustering.apply_R0_threshold(R0_threshold=1)  # negate below threshold points
        ash_clustering.label_connected()  # label connected points
        ranked_cluster_map = ash_clustering.rank_R0_cluster_map(rank_N=cluster_ranks)  # rank top N clusters
        cluster_sizes[beta_index] = len(np.where(ranked_cluster_map)[0]) * coarse_grain_level**2

        if plot:
            plt.title(f'Betas : {ash_ensemble.betas[beta_index]}')
            im = plt.imshow(ranked_cluster_map)
            plt.colorbar(im)
            plt.show()

    if save: # save cluster size in units km^2
        np.save(f'{ash_ensemble.path_to_ensemble}/cluster_size_vs_beta', cluster_sizes)

    return cluster_sizes


def fragment_R0_map(alpha_steps: Union[list, float, int, str],
                    R0_map_raw: np.ndarray, fragmentation_iterations:int) -> np.ndarray:
    """
    Iteratively fragment the largest cluster in the R0_map via targeted tree felling algorithm
    i.e. the `alpha-stepping' method. Save felled patches to file. Return fragmented domain.
    :rtype: object
    """
    R0_map = np.where(R0_map_raw > 1, R0_map_raw, 0)  # consider above threshold positions
    R0_map = R0_map * np.array(get_top_cluster_sizes(R0_map, get_top_n=1)[0] > 0).astype(int)  # concentrate on the largest cluster
    R0_indices = np.where(R0_map)
    R0_indices = [min(R0_indices[0]), max(R0_indices[0]), min(R0_indices[1]), max(R0_indices[1])]
    R0_map = R0_map[R0_indices[0]:R0_indices[1], R0_indices[2]: R0_indices[3]]  # trim domain
    alpha_steps = get_alpha_steps(alpha_steps, R0_max=4, R0_min=0.99, number_of_steps=30)
    for iteration in range(fragmentation_iterations):
        time = datetime.datetime.now()
        critically_connecting_patches = alpha_stepping_method(alpha_steps, R0_map)
        plot_R0_clusters(R0_map=critically_connecting_patches)
        print(f'Time taken to fragment: {datetime.datetime.now() - time}')

    return


"""
High level functions that orchestrate methods related to cluster-finding, domain processing
and map fragmentation.
"""
import os
import json
import datetime
import numpy as np
from typing import Union, Iterable, Type
import matplotlib.pyplot as plt
from run_main import Ensemble_info
from cluster_find import Cluster_sturct, rank_cluster_map
from domain_methods import coarse_grain, get_R0_gradient_fitting
from plotting_methods import  plot_fragmented_domain, plot_R0_clusters
from fragmentation_methods import alpha_stepping_method, update_fragmentation_target

FRAGMENT_RANK = 1


def get_single_R0_cluster_map(ensemble_info:Type[Ensemble_info], coarse_grain_factor:int, beta_index:float) -> np.ndarray:
    """
    Process a single domain, for one beta value, and return R0 map.
    """

    R0_vs_rho = ensemble_info.R0_vs_rho_beta[beta_index]
    if max(R0_vs_rho) < 1:
        print('Warning: trivial data-set, max R0 < 1')

    species_distribution_map = coarse_grain(domain=ensemble_info.raw_data, cg_factor=coarse_grain_factor)

    return get_R0_gradient_fitting(species_distribution_map, ensemble_info.rhos, R0_v_rho_mapping=R0_vs_rho)


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


def orchestrate_fragmentation(ensemble_name):  # import & run delegator methods
    from delegator_methods import get_single_R0_cluster_map, fragment_R0_map
    # define fragmentation parameters
    coarse_grain_factor = 5
    iterations = 10
    beta_index = 1

    # get ensemble
    ensemble_info = Ensemble_info(ensemble_name)

    save_name = f'{ensemble_info.path_to_ensemble}/fragmentation_data/'
    save_name += f'/beta_index_{beta_index}_iterations_{iterations}_cg_{coarse_grain_factor}.json'
    if os.path.exists(save_name):
        print(f'\nWarning: data {save_name} already exists!')

    # if it exists, load R0-map
    if os.path.exists(f'{ensemble_info.path_to_ensemble}/fitted_R0_map.npy'):
        R0_out = np.load(f'{ensemble_info.path_to_ensemble}/fitted_R0_map.npy')
    # generate R0-map
    else:
        print('generating R0-map')
        R0_out = get_single_R0_cluster_map(ensemble_info, coarse_grain_factor, beta_index)
        np.save(f'{ensemble_info.path_to_ensemble}/fitted_R0_map', R0_out)

    # Run fragmentation algorithm
    connecting_patches =  fragment_R0_map(R0_out, iterations, ensemble_info.path_to_ensemble, plot=True)

    # save connecting patches to file.
    with open(save_name, 'w') as outfile:
        json.dump(connecting_patches, outfile, indent=4)


def fragment_R0_map(R0_map_raw: np.ndarray, fragmentation_iterations:int, save_to_path:str, plot:bool=False) -> dict:
    """
    Iteratively fragment the largest cluster in the R0_map via targeted tree felling algorithm
    i.e. the `alpha-stepping' method. Save felled patches to file. Return fragmented domain.
    :rtype: object
    """
    if R0_map_raw.max() < 1:
        # Trivial map
        return None

    connecting_patches = {}
    R0_map = np.where(R0_map_raw > 1, R0_map_raw, 0)  # consider above threshold positions
    R0_map = R0_map * np.array(rank_cluster_map(R0_map, get_ranks=FRAGMENT_RANK)[0] > 0).astype(int)  # concentrate on the largest cluster
    R0_indices = np.where(R0_map)
    R0_indices = [min(R0_indices[0]), max(R0_indices[0]), min(R0_indices[1]), max(R0_indices[1])]
    R0_map = R0_map[R0_indices[0]:R0_indices[1], R0_indices[2]: R0_indices[3]]  # trim domain and save.
    np.save(f'{save_to_path}/R0_map_rank_{FRAGMENT_RANK}_cluster', R0_map)
    R0_target = np.copy(R0_map)
    time = datetime.datetime.now()
    for iteration in range(fragmentation_iterations):
        print(f'iteration {iteration}')

        connecting_patch_indices, R0_target_fragmented = alpha_stepping_method(R0_target)
        connecting_patches[iteration] = connecting_patch_indices
        R0_target = update_fragmentation_target(R0_map, connecting_patch_indices)
        R0_target = R0_target * R0_map

    if plot:
        plt.title(f'Fragmented to {iteration+1} iterations')
        plot_fragmented_domain(connecting_patches, R0_map)

    print(f'Time taken to fragment {fragmentation_iterations} iterations: {datetime.datetime.now() - time}')
    return connecting_patches





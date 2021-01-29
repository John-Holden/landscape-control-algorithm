from domain_methods import coarse_grain, get_R0_map
from cluster_find import Cluster_sturct
from run_main import Landscape_structs
import numpy as np
from run_main import PATH_TO_DATA_STORE


def get_clusters_over_betas(ensemble_name:str, cluster_ranks:int, coarse_grain_level:int) -> np.ndarray:
    """
    For each value of beta, get top N ranked cluster size.
    """
    betas = np.load(f'{PATH_TO_DATA_STORE}/{ensemble_name}/betas.npy')
    ash_landscape = Landscape_structs(ensemble_name=ensemble_name, species='Fex')  # initialise domain structures
    species_distribution_map = coarse_grain(domain=ash_landscape.raw_data, cg_factor=coarse_grain_level)
    cluster_sizes = [None] * len(betas)
    for beta_index in range(len(betas)):
        R0_vs_rho = ash_landscape.R0_vs_rho_beta[beta_index]
        R0_map = get_R0_map(species_distribution_map=species_distribution_map, rhos=ash_landscape.rhos,
                            R0_v_rho_mapping=R0_vs_rho)

        ash_clusters = Cluster_sturct(R0_map=R0_map)
        top_cluster = ash_clusters.rank_cluster_sizes(rank_N=cluster_ranks)
        cluster_sizes[beta_index] = len(np.where(top_cluster)[0])


    return cluster_sizes


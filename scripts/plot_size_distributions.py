import numpy as np
from parameters_and_setup import EnsembleInfo, set_structuring_element, STRUCTURING_ELEMENT
from landscape_control.domain_processing import get_R0_map, rank_cluster_map
import landscape_control.plotting_methods as plt_lib


def get_size_distributions_vs_beta(ens_name: str, cg_factor: int, plot_raw_map: bool = False,
                                   plot_clusters: bool = False) -> np.ndarray:
    """
    Return
    :param ens_name:
    :param plot_raw_map:
    :param plot_clusters:
    :param ranks:
    :return:
    """
    ens = EnsembleInfo(ens_name)
    distributions = []
    non_trivial_betas = []
    for beta in [10]:
        print(f'beta {beta}')
        R0_vs_rho_for_beta = ens.R0_vs_rho_beta[beta]
        R0_map = get_R0_map(ens.raw_data, R0_vs_rho_for_beta, ens.rhos, coarse_grain_factor=cg_factor)
        if R0_map is None:
            print('R0 map is too small')
            continue

        if plot_raw_map:
            plt_lib.plot_R0_map(R0_map)

        R0_map, cluster_sizes, _ = rank_cluster_map(R0_map > 1)
        cluster_sizes = cluster_sizes * (cg_factor ** 2)
        # print(cluster_sizes[0])
        distributions.append(cluster_sizes)
        non_trivial_betas.append(ens.betas[beta])

        if plot_clusters:
            plt_lib.plot_R0_clusters(R0_map, rank=25)

    if distributions:
        plt_lib.plot_cluster_size_distribution(distributions, non_trivial_betas, save=True)


def get_size_distributions_vs_structure(ens_name: str, cg_factor: int, plot_raw_map: bool = False,
                                        plot_clusters: bool = False) -> np.ndarray:
    """
    Return
    :param ens_name:
    :param plot_raw_map:
    :param plot_clusters:
    :param ranks:
    :return:
    """
    ens = EnsembleInfo(ens_name)
    distributions = []
    non_trivial_betas = []
    for beta in [10]:
        print(f'beta {ens.betas[beta]}')
        for NN in ['VON-N', 'MOORE']:
            R0_vs_rho_for_beta = ens.R0_vs_rho_beta[beta]
            R0_map = get_R0_map(ens.raw_data, R0_vs_rho_for_beta, ens.rhos, coarse_grain_factor=cg_factor)
            if R0_map is None:
                print('R0 map is too small')
                continue

            if plot_raw_map:
                plt_lib.plot_R0_map(R0_map)

            R0_map, cluster_sizes, _ = rank_cluster_map(R0_map > 1, structure=NN)
            cluster_sizes = cluster_sizes * (cg_factor ** 2)
            distributions.append(cluster_sizes)
            non_trivial_betas.append(ens.betas[beta])

            if plot_clusters:
                plt_lib.plot_R0_clusters(R0_map, rank=25)

    if distributions:
        plt_lib.plot_cluster_size_distribution(distributions, non_trivial_betas, save=True)

if __name__ == '__main__':
    # get_size_distributions_vs_beta('landscape_control_package_2021-07-12_ga-phi2', cg_factor=1)
    get_size_distributions_vs_structure('landscape_control_package_2021-07-12_ga-phi2', cg_factor=1)

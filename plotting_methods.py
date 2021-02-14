import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from parameters_and_setup import PATH_TO_INPUT_DATA
from cluster_find import rank_cluster_map
from matplotlib.colors import LinearSegmentedColormap


pltParams = {'figure.figsize': (7.5, 5.5),
             'axes.labelsize': 15,
             'ytick.labelsize': 15,
             'xtick.labelsize': 15,
             'legend.fontsize': 'x-large'}
plt.rcParams.update(pltParams)


def plot_R0_vs_rho_over_ensemble(ensemble_name):
    'Plot R0 vs Rho for beta values'
    from domain_methods import linear_func
    from scipy.optimize import curve_fit

    path_to_ensemble = f'{PATH_TO_DATA_STORE}/{ensemble_name}'

    R0_vs_rho_ensemble = np.load(f'{path_to_ensemble}/ensemble.npy')
    rhos = np.load(f'{path_to_ensemble}/rhos.npy')
    betas = np.load(f'{path_to_ensemble}/betas.npy')
    for i, R0_vs_rho in enumerate(R0_vs_rho_ensemble):
        plt.plot(rhos, R0_vs_rho, label=f'Betas : {betas[i]}', c=f'C{i}')
        plt.scatter(rhos, R0_vs_rho, c=f'C{i}')
        popt, pcov = curve_fit(linear_func, rhos, R0_vs_rho)
        print(f'BETA: {betas[i]} | P out = {popt[0]}, Variance = {pcov[0]}')
        plt.plot(rhos, rhos * popt[0], ls='--',
                 label=f'fitted {round(popt[0])}', c=f'C{i}')

    plt.plot([rhos[0], rhos[-1]], [1, 1], c='r', ls='--')
    plt.show()
    
def plot_top_cluster_sizes_vs_beta(ensemble_name):
    """
    Plot how top cluster size varies with infectivity beta.
    """
    path_to_ensemble = f'{PATH_TO_DATA_STORE}/{ensemble_name}'
    cluster_sizes = np.load(f'{path_to_ensemble}/cluster_size_vs_beta.npy')
    print('cluster sizes', cluster_sizes)
    betas = np.load(f'{path_to_ensemble}/betas.npy')
    print([b for b in betas])
    fig, ax = plt.subplots()
    ax.plot(betas, cluster_sizes, label=f'Gaussian dispersal 1km')
    ax.plot([0.00014, 0.00014], [0, 2*10**5])
    ax.scatter(betas, cluster_sizes)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'Max cluster size $\mathrm{km}^2$')
    plt.legend()
    plt.show()
    return


def plot_cluster_size_vs_alpha(iteration:int, alpha_steps:Union[list, np.ndarray],
                               largest_cluster_size_vs_alpha: np.ndarray,
                               discontinuity_index:Union[int, None] = None):
    """
    Plot cluster sizes for one iteration and one value of alpha.
    """
    plt.title(f'cluster sizes & index | iteration {iteration}')
    plt.plot(alpha_steps, largest_cluster_size_vs_alpha)
    plt.scatter(alpha_steps, largest_cluster_size_vs_alpha)
    if discontinuity_index is not None:
        plt.plot([alpha_steps[discontinuity_index - 1], alpha_steps[discontinuity_index - 1]],
                 [0, largest_cluster_size_vs_alpha[discontinuity_index - 1]])
    plt.show()
    return


def plot_R0_clusters(R0_map:np.ndarray, rank: Union[None, int] = None):
    """
    Rank and plot clusters
    """
    if rank is not None and isinstance(rank, int):
        R0_map_background = np.array(R0_map > 0).astype(int)
        R0_map = rank_cluster_map(R0_map, get_ranks=rank)[0]
        assert len(np.unique(R0_map)) - 1 == rank, f'expected len {rank}, got {len(np.unique(R0_map)) - 1}'
        R0_map_background = np.array(R0_map > 0).astype(int) - R0_map_background

    cluster_number = len(np.unique(R0_map)) - 1
    colors = [f'C{i}' for i in range(len(np.unique(R0_map)) - 1)]

    if rank is not None and isinstance(rank, int):  # Plot back-ground as grey
        R0_map = R0_map + R0_map_background
        colors.insert(0, 'lightgrey')
        colors.insert(1, 'white')
        nbins = cluster_number+2
    else:
        colors.insert(0, 'white')
        nbins = cluster_number+1

    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
    im = plt.imshow(R0_map, cmap=cm)
    plt.colorbar(im)
    plt.show()
    return


def plot_fragmented_domain(connecting_patches:dict, R0_map:np.ndarray):
    """
    Plot the domain after it has been fragmented
    """
    connecting_patch_arr = np.zeros_like(R0_map)
    for index, indicies in connecting_patches.items():
        R0_map[indicies] = 0
        connecting_patch_arr[indicies] = 1

    plt.imshow(connecting_patch_arr)
    plt.show()

    R0_fragmented = rank_cluster_map(R0_map)[0]
    plot_R0_clusters(R0_fragmented)
    return






if __name__ == '__main__':
    plot_top_cluster_sizes_vs_beta(ensemble_name='landscape_control_input_test_data')
    plot_R0_vs_rho_over_ensemble(ensemble_name='landscape_control_input_test_data')
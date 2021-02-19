import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Iterable
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

    path_to_ensemble = f'{PATH_TO_INPUT_DATA}/{ensemble_name}'

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
    path_to_ensemble = f'{PATH_TO_INPUT_DATA}/{ensemble_name}'
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


def plot_R0_clusters(R0_map:np.ndarray, rank: Union[None, int] = None, epi_c:Union[None, tuple] = None):
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

    if epi_c is not None:
        circle = plt.Circle((epi_c[1], epi_c[0]), 1.5, fc='black', ec="red")
        plt.gca().add_patch(circle)

    plt.show()
    return


def plot_fragmented_domain(fragmented_domain:np.ndarray, R0_map:np.ndarray, epi_c:Union[None, tuple] = None,
                           show_text:bool=False):
    """
    Plot the domain after it has been fragmented
    """
    frag_number = np.unique(fragmented_domain)
    frag_number = frag_number[1:] if 0 in frag_number else frag_number
    colors = [f'C{i}' for i, line in enumerate(frag_number)]
    colors.insert(0, 'white')
    colors.insert(1, 'lightgrey')

    nbins = len(colors)
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)

    # Color fragmentation lines from C0
    show_text_dict = {} if show_text else None
    fragmented_domain_ = np.zeros_like(fragmented_domain)
    for i, frag_line in enumerate(frag_number):
        fragmented_domain_[np.where(fragmented_domain == frag_line)] = i+2
        if show_text:
            show_text_dict[frag_line] = i+2

    # Include background of R0 map having numerical value 1 shown as light-grey
    R0_map[np.where(fragmented_domain)] = 0
    fragmented_domain_ += np.where(R0_map > 1, 1, 0)

    im = plt.imshow(fragmented_domain_, cmap=cm)
    plt.colorbar(im)

    # Optional, show epicenter
    if epi_c is not None:
        circle = plt.Circle((epi_c[1], epi_c[0]), 1.5, fc='black', ec="red")
        plt.gca().add_patch(circle)

    if show_text:
        print(show_text_dict)
        for frag_line, numerical_val in show_text_dict.items():
            line_ind = np.where(fragmented_domain_ == numerical_val)
            N_points = len(line_ind[0])
            x, y = line_ind[0].sum()/N_points, line_ind[1].sum()/N_points
            plt.text(y, x, f'{frag_line}', c='b', size=10)

    plt.show()
    return


def plot_payoff_efficiencies(payoff_store: dict):
    """
    Plot payoff found from scenario test.
    """
    N_saved = []
    N_culled = []
    for epic, payoffs in payoff_store.items():
        # Iterate through epicenters
        for comb, result in payoffs.items():
            # Iterate through each result in epicenters
            N_saved.append(result['Ns'])
            N_culled.append(result['Nc'])

    print('len ns', len(N_saved))
    print('len unique elements ', len(np.unique(N_saved)))

    plt.title('payoff1')
    plt.scatter(N_culled, N_saved)
    plt.xlabel('N culled')
    plt.ylabel('N saved')
    plt.show()

    plt.title('payoff2')
    N_saved, N_culled = np.array(N_saved), np.array(N_culled)
    plt.scatter(range(len(N_culled)), np.sort(N_saved/N_culled))
    plt.xlabel('rank')
    plt.ylabel('Ns/Nc')
    plt.show()
    return





if __name__ == '__main__':
    plot_top_cluster_sizes_vs_beta(ensemble_name='landscape_control_input_test_data')
    plot_R0_vs_rho_over_ensemble(ensemble_name='landscape_control_input_test_data')
import matplotlib.pyplot as plt
import numpy as np

pltParams = {'figure.figsize': (7.5, 5.5),
             'axes.labelsize': 15,
             'ytick.labelsize': 15,
             'xtick.labelsize': 15,
             'legend.fontsize': 'x-large'}
plt.rcParams.update(pltParams)


def plot_top_cluster_sizes_vs_beta(path_to_ensemble):
    """
    Plot how top cluster size varies with infectivity beta.
    """
    cluster_sizes = np.load(f'{path_to_ensemble}/cluster_size_vs_beta.npy')
    betas = np.load(f'{path_to_ensemble}/betas.npy')
    map_resolution = 5 # that is, 5km^2 x 5km^2
    cluster_sizes = cluster_sizes * map_resolution**2
    fig, ax = plt.subplots()
    ax.plot(betas, cluster_sizes, label=f'Gaussian dispersal 1km')
    ax.scatter(betas, cluster_sizes)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'Max cluster size $\mathrm{km}^2$')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    from run_main import PATH_TO_DATA_STORE
    ensemble_name = 'landscape_control_input_test_data'
    plot_top_cluster_sizes_vs_beta(path_to_ensemble=f'{PATH_TO_DATA_STORE}/{ensemble_name}')
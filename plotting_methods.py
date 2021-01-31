import matplotlib.pyplot as plt
import numpy as np
from run_main import PATH_TO_DATA_STORE


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


if __name__ == '__main__':
    plot_top_cluster_sizes_vs_beta(ensemble_name='landscape_control_input_test_data')
    plot_R0_vs_rho_over_ensemble(ensemble_name='landscape_control_input_test_data')
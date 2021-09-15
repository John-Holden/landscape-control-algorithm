import os
import pickle5 as pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, List

from parameters_and_setup import EnsembleInfo, PATH_TO_INPUT_DATA
from landscape_control import ClusterFrag, ScenarioTest

from landscape_control.domain_processing import get_clusters_over_betas
from landscape_control.plotting_methods import process_payoffs, plot_payoffs_over_beta, plot_cluster_sizes_vs_beta


def get_efficiency_over_beta(package_name: str, sample_size: int = 5, save: Optional[bool] = False,
                             plot: Optional[bool] = True, beta_indices: Optional[list] = None):
    """
    Iterate over the different payoff data entries, for different beta values, and find the top ranked payoff
    (up-to rank N='sample-size'), plot the outcome.

    :param package_name:
    :param sample_size:
    :param save:
    :param plot:
    :param beta_indices:
    :return:
    """
    ensemble = EnsembleInfo(package_name)
    path = f'{PATH_TO_INPUT_DATA}/{package_name}/fragmentation_payoff_data'
    beta_indices = [i for i in range(len(ensemble.betas))] if beta_indices is None else beta_indices
    payoff = np.zeros((len(ensemble.betas), sample_size))
    iterations = 'auto'
    cg = 5
    for i in beta_indices:
        print(f'loading beta index: {i}')
        if not os.path.exists(f'{path}/Fex_cg_{cg}_beta_{i}_iterations_{iterations}.pickle'):
            print(f'path : {path}/Fex_cg_{cg}_beta_{i}_iterations_{iterations}.pickle does not exist!')
            plt.scatter([ensemble.betas[i]], [0], marker='x')
            plt.plot([ensemble.betas[i]], [0])
            continue

        with open(f'{path}/Fex_cg_{cg}_beta_{i}_iterations_{iterations}.pickle', 'rb') as f:
            beta_payoff = pickle.load(f)
            beta_payoff = process_payoffs(beta_payoff)[0]
            beta_payoff = beta_payoff[-sample_size:]
            print(f' {i} payoff :  {beta_payoff},  len : {len(np.unique(beta_payoff))} ')
            if len(np.unique(beta_payoff)) < len(beta_payoff) or len(beta_payoff) < sample_size:
                msg = f'\n Correct/double check : Beta index {i}'
                warnings.warn(msg)

            assert len(beta_payoff) == sample_size, fr'found {payoff} | len {len(payoff)}'
            payoff[i] = beta_payoff

    if save:
        np.save('fragmentation_payoff_over_beta', payoff)

    if plot:
        plot_payoffs_over_beta(payoff, ensemble.betas, save=save)


def run_fragmentation_over_beta(package_name: str):
    ensemble = EnsembleInfo(package_name)
    beta_inds = [20]
    cg_factor = 5
    iters = [16] * len(beta_inds)
    for i, beta_ind in enumerate(beta_inds):
        print(f'Running beta {round(ensemble.betas[beta_ind], 5)}, for {iters[i]} iterations ')
        c_frag = ClusterFrag(ensemble, cg_factor=cg_factor, beta_index=beta_ind, iterations=iters[i])
        result = c_frag.execute(plot=True)
        print(f'success : {result} ')


def get_plot_cluster_size_vs_fragmentation(package_name: str):
    """
    Load, and plot, the cluster size decarease with each fragmentation iteration
    Figures 1-2 ch 7.
    :param package_name:
    :return:
    """
    import json
    ensemble = EnsembleInfo(package_name)
    iters = 25
    beta_ind = [5, 8, 10, 13, 16]
    cg_factor = 3
    cluster_sizes = np.zeros(shape=(len(beta_ind), iters))
    connected_patch_area = np.zeros_like(cluster_sizes)

    iterations = []
    number_of_patches = []
    beta_values = []

    for i, beta in enumerate(beta_ind):
        if not os.path.exists(f'{ensemble.path2_culled_indices}'):
            raise Exception(f'beta: {beta} not found -- re-run fragmentation!')

        filename = f'{ensemble.path2_culled_indices}/Fex_cg_{cg_factor}_beta_{beta}_iterations_{iters}.json'
        with open(filename, 'r') as fragmentation_data:
            data = json.load(fragmentation_data)

        for iteration in range(iters):
            cluster_sizes[i][iteration] = data[f'{iteration}_size']
            connected_patch_area[i][iteration] = len(data[f'{iteration}'][0])

        iterations.extend(np.arange(1, iters+1, 1))
        number_of_patches.extend(connected_patch_area[i])
        beta_values.extend(np.ones(iters) * ensemble.betas[beta])

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy.optimize import curve_fit

    def inverser_power_law(xdata, c, exponent):
        return c * xdata ** (-exponent)

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, beta in enumerate(beta_ind):
        xdata = np.arange(1, len(cluster_sizes[i])+1, 1)
        pout, pcov = curve_fit(inverser_power_law, xdata=xdata, ydata=cluster_sizes[i])
        ax.scatter(xdata, cluster_sizes[i])
        ax.plot(xdata, cluster_sizes[i], label=f'beta = {ensemble.betas[beta]} a = {round(pout[0], 3)}, k = {round(pout[1], 3)}')
        ax.plot(xdata, inverser_power_law(xdata, pout[0], pout[1]), c=f'C{i}', ls='--')

    # plt.legend()
    # plt.yscale('log')
    # plt.xscale('log')
    plt.grid(False)
    plt.tick_params(labelsize=20)
    plt.savefig('fragmentation-size-decrease.pdf')
    plt.show()

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    data = pd.DataFrame({'betas': beta_values, 'iteration': iterations, 'number_of_patches': number_of_patches})
    sns.barplot(data=data, x='iteration', y='number_of_patches', hue='betas', ax=ax)
    plt.xlim(0, 14)
    plt.tick_params(labelsize=20)
    plt.legend([], [], frameon=False)
    plt.savefig('connecting-patch-area.pdf')
    plt.show()

    assert 0


def run_scenario_test_over_beta(package_name: str, job_id: Union[None, str] = None):
    """
    Find all payoff-combinations for beta-epicenter scenarios
    :param package_name:
    :param job_id: optional, hpc job id
    :return:
    """

    if job_id:
        beta_index = int(job_id)-1
        scenario_test = ScenarioTest(package_name, beta_index)
        if not scenario_test.is_valid:
            print(f'\t skipping beta index {beta_index}')
            return
        print(f'Running beta index {beta_index}')
        scenario_test.find_all_payoffs(plot_check=False, verbose=True)
        return

    for beta_index in [1]:
        print(f'Running beta index {beta_index}')
        scenario_test = ScenarioTest(package_name, beta_index, cg_factor=5, iterations=10)
        if not scenario_test.is_valid:
            print(f'skipping beta index {beta_index}')
            continue

        scenario_test.find_all_payoffs(plot_check=False)


def cluster_size_over_beta(package_name: str, cg: int, save_fig: Optional[bool]=False):
    from parameters_and_setup import STRUCTURING_ELEMENT
    print(f'Structuring element \n {STRUCTURING_ELEMENT}')

    ensemble = EnsembleInfo(package_name)
    cluster_sizes = get_clusters_over_betas(ensemble, plot_clusters=False, plot_R0_maps=False, cg_factor=cg, get_rank=5)
    np.save(f'{ensemble.path_to_ensemble}/cluster_sizes-Moore-{cg}km', cluster_sizes)
    plot_cluster_sizes_vs_beta(cluster_sizes=cluster_sizes, betas=ensemble.betas, save=save_fig)


def comp_cluster_sizes(package_names: List[str]):
    from landscape_control.plotting_methods import plot_cluster_size_comparison_over_beta
    cluster_dat, beta_dat = [], []
    for name in package_names:
        ens = EnsembleInfo(name)
        cluster_dat.append(np.load(f'{ens.path_to_ensemble}/cluster_sizes.npy'))
        beta_dat.append(ens.betas)

    plot_cluster_size_comparison_over_beta(cluster_dat, beta_dat, save=True)


def plot_multi_cluster_size_over_beta(ensemble: list):
    """

    :param ens1:
    :param ens2:
    :return:
    """
    clusters = np.zeros(shape=(6, 21))
    c = 0
    for i in range(2):
        name = 'Moore' if i == 0 else 'vonN'
        for sz in [5, 3, 1]:
            clusters[c] = np.load(f'{PATH_TO_INPUT_DATA}/{ensemble[0]}/cluster_sizes-{name}-{sz}km.npy')[:, 0]
            c += 1

    # clusters1 = [np.load(f'{PATH_TO_INPUT_DATA}/{ensemble[0]}/cluster_sizes-Moore-{i}km.npy') for i in (5, 3, 1)]
    # clusters2 = [np.load(f'{PATH_TO_INPUT_DATA}/{ensemble[0]}/cluster_sizes-vonN-{i}km.npy') for i in (5, 3, 1)]
    # clusters = [np.load(f'{PATH_TO_INPUT_DATA}/{ensemble[0]}/{name}') for name in ['cluster_sizes-Moore-5km.npy', 'cluster_sizes-vonN-5km.npy']]

    betas = np.load(f'{PATH_TO_INPUT_DATA}/{ensemble[0]}/betas.npy')
    plot_cluster_sizes_vs_beta(cluster_sizes=clusters, betas=betas, save=True)


if __name__ == '__main__':
    # get_plot_cluster_size_vs_fragmentation('landscape_control_package_2021-07-10_ga-phi1')
    # run_fragmentation_over_beta('landscape_control_package_2021-07-10_ga-phi1')
    # run_scenario_test_over_beta('landscape_control_package_2021-07-10_ga-phi1')
    get_efficiency_over_beta('landscape_control_package_2021-07-10_ga-phi1', save=False)
    # cluster_size_over_beta('landscape_control_package_2021-07-12_ga-phi2', cg=2)
    # plot_multi_cluster_size_over_beta(['landscape_control_package_2021-07-10_ga-phi1'])

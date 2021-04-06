import os
import numpy as np
import pickle
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
    avg_payoff = []
    beta_indices = [i for i in range(len(ensemble.betas))] if beta_indices is None else beta_indices
    payoff = np.zeros((len(ensemble.betas), sample_size))

    for i in beta_indices:
        print(f'loading {i}')
        if not os.path.exists(f'{path}/Fex_cg_5_beta_{i}_iterations_auto.pickle'):
            print(f'path : {path}/Fex_cg_5_beta_{i}_iterations_auto.pickle does not exist!')
            plt.scatter([ensemble.betas[i]], [0], marker='x')
            plt.plot([ensemble.betas[i]], [0])
            avg_payoff.append(0)
            continue

        with open(f'{path}/Fex_cg_5_beta_{i}_iterations_auto.pickle', 'rb') as f:
            beta_payoff = pickle.load(f)
            beta_payoff = process_payoffs(beta_payoff)[0]
            beta_payoff = beta_payoff[-5:]
            print(beta_payoff, 'len ', len(np.unique(beta_payoff)) )
            assert len(beta_payoff) == 5, fr'found {payoff} | len {len(payoff)}'
            payoff[i] = beta_payoff

    if save:
        np.save('fragmentation_payoff_over_beta', payoff)

    if plot:
        plot_payoffs_over_beta(payoff, ensemble.betas, save=save)



def run_fragmentation_over_beta(package_name: str):
    ensemble = EnsembleInfo(package_name)
    beta_ind = 12
    iters = 20
    print(f'Running beta {round(ensemble.betas[beta_ind], 5)}, for {iters} iterations ')
    c_frag = ClusterFrag(ensemble, cg_factor=5, beta_index=beta_ind, iterations=iters)
    result = c_frag.execute(plot=True)
    print(f'success : {result} ')


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

        scenario_test.find_all_payoffs(plot_check=False)
        return

    for beta_index in [1,2,3,4,5]:
        scenario_test = ScenarioTest(package_name, beta_index)
        if not scenario_test.is_valid:
            print(f'skipping beta index {beta_index}')
            continue

        scenario_test.find_all_payoffs(plot_check=False)


def cluster_size_over_beta(package_name: str, savefig: Optional[bool]=False):
    ensemble = EnsembleInfo(package_name)
    cluster_sizes = get_clusters_over_betas(ensemble, plot_clusters=False)
    np.save(f'{ensemble.path_to_ensemble}/cluster_sizes', cluster_sizes)

    plot_cluster_sizes_vs_beta(ensemble.betas, cluster_sizes, save=savefig)



def comp_cluster_sizes(package_names: List[str]):
    from landscape_control.plotting_methods import plot_cluster_size_comparison_over_beta
    cluster_dat, beta_dat = [], []
    for name in package_names:
        ens = EnsembleInfo(name)
        cluster_dat.append(np.load(f'{ens.path_to_ensemble}/cluster_sizes.npy'))
        beta_dat.append(ens.betas)

    plot_cluster_size_comparison_over_beta(cluster_dat, beta_dat, save=True)



if __name__ == '__main__':
    # run_fragmentation_over_beta('landscape_control_package_adb_pl')
    run_scenario_test_over_beta('landscape_control_package_adb_pl', job_id=13)
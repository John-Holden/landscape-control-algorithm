from landscape_control.domain_processing import get_R0_map, process_R0_map
from landscape_control.plotting_methods import plot_spatial_payoff_rank
from parameters_and_setup import EnsembleInfo
import pickle
import os
import re
from typing import Optional


def open_payoff(path: str, beta_index):
    payoff_dat_name = [f for f in os.listdir(path)
                       if re.match(rf'^Fex_cg_5_beta_{beta_index}.*', f)]

    if not payoff_dat_name:
        print('trivial payoff dat')

    with open(f'{path}/{payoff_dat_name[0]}', 'rb') as f:
        payoff_dat = pickle.load(f)

    return payoff_dat, payoff_dat_name[0]

def plot_R0_map(ens:str, beta_index:int, rank:Optional=None, plot:bool=False, cg_factor:int = 5):
    from landscape_control.plotting_methods import plot_R0_clusters, plot_R0_map
    print(f'beta = {ens.betas[beta_index]}')
    R0_map = get_R0_map(ens.raw_data, ens.R0_vs_rho_beta[beta_index], ens.rhos, cg_factor)
    if plot:
        import numpy as np
        plot_R0_clusters(np.where(R0_map < 1, 0, R0_map), rank=rank, save=True, ext='pdf', cg_factor=cg_factor)
        plot_R0_map(R0_map, save=True)

    assert 0
    R0_map = process_R0_map(R0_map, get_cluster=1)
    return R0_map

def plot_spatial_rank(package_name, beta_index, rank, save: Optional[bool] = False):
    ens = EnsembleInfo(package_name)
    R0_map = plot_R0_map(ens, beta_index, plot=False)
    if R0_map is None:
        print('Trivial map')
        return

    payoff_dat = open_payoff(ens.path2_payoff_data, beta_index)[0]

    msg = rf'$\beta$ = {round(ens.betas[beta_index], 6)}, rank = {rank}'
    plot_spatial_payoff_rank(R0_map, payoff_dat, rank, title=msg, save=save)


def add_flag_to_payoff(package_name: str, beta_index: int, rank: int, flag: dict):
    # add a flag to the pay off data, either skip or ...?
    # usage:
    # add_flag_to_payoff('landscape_control_package_adb_pl_2', beta_index=12, rank=5, flag={'skip_flag': True})
    ens = EnsembleInfo(package_name)
    payoff_dat, payoff_dat_name = open_payoff(ens.path2_payoff_data, beta_index)
    for epic, payoffs in payoff_dat.items():
        # Iterate through epicenters
        for comb, result in payoffs.items():
            # Iterate through each result in comb
            if result is None:
                continue

            if result['rank'] == rank:
                for key, value in flag.items():
                    result[key] = value

                with open(f'{ens.path2_payoff_data}/{payoff_dat_name}', 'wb') as handle:
                    pickle.dump(payoff_dat, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    return


if __name__ == '__main__':
    ens = EnsembleInfo('landscape_control_package_2021-07-10_ga-phi1')
    plot_R0_map(ens, beta_index=10, rank=1, plot=True, cg_factor=1)

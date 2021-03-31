from landscape_control.domain_processing import get_R0_map, process_R0_map
from landscape_control.plotting_methods import plot_spatial_payoff_rank
from parameters_and_setup import EnsembleInfo
import pickle
import os
import re


def open_payoff(path: str, beta_index):
    payoff_dat_name = [f for f in os.listdir(path)
                       if re.match(rf'^Fex_cg_5_beta_{beta_index}.*', f)]

    if not payoff_dat_name:
        print('trivial payoff dat')

    with open(f'{path}/{payoff_dat_name[0]}', 'rb') as f:
        payoff_dat = pickle.load(f)

    return payoff_dat, payoff_dat_name[0]


def plot_spatial_rank(package_name, beta_index, rank):
    ens = EnsembleInfo(package_name)
    R0_map = get_R0_map(ens.raw_data, ens.R0_vs_rho_beta[beta_index], ens.rhos, coarse_grain_factor=5)
    R0_map = process_R0_map(R0_map, get_cluster=1)

    if R0_map is None:
        print('Trivial map')
        return

    payoff_dat = open_payoff(ens.path2_payoff_data, beta_index)[0]

    msg = rf'$\beta$ = {round(ens.betas[beta_index], 6)}, rank = {rank}'
    plot_spatial_payoff_rank(R0_map, payoff_dat, rank, title=msg)


def add_flag_to_payoff(package_name: str, beta_index: int, rank: int, flag: dict):
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
    # plot_spatial_rank('landscape_control_package_adb_full', beta_index=9,  rank=2)
    add_flag_to_payoff('landscape_control_package_adb_full', beta_index=17, rank=2, flag={'skip_flag': True})

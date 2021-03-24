import numpy as np
from parameters_and_setup import EnsembleInfo
from landscape_control import ClusterFrag, ScenarioTest
from landscape_control.plotting_methods import plot_payoff_efficiencies_1, plot_spatial_payoff_rank

from landscape_control.domain_processing import rank_cluster_map, plot_R0_clusters


def run_fragmentation_over_beta(package_name: str):

    ensemble = EnsembleInfo(package_name)

    c_frag = ClusterFrag(ensemble, cg_factor=5, beta_index=3, iterations=20)
    result = c_frag.execute(plot=True)
    print(f'success : {result} ')


def run_scenari
    o_test_over_beta(package_name: str):

    for beta_index in [9]:
        scenario_test = ScenarioTest(package_name, beta_index)
        if not scenario_test.is_valid:
            print(f'skipping beta index {beta_index}')
            continue

        payoffs, num = scenario_test.find_all_payoffs(plot_check=True)
        plot_payoff_efficiencies_1(payoffs)
        plot_spatial_payoff_rank(scenario_test.R0_domain, payoffs, rank=1)


if __name__ == '__main__':
    # run_fragmentation_over_beta('landscape_control_package_adb_full')
    run_scenario_test_over_beta('landscape_control_package_adb_full')
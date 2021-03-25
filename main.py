import os
import sys
from landscape_control import ClusterFrag, ScenarioTest
from parameters_and_setup import EnsembleInfo

from landscape_control.plotting_methods import plot_payoff_efficiencies_1, append_payoffs
from scripts.over_beta_processing import run_scenario_test_over_beta

def run_fragmentation():
    ensemble = EnsembleInfo('landscape_control_package')
    cfrag = ClusterFrag(ensemble, cg_factor=5, beta_index=1, iterations=10)
    cfrag.execute(plot=True)


def run_scenario_test():
    scenario_test = ScenarioTest('landscape_control_package', beta_index=1, iterations=10)
    payoffs, num = scenario_test.find_all_payoffs(plot_check=False)
    plot_payoff_efficiencies_1(payoffs)


def load_and_plot_scenario():
    import pickle, os
    path2_dat = os.getcwd()
    path2_dat += '/data_store/landscape_control_package/fragmentation_payoff_data/Fex_cg_5_beta_1_iterations_10.pickle'
    file = open(f"{path2_dat}", 'rb')
    payoff_dat = pickle.load(file)
    plot_payoff_efficiencies_1(payoff_dat)


if __name__ == '__main__':
    if 'HPC_MODE' in os.environ:
        assert os.environ['HPC_MODE'] == 'TRUE'
        assert len(sys.argv) == 2 and sys.argv[1].isdigit()
        run_scenario_test_over_beta('landscape_control_package_adb_full', sys.argv[1])

    else:
        run_scenario_test_over_beta('landscape_control_package_adb_full')
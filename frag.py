from landscape_control import ClusterFrag, Scenario_test
from parameters_and_setup import Ensemble_info

from landscape_control._plotting_methods import plot_payoff_efficiencies


def run_fragmentation():
    ensemble = Ensemble_info('landscape_control_package')
    cfrag = ClusterFrag(ensemble, cg_factor=5, beta_index=3, iterations=10)
    cfrag.execute()

def run_scenario_test():
    scenario_test = Scenario_test('landscape_control_package', beta_index=3, iterations=10)
    payoffs, num = scenario_test.find_all_payoffs(epi_center_number=100, plot_check=False)
    plot_payoff_efficiencies(payoffs)

if __name__ == '__main__':
    run_scenario_test()

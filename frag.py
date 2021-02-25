from landscape_control import ClusterFrag, Scenario_test
from parameters_and_setup import Ensemble_info

from landscape_control._plotting_methods import plot_payoff_efficiencies_1, append_payoffs


def run_fragmentation():
    ensemble = Ensemble_info('landscape_control_package')
    cfrag = ClusterFrag(ensemble, cg_factor=5, beta_index=1, iterations=10)
    cfrag.execute(plot=True)

def run_scenario_test():
    scenario_test = Scenario_test('landscape_control_package', beta_index=3, iterations=10)
    payoffs, num = scenario_test.find_all_payoffs(epi_center_number=100, plot_check=False)
    plot_payoff_efficiencies_1(payoffs)


def load_and_plot_scenario():
    import pickle
    path2_dat = f'./data_store/landscape_control_package/fragmentation_payoff_data/fex_cg_5_beta_1_iterations_10.pickle'
    file = open(f"{path2_dat}", 'rb')
    payoff_dat = pickle.load(file)
    # plot_payoff_efficiencies_1(payoff_dat )
    payoff, N_saved, N_culled, epi_centers, combinations = append_payoffs(payoff_dat, return_top=1)
    print(combinations)

if __name__ == '__main__':
    run_scenario_test()

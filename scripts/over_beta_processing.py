import os
import pickle
from typing import Union
import matplotlib.pyplot as plt
from parameters_and_setup import EnsembleInfo, PATH_TO_INPUT_DATA
from landscape_control import ClusterFrag, ScenarioTest
from landscape_control.plotting_methods import plot_payoff_efficiencies_1, process_payoffs


def get_efficiency_over_beta(package_name: str):
    ensemble = EnsembleInfo(package_name)
    path = f'{PATH_TO_INPUT_DATA}/{package_name}/fragmentation_payoff_data'
    avg_payoff = []
    for i in range(0, 15):
        print(f'loading {i}')
        if not os.path.exists(f'{path}/Fex_cg_5_beta_{i}_iterations_auto.pickle'):
            print(f'path : {path}/Fex_cg_5_beta_{i}_iterations_auto.pickle does not exist!')
            plt.scatter([ensemble.betas[i]], [0], marker='x')
            plt.plot([ensemble.betas[i]], [0])
            avg_payoff.append(0)
            continue

        with open(f'{path}/Fex_cg_5_beta_{i}_iterations_auto.pickle', 'rb') as f:
            beta_payoff = pickle.load(f)
            payoff = process_payoffs(beta_payoff)[0]
            payoff = payoff[-5:]

        payoff = list(payoff)
        avg_payoff.append(sum(payoff)/len(payoff))
        xdata = [ensemble.betas[i]] * len(payoff)
        plt.scatter(xdata, payoff, marker='x')
        xdata.append(xdata[0]), payoff.append(0)
        plt.plot(xdata, payoff)

    plt.xlim(-0.00001, ensemble.betas[15])
    plt.show()

    plt.savefig('payoff_over_beta.pdf')



def run_fragmentation_over_beta(package_name: str):
    ensemble = EnsembleInfo(package_name)
    c_frag = ClusterFrag(ensemble, cg_factor=5, beta_index=3, iterations=20)
    result = c_frag.execute(plot=True)
    print(f'success : {result} ')


def run_scenario_test_over_beta(package_name: str, job_id: Union[None, str]):

    if job_id:
        beta_index = int(job_id)-1
        scenario_test = ScenarioTest(package_name, beta_index)
        if not scenario_test.is_valid:
            print(f'\t skipping beta index {beta_index}')
            return

        scenario_test.find_all_payoffs(plot_check=False)
        return

    for beta_index in range(1, 5):
        scenario_test = ScenarioTest(package_name, beta_index)
        if not scenario_test.is_valid:
            print(f'skipping beta index {beta_index}')
            continue

        scenario_test.find_all_payoffs(plot_check=False)


if __name__ == '__main__':
    # run_fragmentation_over_beta('landscape_control_package_adb_full')
    # run_scenario_test_over_beta('landscape_control_package_adb_full')
    get_efficiency_over_beta('landscape_control_package_adb_full')
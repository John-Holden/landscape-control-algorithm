from typing import Union
from parameters_and_setup import EnsembleInfo
from landscape_control import ClusterFrag, ScenarioTest


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
    run_scenario_test_over_beta('landscape_control_package_adb_full')
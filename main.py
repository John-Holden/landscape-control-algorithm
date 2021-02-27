import numpy as np
from typing import Type, Union

from parameters_and_setup import EnsembleInfo
from landscape_control import ClusterFrag, ScenarioTest




def run_fragmentation(ensemble_name:str, cg_factor:int, beta:int, iterations:int):
    """
    Load ensemble and run cluster fragmentation algorithm.
    """
    ensemble = EnsembleInfo(ensemble_name)
    cluster_frag = ClusterFrag(ensemble, cg_factor, beta, iterations)
    cluster_frag.execute(plot=True)


def run_scenario_test(ensemble_name:str, beta:int, iterations:int):
    """
    For a given cluster-fragmentation, scenario-test the containment and return payoff
    """
    scenario_test = ScenarioTest(ensemble_name, beta, iterations)
    scenario_test.find_all_payoffs(plot_check=False)


def load_scenario_payoff(ensemble: Type[EnsembleInfo], beta:int, iterations:int) -> dict:
    """
    Load in and return scenario-payoff data for a given ensemble
    """
    import pickle
    path2_dat = ensemble.path_to_ensemble
    path2_dat += f'/fragmentation_payoff_data/Fex_cg_5_beta_{beta}_iterations_{iterations}.pickle'
    file = open(f"{path2_dat}", 'rb')
    return pickle.load(file)


def load_processed_R0_domain(ensemble: Type[EnsembleInfo], beta:int) -> np.ndarray:
    """
    Load in and return the processed R0-domain for a given ensemble -- here `processed' means the cluster-target
    which underwent fragmentation, by default, the target is the largest-ranked cluster of susceptible trees in
    the population.
    """
    path2_dat = ensemble.path_to_ensemble
    path2_dat += f'/processed_R0_maps/Fex_cg_5_beta_{beta}_processed_R0_map.npy'
    return np.load(path2_dat)


def load_fragmented_domain(ensemble: Type[EnsembleInfo], beta:int) -> np.ndarray :
    """
    Load in and return the fragmented domain -- here, `fragmented_domain' means a map of culling targets ie. connector
    patches that if culled fragment the R0 cluster.
    """
    path2_dat = ensemble.path_to_ensemble
    path2_dat += f'/fragmented_R0_domain/Fex_cg_5_beta_{beta}_fragmented_domain.npy'
    return np.load(path2_dat)


def load_scenario_data(beta:int, iterations:int, ensemble_name:str) -> Union[dict, np.ndarray]:
    ensemble = EnsembleInfo(ensemble_name)
    payoff = load_scenario_payoff(ensemble, beta, iterations)
    processed_R0_domain = load_processed_R0_domain(ensemble, beta)
    return payoff, processed_R0_domain


if __name__ == '__main__':
    # run_fragmentation()
    # run_scenario_test()
    # payoff, processed_R0_domain = load_scenario_data(beta=2, iterations=10, ensemble_name='landscape_control_package')
    # plot_spatial_payoff_rank(processed_R0_domain, payoff, rank=100)
    from landscape_control.domain_processing import get_clusters_over_betas
    ensemble = EnsembleInfo('landscape_control_package')
    get_clusters_over_betas(ensemble, plot_output=True, plot_clusters=True)





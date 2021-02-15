import sys
import json
import numpy as np
from typing import Union, Iterable
import matplotlib.pyplot as plt

from cluster_find import rank_cluster_map
from plotting_methods import plot_R0_clusters
from parameters_and_setup import PATH_TO_INPUT_DATA, ENSEMBLES


class Scenario_test():
    # loa
    def __init__(self, ensemble_name:str, beta_index:int, cg_factor:int=5, species:str = 'fex', iterations:int=10):
        path2_ensemble = f'{PATH_TO_INPUT_DATA}/{ensemble_name}'
        path2_scenario = f'{species}_cg_{cg_factor}_beta_{beta_index}'
        path2_patch_data = f'connecting_patch_data/{path2_scenario}_iterations_{iterations}.json'
        path2_processed_R0_domain = f'processed_R0_domains/{path2_scenario}_processed_R0_domain.npy'
        try:
            self.R0_domain = np.load(f'{path2_ensemble}/{path2_processed_R0_domain}')  # load domain
            with open(f'{path2_ensemble}/{path2_patch_data}', 'r') as infile:  # load fragmentation data
                connecting_patches = json.load(infile)

        except Exception as exc:
            sys.exit(f'Error, file(s) not found. Have you run the fragmentation algorithm ? \n {exc}')

        self.connecting_patches = {}
        for iteration, indices in connecting_patches.items():
            self.connecting_patches[int(iteration)] = (tuple(indices[0]), tuple(indices[1]))

        self.iterations = iterations
        self.species = species
        self.cg_factor = cg_factor
        self.beta_index = beta_index

    def domain_at_iteration(self, iterations: Union[int, Iterable]):
        # Find fragmented domain @ given iteration

        if isinstance(iterations, int):
            iterations = [iterations]

        if max(iterations) > self.iterations:
            print(f'Error, {iterations} too high, max iteration in dataset : {self.iterations}')
            return None

        R0_fragmented = np.copy(self.R0_domain)
        for iteration in iterations:
            R0_fragmented[self.connecting_patches[iteration - 1]] = 0

        return R0_fragmented


if __name__ == '__main__':

    iter_ = 5
    scenario_test = Scenario_test(ENSEMBLES[2], beta_index=2, iterations=5)


    R0_domain = scenario_test.domain_at_iteration(iterations=iter_)

    plt.title(f'R0 at iteration {iter_}')
    plot_R0_clusters(rank_cluster_map(R0_domain)[0])
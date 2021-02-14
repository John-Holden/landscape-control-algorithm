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
        path_to_ensemble = f'{PATH_TO_INPUT_DATA}/{ensemble_name}'
        path2_frag_data = f'beta_index_{beta_index}_iterations_{iterations}_cg_{cg_factor}.json'
        try:
            self.R0_domain = np.load(f'{path_to_ensemble}/{species}_R0_map_rank_1_cluster.npy')
            with open(f'{path_to_ensemble}/fragmentation_data/{path2_frag_data}', 'r') as infile:
                connecting_patches = json.load(infile)
        except FileNotFoundError:
            sys.exit('Error, file(s) not found. Run fragmentation algorithm.')

        self.connecting_patches = {}
        for iteration, indices in connecting_patches.items():
            self.connecting_patches[int(iteration)] = (tuple(indices[0]), tuple(indices[1]))

        print(self.connecting_patches)
        self.iterations = 10
        self.species = species
        self.cg_factor = cg_factor
        self.beta_index = beta_index

    def domain_at_iteration(self, iterations: Union[int, Iterable]):
        # Find fragmented domain @ given iteration
        if isinstance(iterations, int):
            iterations = [iterations]

        R0_fragmented = np.copy(self.R0_domain)
        for iteration in iterations:
            R0_fragmented[self.connecting_patches[iteration]] = 0

        return R0_fragmented


if __name__ == '__main__':

    scenario_test = Scenario_test(ENSEMBLES[2], beta_index=1)
    plt.title('R0 at iteration 1')
    plot_R0_clusters(rank_cluster_map(scenario_test.R0_domain)[0])

    R0_domain = scenario_test.domain_at_iteration(0)

    plt.title('R0 at iteration 1')
    plot_R0_clusters(rank_cluster_map(R0_domain)[0])
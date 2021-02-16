import sys
import json
import random
import numpy as np

import itertools
import matplotlib.pyplot as plt
from typing import Union, Iterable, Tuple
from scipy.ndimage import binary_dilation
from collections import defaultdict


from cluster_find import rank_cluster_map
from plotting_methods import plot_R0_clusters
from parameters_and_setup import PATH_TO_INPUT_DATA, ENSEMBLES, STRUCTURING_ELEMENT


class Scenario_test():


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
        self.frag_comb = []
        self.epi_c = (None, None)
        self.payoffs = {}


    def domain_at_iteration(self, iterations: Union[int, Iterable]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find fragmented domain @ given iteration
        """
        fragment_lines = np.zeros_like(self.R0_domain)
        if isinstance(iterations, int):
            iterations = [iterations]

        if max(iterations) > self.iterations:
            print(f'Error, {iterations} too high, max iteration in dataset : {self.iterations}')
            return None

        R0_fragmented = np.copy(self.R0_domain)
        for iteration in iterations:
            R0_fragmented[self.connecting_patches[iteration - 1]] = 0
            fragment_lines[self.connecting_patches[iteration - 1]] = iteration

        return rank_cluster_map(R0_fragmented)[0], fragment_lines


    def fragment_combination(self) -> list:
        """
        Find all combinations for given iteration
        """
        iter_ = [1+i for i in range(self.iterations)]
        for i in iter_:
            comb = list(itertools.combinations(iter_, r=i))
            if len(comb) == 0:
                continue

            self.frag_comb.extend(comb)


    def get_epi_c(self, number: int):
        """
        Update list of N random epicenters
        """

        domain_indices = np.where(self.R0_domain)

        randomise_index = random.sample(range(0, len(domain_indices[0])), number)

        self.epi_c = [domain_indices[0][randomise_index],
                      domain_indices[1][randomise_index]]

        self.epi_c = tuple((i, j) for i, j in zip(self.epi_c[0], self.epi_c[1]))


    def find_single_payoff(self, epicenter: tuple, R0_fragmented:np.ndarray,
                           fragment_lines:np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Find the payoff for a single fragmentation combination
        """
        target = R0_fragmented[epicenter[0], epicenter[1]]
        assert target, 'Error, not expecting a zero-valued cluster-target.'

        target = np.where(R0_fragmented == target)
        num_patches_removed = len(target[0])
        arr_mask = np.zeros_like(R0_fragmented)
        arr_mask[target] = 1

        bd_target = binary_dilation(arr_mask, structure=STRUCTURING_ELEMENT)

        relevant_lines = fragment_lines[np.where(np.logical_and(bd_target, fragment_lines))]
        relevant_lines = np.unique(relevant_lines)
        relevant_lines = [int(line) for line in relevant_lines]

        num_culled = [len(np.where(fragment_lines == i)[0]) for i in relevant_lines]


        return tuple(relevant_lines), num_patches_removed, sum(num_culled)


    def find_all_payoffs(self, epi_center_number:int = 1):
        """
        For a sample of random epicenters, find the payoff : N_saved / N_culled
        """
        self.fragment_combination()
        self.get_epi_c(epi_center_number)
        for epi_c in self.epi_c:
            # Iterate through each epicenter
            assert not epi_c in self.payoffs  # ignore edge-case epicenters that already exist
            self.payoffs[epi_c] = {}
            print('epi c ', epi_c)

            for index, comb in enumerate(self.frag_comb):
                print('comb ', comb)
                # Iterate through all combinations of containment
                R0_fragmented, fragment_lines = self.domain_at_iteration(comb)
                relevant_lines, num_rem, num_culled = self.find_single_payoff(epi_c, R0_fragmented, fragment_lines)
                if relevant_lines in self.payoffs[epi_c]:
                    print('\t already processed ', relevant_lines)
                    plt.title(f'Og : {comb}, rel lines  {relevant_lines}')
                    plot_R0_clusters(R0_fragmented, epi_c=epi_c)
                    continue

                print('adding ', relevant_lines, num_rem, num_culled)
                self.payoffs[epi_c][relevant_lines] = {'Ns': num_rem, 'Nc':num_culled}

                # if index % 5 == 0:
                #     print('bounding lines : ', relevant_lines)
                #     print('N_R : ', num_rem)
                #     print('N_C : ', num_culled)
                #     plt.title(f'{comb}')
                #     plot_R0_clusters(R0_fragmented, epi_c=epi_c)

                index += 1

        print(self.payoffs)
        print('processed #', index)






if __name__ == '__main__':

    iter_ = 5
    scenario_test = Scenario_test(ENSEMBLES[2], beta_index=2, iterations=5)

    # scenario_test.fragment_combination()

    R0_arr = scenario_test.domain_at_iteration(iterations=1)
    # plot_R0_clusters(R0_arr)
    scenario_test.find_all_payoffs()
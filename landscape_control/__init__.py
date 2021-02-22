import os
import sys
import json
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import ClassVar, Tuple
from parameters_and_setup import PATH_TO_INPUT_DATA

class ClusterFrag():

    def __init__(self, ensemble_info: ClassVar, cg_factor:int, beta_index:int, iterations:int):
        scenario_name = f'{ensemble_info.species}_cg_{cg_factor}_beta_{beta_index}'
        self.path2_R0_processed = f'{ensemble_info.path2_R0_raw}/{scenario_name}_processed_R0_map.npy'
        self.path2_R0_fragmented = f'{ensemble_info.path2_R0_processed}/{scenario_name}_fragmented_R0_domain'
        self.path2_patch_data = f'{ensemble_info.path2_culled_indices}/{scenario_name}_iterations_{iterations}.json'

        self.cg_factor = cg_factor
        self.iterations = iterations
        self.rhos = ensemble_info.rhos
        self.raw_species_data = ensemble_info.raw_data
        self.R0_vs_rho = ensemble_info.R0_vs_rho_beta[beta_index]


    def execute(self, plot:bool=False):
        from ._domain_processing import get_R0_map, process_R0_map

        if os.path.exists(self.path2_patch_data):  # Check if data already exists
            print(f'\nWarning: data {self.path2_patch_data} already exists!')

        if os.path.exists(self.path2_R0_processed):  # load R0-map
            R0_processed = np.load(self.path2_R0_processed)

        else:  # generate R0-map from scratch
            R0_raw = get_R0_map(self.raw_species_data,  self.R0_vs_rho, self.rhos,  self.cg_factor)
            R0_processed = process_R0_map(R0_raw, get_cluster=1)
            np.save(self.path2_R0_processed, R0_processed)

        # run fragmentation
        connecting_patches, R0_fragmented = self.fragment_domain(R0_processed,  plot)
        np.save(self.path2_R0_fragmented, R0_fragmented)

        with open(f'{self.path2_patch_data}', 'w') as outfile:
            json.dump(connecting_patches, outfile, indent=4)
        return 'success'


    def fragment_domain(self, R0_map_raw:np.ndarray, plot: bool = False) -> Tuple[dict, np.ndarray]:
        """
        Iteratively fragment the largest cluster in the R0_map via targeted tree felling algorithm
        i.e. the `alpha-stepping' method. Save felled patches to file. Return fragmented domain.
        :rtype: object
        """

        from ._cluster_find import rank_cluster_map
        from ._plotting_methods import plot_R0_clusters, plot_fragmented_domain
        from ._fragmentation_methods import alpha_stepping_method, update_fragmentation_target
        from parameters_and_setup import FRAGMENT_RANK

        if R0_map_raw.max() < 1:  # Trivial map
            return None

        connecting_patches = {}
        R0_map = np.where(R0_map_raw > 1, R0_map_raw, 0)  # consider above threshold positions

        R0_map = R0_map * np.array(rank_cluster_map(R0_map, get_ranks=FRAGMENT_RANK)[0] > 0).astype(
            int)  # concentrate on the largest cluster
        R0_indices = np.where(R0_map)
        R0_indices = [min(R0_indices[0]) - 2, max(R0_indices[0]) + 2, min(R0_indices[1]) - 2, max(R0_indices[1]) + 2]
        R0_map = R0_map[R0_indices[0]:R0_indices[1], R0_indices[2]: R0_indices[3]]  # trim domain and save.

        R0_map_ = np.copy(R0_map)  # copy of processed domain - to save
        fragmented_domain = np.zeros_like(R0_map)

        if plot:
            plt.title('R0-map in put:')
            plot_R0_clusters(rank_cluster_map(R0_map)[0])

        R0_target = np.copy(R0_map)
        time = datetime.datetime.now()
        for iteration in range(self.iterations):
            print(f'iteration {iteration}')
            connecting_patch_indices, R0_target_fragmented = alpha_stepping_method(R0_target)
            connecting_patches[iteration] = connecting_patch_indices
            R0_target = update_fragmentation_target(R0_map, connecting_patch_indices)
            R0_target = R0_target * R0_map
            fragmented_domain[connecting_patch_indices] = iteration + 1

        if plot:
            plt.title(f'Fragmented to {iteration + 1} iterations')
            plot_fragmented_domain(fragmented_domain, R0_map)

        print(f'Time taken to fragment {self.iterations} iterations: {datetime.datetime.now() - time}')
        return connecting_patches, R0_map_



class Scenario_test():

    def __init__(self, ensemble_name:str, beta_index:int, cg_factor:int=5, species:str = 'fex', iterations:int=10):
        path2_ensemble = f'{PATH_TO_INPUT_DATA}/{ensemble_name}'
        path2_scenario = f'{species}_cg_{cg_factor}_beta_{beta_index}'
        path2_patch_data = f'connecting_patch_data/{path2_scenario}_iterations_{iterations}.json'
        path2_processed_R0_domain = f'processed_R0_maps/{path2_scenario}_processed_R0_map.npy'

        self.path2_payoff_data = f'{path2_ensemble}/fragmentation_payoff_data/'
        self.payoff_save_name =   f'{self.path2_payoff_data}{path2_scenario}_iterations_{iterations}.pickle'

        try:
            self.R0_domain = np.load(f'{path2_ensemble}/{path2_processed_R0_domain}')  # load domain
            with open(f'{path2_ensemble}/{path2_patch_data}', 'r') as infile:  # load fragmentation data
                connecting_patches = json.load(infile)

        except Exception as exc:
            sys.exit(f'Error, file(s) not found. Have you run the fragmentation algorithm ? \n {exc}')

        self.connecting_patches = {}
        self.all_culled_patches = np.zeros_like(self.R0_domain)
        for iteration, indices in connecting_patches.items():
            self.connecting_patches[int(iteration)] = (tuple(indices[0]), tuple(indices[1]))
            self.all_culled_patches[self.connecting_patches[int(iteration)]] = int(iteration) + 1

        self.iterations = iterations
        self.species = species
        self.cg_factor = cg_factor
        self.beta_index = beta_index
        self.payoffs = {}

        from ._scenario_test import fragment_combination, get_epi_c, domain_at_iteration, get_epicenter_payoff
        from ._plotting_methods import plot_fragmented_domain

        self.get_epi_c = get_epi_c
        self.domain_at_iteration = domain_at_iteration
        self.get_epicenter_payoff = get_epicenter_payoff
        self.fragment_combination = fragment_combination
        self.plot_fragmented_domain = plot_fragmented_domain



    def find_all_payoffs(self, epi_center_number:int = 1, plot_check:bool=False) -> Tuple[dict, int]:
        """
        For a sample of random epicenters, find the payoff : N_saved / N_culled
        """

        containment_combos = self.fragment_combination(self.iterations)
        epi_centers = self.get_epi_c(self.R0_domain, self.all_culled_patches, epi_center_number)
        for index, epi_c in enumerate(epi_centers):
            print(f'{index}/{epi_center_number}')
            # Iterate through each epicenter
            assert not epi_c in self.payoffs  # ignore edge-case epicenters that already exist
            self.payoffs[epi_c] = {}

            for index, comb in enumerate(containment_combos):
                # Iterate through all combinations of containment
                R0_fragmented, fragment_lines = self.domain_at_iteration(self.R0_domain, self.connecting_patches,
                                                                         iterations=comb)

                relevant_lines, num_rem, num_culled = self.get_epicenter_payoff(epi_c, self.R0_domain, R0_fragmented,
                                                                                fragment_lines)

                if relevant_lines in self.payoffs[epi_c]:
                    if plot_check and index % 100 == 0:
                        striped = [i for i in comb if i not in relevant_lines]
                        plt.title(f'combination : {comb} | relevant lines : {relevant_lines} -> strip : {striped}')
                        self.plot_fragmented_domain(fragment_lines, np.copy(self.R0_domain), epi_c, show_text=True)
                    continue

                self.payoffs[epi_c][relevant_lines] = {'Ns': num_rem, 'Nc': num_culled}
                index += 1

        if not os.path.exists(f'{self.path2_payoff_data}'):
            os.mkdir(f'{self.path2_payoff_data}')


        with open(self.payoff_save_name, 'wb') as handle:
            pickle.dump(self.payoffs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.payoffs, index


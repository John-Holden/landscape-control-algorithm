import os
import sys
import json
import pickle
import datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import ClassVar, Tuple
from parameters_and_setup import PATH_TO_INPUT_DATA


class ClusterFrag:

    def __init__(self, ensemble_info: ClassVar, cg_factor:int, beta_index:int, iterations:int):
        scenario_name = f'{ensemble_info.species}_cg_{cg_factor}_beta_{beta_index}'

        self.path2_R0_processed = f'{ensemble_info.path2_R0_raw}/{scenario_name}_processed_R0_map.npy'
        self.path2_fragmented_map = f'{ensemble_info.path2_R0_processed}/{scenario_name}_fragmented_domain'
        self.path2_patch_data = f'{ensemble_info.path2_culled_indices}/{scenario_name}_iterations_{iterations}.json'

        self.cg_factor = cg_factor
        self.iterations = iterations
        self.rhos = ensemble_info.rhos
        self.raw_species_data = ensemble_info.raw_data
        self.R0_vs_rho = ensemble_info.R0_vs_rho_beta[beta_index]

    def execute(self, plot: bool = False):
        from ._domain_processing import get_R0_map, process_R0_map

        if os.path.exists(self.path2_patch_data):  # Check if data already exists
            msg = f'\n Overwriting:  {self.path2_patch_data}'
            warnings.warn(msg)
        if os.path.exists(self.path2_R0_processed):  # load R0-map
            R0_processed = np.load(self.path2_R0_processed)
        else:  # generate R0-map from scratch, take threshold values of R0 and crop around the target cluster
            R0_raw = get_R0_map(self.raw_species_data,  self.R0_vs_rho, self.rhos,  self.cg_factor)
            R0_processed = process_R0_map(R0_raw, get_cluster=1)
            np.save(self.path2_R0_processed, R0_processed)

        if R0_processed.max() < 1:  # Trivial map
            return None

        connecting_patches, fragmented_map = self.fragment_domain(R0_processed,  plot)

        np.save(self.path2_fragmented_map, fragmented_map)
        with open(f'{self.path2_patch_data}', 'w') as outfile:
            json.dump(connecting_patches, outfile, indent=4)

        return 'success'

    def fragment_domain(self, R0_map: np.ndarray, plot: bool = False) -> Tuple[dict, np.ndarray]:
        """
        Iteratively fragment the largest cluster in the R0_map via targeted tree felling algorithm -`alpha-stepping' .
        Return a dictionary of indices and a spatial representation of the  fragmented domain.
        """

        from ._cluster_find import rank_cluster_map
        from ._fragmentation_methods import alpha_stepping_method, update_fragmentation_target, patch_tidy
        from .plotting_methods import plot_R0_clusters, plot_fragmented_domain

        connecting_patches = {}
        fragmented_domain = np.zeros_like(R0_map)

        if plot:
            plt.title('R0-map input:')
            plot_R0_clusters(rank_cluster_map(R0_map)[0])

        R0_target = np.copy(R0_map)
        time = datetime.datetime.now()
        for iteration in range(self.iterations):
            print(f'iteration {iteration}')
            connector_patch_indices = alpha_stepping_method(R0_target)
            connecting_patches[iteration] = connector_patch_indices
            R0_target = update_fragmentation_target(R0_map, connector_patch_indices)
            R0_target = R0_target * R0_map
            fragmented_domain[connector_patch_indices] = iteration + 1

        if plot:
            plt.title(f'Fragmented to {self.iterations} iterations')
            plot_fragmented_domain(fragmented_domain, R0_map)

        print(f'Time taken to fragment {self.iterations} iterations: {datetime.datetime.now() - time}')
        return connecting_patches, fragmented_domain


class ScenarioTest:

    def __init__(self, ensemble_name: str, beta_index: int, cg_factor: int = 5, species: str = 'Fex',
                 iterations: int = 10):

        path2_ensemble = f'{PATH_TO_INPUT_DATA}/{ensemble_name}'
        path2_scenario = f'{species}_cg_{cg_factor}_beta_{beta_index}'
        path2_patch_data = f'connecting_patch_data/{path2_scenario}_iterations_{iterations}.json'
        path2_processed_R0_domain = f'processed_R0_maps/{path2_scenario}_processed_R0_map.npy'
        path2_fragmented_R0_domain = f'fragmented_R0_domain/{path2_scenario}_fragmented_domain.npy'

        self.path2_payoff_data = f'{path2_ensemble}/fragmentation_payoff_data/'
        self.payoff_save_name = f'{self.path2_payoff_data}{path2_scenario}_iterations_{iterations}.pickle'

        try:
            self.R0_domain = np.load(f'{path2_ensemble}/{path2_processed_R0_domain}')  # load domain
            self.fragmented_domain = np.load(f'{path2_ensemble}/{path2_fragmented_R0_domain}')  # load domain

        except Exception as exc:
            sys.exit(f'Error, file(s) not found. Have you run the fragmentation algorithm ? \n {exc}')

        self.species = species
        self.scenario_store = {}
        self.cg_factor = cg_factor
        self.iterations = iterations
        self.beta_index = beta_index
        self.population_size = len(np.where(self.R0_domain)[0])

        from ._scenario_test import fragment_combination, get_epi_c, domain_at_iteration, get_epicenter_payoff, \
            add_rank_to_dict

        from .plotting_methods import plot_fragmented_domain, plot_R0_clusters

        self.get_epi_c = get_epi_c
        self.add_rank_to_dict = add_rank_to_dict
        self.domain_at_iteration = domain_at_iteration
        self.get_epicenter_payoff = get_epicenter_payoff
        self.fragment_combination = fragment_combination

        self.plot_R0_clusters = plot_R0_clusters
        self.plot_fragmented_domain = plot_fragmented_domain

    def find_all_payoffs(self, plot_check: bool = False) -> Tuple[dict, int]:
        """
        For a sample of random epicenters, find the payoff : N_saved / N_culled
        """

        payoffs_list = []
        epi_center_list = []
        relevant_lines_list = []

        containment_combos = self.fragment_combination(self.iterations)
        epi_centers = self.get_epi_c(self.R0_domain, self.fragmented_domain)

        for i, epi_c in enumerate(epi_centers):
            print(f'{i}/ {len(epi_centers)}')
            # Iterate through each epicenter
            assert epi_c not in self.scenario_store  # ignore edge-case epicenters that already exist
            self.scenario_store[epi_c] = {}
            for c, comb in enumerate(containment_combos):
                # Iterate through all combinations of containment
                R0_fragmented, fragment_lines = self.domain_at_iteration(self.R0_domain, self.fragmented_domain, comb)

                if R0_fragmented is None:
                    continue

                fragment_lines, relevant_lines, num_rem, num_culled = self.get_epicenter_payoff(epi_c, R0_fragmented, fragment_lines)

                if relevant_lines in self.scenario_store[epi_c]:
                    if plot_check and c % 50 == 0:
                        striped = [i for i in comb if i not in relevant_lines]
                        plt.title(f'combination : {comb} | relevant lines : {relevant_lines} -> strip : {striped}')
                        self.plot_fragmented_domain(fragment_lines, np.copy(self.R0_domain), epi_c, show_text=True)
                    continue

                num_saved = self.population_size - num_rem
                self.scenario_store[epi_c][relevant_lines] = {'Ns': num_saved, 'Nr': num_rem, 'Nc': num_culled}
                relevant_lines_list.append(relevant_lines)
                payoffs_list.append(num_saved / num_culled)
                epi_center_list.append(epi_c)

        self.scenario_store = self.add_rank_to_dict(payoffs_list, epi_center_list, relevant_lines_list, self.scenario_store)

        if not os.path.exists(f'{self.path2_payoff_data}'):
            os.mkdir(f'{self.path2_payoff_data}')

        with open(self.payoff_save_name, 'wb') as handle:
            pickle.dump(self.scenario_store, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.scenario_store, len(payoffs_list)


import os
import json
import numpy as np
from typing import ClassVar


class ClusterFrag():

    def __init__(self, ensemble_info: ClassVar, cg_factor:int, beta_index:int, iterations:int):
        scenario_name = f'{ensemble_info.species}_cg_{cg_factor}_beta_{beta_index}'
        self.path2_R0_raw = f'{ensemble_info.path2_R0_raw}/{scenario_name}_raw_fitted_R0_map.npy'
        self.path2_processed_R0 = f'{ensemble_info.path2_R0_processed}/{scenario_name}_processed_R0_domain'
        self.path2_patch_data = f'{ensemble_info.path2_culled_indices}/{scenario_name}_iterations_{iterations}.json'
        self.raw_species_data = ensemble_info.raw_data
        self.R0_vs_rho = ensemble_info.R0_vs_rho_beta[beta_index]
        self.rhos = ensemble_info.rhos
        self.iterations = iterations

    def execute(self):
        from ._domain_processing import get_R0_map
        from ._fragmentation_methods import fragment_domain

        if os.path.exists(self.path2_patch_data):  # Check if data already exists
            print(f'\nWarning: data {self.path2_patch_data} already exists!')

        if os.path.exists(self.path2_R0_raw):  # load R0-map
            R0_raw = np.load(self.path2_R0_raw)

        else:  # generate R0-map from scrat h
            R0_raw = get_R0_map(self.raw_species_data,  self.R0_vs_rho, self.rhos,  self.coarse_grain_factor)
            np.save(self.path2_R0_raw, R0_raw)

        connecting_patches, R0_processed_domain = fragment_domain(R0_raw, self.iterations, True)  # run fragmentation
        np.save(self.path2_processed_R0, R0_processed_domain)

        with open(f'{self.path2_patch_data}', 'w') as outfile:
            json.dump(connecting_patches, outfile, indent=4)
        return 'success'



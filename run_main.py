import os
import numpy as np

PATH_TO_DATA_STORE = f'{os.getcwd()}/data_store'

class Landscape_structs():
    def __init__(self, ensemble_name :str, species:str):
        self.beta_index = 0  # choose index of
        self.path_to_ensemble = f'{PATH_TO_DATA_STORE}/{ensemble_name}'
        self.R0_vs_rho_beta = np.load(f'{self.path_to_ensemble}/ensemble.npy')
        self.rhos = np.load(f'{self.path_to_ensemble}/rhos.npy')
        self.betas = np.load(f'{self.path_to_ensemble}/betas.npy')
        self.raw_data = 0.01 * np.genfromtxt(f'{PATH_TO_DATA_STORE}/{species}.csv', delimiter=',')  # Covert to density
        self.raw_data = self.raw_data * np.load(f'{PATH_TO_DATA_STORE}/uk_isle_shape.npy')[1:-1, 1:-1]  # Clean channel isles etc.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   from delegator_methods import get_clusters_over_betas
   ensemble_name = 'landscape_control_input_test_data'
   coarse_grain_level = 5
   cluster_ranks = 1
   betas = np.load(f'{PATH_TO_DATA_STORE}/{ensemble_name}/betas.npy')
   cluster_sizes = get_clusters_over_betas(ensemble_name=ensemble_name, cluster_ranks=cluster_ranks,
                                           coarse_grain_level=coarse_grain_level)

   np.save(f'{PATH_TO_DATA_STORE}/{ensemble_name}/cluster_size_vs_beta', cluster_sizes)


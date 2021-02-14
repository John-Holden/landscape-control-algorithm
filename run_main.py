import os
import numpy as np

PATH_TO_INPUT_DATA = f'{os.getcwd()}/data_store'

STRUCTURING_ELEMENT1  = np.ones(shape=[3,3]) # defined nearest neighbourhood of land patches.
STRUCTURING_ELEMENT2  = np.ones(shape=[3,3])
STRUCTURING_ELEMENT2[(0, 0, 2, 2), (0, 2, 0, 2)] = 0 # defined nearest neighbourhood of land patches.
STRUCTURING_ELEMENTS = {'MOORE': STRUCTURING_ELEMENT1,
                         'VON-N': STRUCTURING_ELEMENT2}

STRUCTURING_ELEMENT = STRUCTURING_ELEMENTS['MOORE']


class Ensemble_info():   # High level struct, holds all ensemble info
    def __init__(self, ensemble_name :str):
        self.species = 'fex'
        self.path_to_ensemble = f'{PATH_TO_INPUT_DATA}/{ensemble_name}'
        self.R0_vs_rho_beta = np.load(f'{self.path_to_ensemble}/ensemble.npy')
        self.rhos = np.load(f'{self.path_to_ensemble}/rhos.npy')
        self.betas = np.load(f'{self.path_to_ensemble}/betas.npy')
        # x 0.01 ==> Covert to canopy cover density
        self.raw_data = 0.01 * np.genfromtxt(f'{PATH_TO_INPUT_DATA}/{self.species}.csv', delimiter=',')
        # Clean channel isles etc.
        self.raw_data = self.raw_data * np.load(f'{PATH_TO_INPUT_DATA}/uk_isle_shape.npy')[1:-1, 1:-1]
        # if not present, init fragmentation directory
        if not os.path.exists(f'{self.path_to_ensemble}/fragmentation_data'):
            os.mkdir(f'{self.path_to_ensemble}/fragmentation_data')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from delegator_methods import orchestrate_fragmentation
    ensembles = {0: 'landscape_control_input_test_data',
                 1: 'landscape_control_input_test_beta_cluster_sizes',
                 2: 'landscape_control_package'}

    orchestrate_fragmentation(ensemble_name=ensembles[2])





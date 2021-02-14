import os
import numpy as np

PATH_TO_INPUT_DATA = f'{os.getcwd()}/data_store'

STRUCTURING_ELEMENT1  = np.ones(shape=[3,3]) # defined nearest neighbourhood of land patches.
STRUCTURING_ELEMENT2  = np.ones(shape=[3,3])
STRUCTURING_ELEMENT2[(0, 0, 2, 2), (0, 2, 0, 2)] = 0 # defined nearest neighbourhood of land patches.
STRUCTURING_ELEMENTS = {'MOORE': STRUCTURING_ELEMENT1,
                         'VON-N': STRUCTURING_ELEMENT2}

STRUCTURING_ELEMENT = STRUCTURING_ELEMENTS['MOORE']

ENSEMBLES = {0: 'landscape_control_input_test_data',
             1: 'landscape_control_input_test_beta_cluster_sizes',
             2: 'landscape_control_package'}


TEST_TOP_N = 5  # Test the top N clusters connect to from R0-connected
INTERFACE_MARGIN = 5
MIN_CLUSTER_INTERMEDIATE_SIZE = 2
TARGETS_C1_C2 = [1, 2]
MIN_CLUSTER_JOIN_SIZE = 5
MIN_CLUSTER_JOIN_RATIO = 0.10
FRAGMENT_RANK = 1


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









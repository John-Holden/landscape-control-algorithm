import os
import numpy as np

PATH_TO_INPUT_DATA = f'{os.getcwd()}/data_store'
PATH_TO_EXCEPTIONS = f'{os.getcwd()}/data_store/exceptions/'

STRUCTURING_ELEMENT1 = np.ones(shape=[3,3]) # defined nearest neighbourhood of land patches.
STRUCTURING_ELEMENT2 = np.ones(shape=[3,3])
STRUCTURING_ELEMENT2[(0, 0, 2, 2), (0, 2, 0, 2)] = 0 # defined nearest neighbourhood of land patches.
STRUCTURING_ELEMENTS = {'MOORE': STRUCTURING_ELEMENT1,
                        'VON-N': STRUCTURING_ELEMENT2,
                        "MOORE2": np.ones(shape=[4, 4])}

STRUCTURING_ELEMENT = STRUCTURING_ELEMENTS['MOORE']


def set_structuring_element(structure_name: str) -> np.ndarray:
    """

    :param structure_name:
    :return:
    """
    STRUCTURING_ELEMENT = STRUCTURING_ELEMENTS[structure_name]
    print(f'setting element to {structure_name} \n {STRUCTURING_ELEMENT}')
    return STRUCTURING_ELEMENT


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


class EnsembleInfo:   # High level struct, holds all ensemble info

    def __init__(self, ensemble_name: str):
        self.species = 'Fex'
        self.path_to_ensemble = f'{PATH_TO_INPUT_DATA}/{ensemble_name}'
        self.R0_vs_rho_beta = np.load(f'{self.path_to_ensemble}/ensemble.npy')
        self.rhos = np.load(f'{self.path_to_ensemble}/rhos.npy')
        self.betas = np.load(f'{self.path_to_ensemble}/betas.npy')
        # x 0.01 ==> Covert to canopy cover density
        raw_data = 0.01 * np.genfromtxt(f'{PATH_TO_INPUT_DATA}/{self.species}.csv', delimiter=',')
        # Clean channel isles etc.
        self.raw_data = raw_data * np.load(f'{PATH_TO_INPUT_DATA}/uk_isle_shape.npy')[1:-1, 1:-1]

        # save/load paths
        self.path2_R0_raw = f'{self.path_to_ensemble}/processed_R0_maps'
        self.path2_R0_processed = f'{self.path_to_ensemble}/fragmented_R0_domain'
        self.path2_culled_indices = f'{self.path_to_ensemble}/connecting_patch_data'
        self.path2_payoff_data = f'{self.path_to_ensemble}/fragmentation_payoff_data'

        # if not present, init directories
        if not os.path.exists(self.path2_R0_raw):
            os.mkdir(self.path2_R0_raw)

        if not os.path.exists(self.path2_R0_processed):
            os.mkdir(self.path2_R0_processed)

        if not os.path.exists(self.path2_payoff_data):
            os.mkdir(self.path2_payoff_data)

        if not os.path.exists(self.path2_culled_indices):
            os.mkdir(self.path2_culled_indices)













from landscape_control import ClusterFrag
from parameters_and_setup import Ensemble_info

enemble = Ensemble_info('landscape_control_package')

cfrag = ClusterFrag(enemble, cg_factor=5, beta_index=1, iterations=2)

cfrag.execute()
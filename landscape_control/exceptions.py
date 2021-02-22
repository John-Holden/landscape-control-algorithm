import numpy as np
import matplotlib.pyplot as plt


class ClustersDidNotFragment(Exception):

    def __init__(self, R0_dsconnected, R0_connected, R0_fragmented, cluster_targets, connector_patches,
                 num, msg='Error, clusters did not fragment!', show=True):

        self.msg = msg
        self.num = num
        self.R0_frag = R0_fragmented
        self.R0_connect = R0_connected,
        self.R0_discon = R0_dsconnected
        self.cluster_targets = cluster_targets
        self.connector_patches = connector_patches

        self.plot_save_errors(show)


    def __str__(self):
        return f'\n {self.msg} \n Found {self.num} of patches to remove. \n Saved domain-map data to file.'

    def plot_save_errors(self, show):
        """
        Display errors visually and save to exception folder for further inspection.
        """

        from ._fragmentation_methods import rank_cluster_map, TIMESTAMP
        from ._plotting_methods import plot_R0_clusters

        if show:
            plt.title('Error pre-connected map')
            plot_R0_clusters(rank_cluster_map(self.R0_discon)[0])
            plt.title('Error post-connected map')
            plot_R0_clusters(rank_cluster_map(self.R0_connect)[0])
            plt.title('Error, domain did not fragment')
            plot_R0_clusters(rank_cluster_map(self.R0_frag)[0])
            plt.title('Error, cluster targets')
            plot_R0_clusters(self.cluster_targets)
            if self.num:
                plt.title(f'Error, connecting patches, number removed {self.num}')
                plot_R0_clusters(self.connector_patches)

        np.save(f'./data_store/exceptions/e_fragments_{TIMESTAMP}', self.R0_frag)
        np.save(f'./data_store/exceptions/e_targets_{TIMESTAMP}', self.cluster_targets)
        np.save(f'./data_store/exceptions/e_pre_connected_map_{TIMESTAMP}', self.R0_discon)
        np.save(f'./data_store/exceptions/e_post_connected_map_{TIMESTAMP}', self.R0_connect)
        np.save(f'./data_store/exceptions/e_patches_detected_{TIMESTAMP}', self.connector_patches)


if 0 != 1:
    raise  ClustersDidNotFragment(1)
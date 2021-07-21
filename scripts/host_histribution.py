import numpy as np
from parameters_and_setup import PATH_TO_INPUT_DATA
import seaborn as sns
import landscape_control.plotting_methods as treePlt


def get_plot_host_dist(species: str):
    raw_data = 0.01 * np.genfromtxt(f'{PATH_TO_INPUT_DATA}/{species}.csv', delimiter=',')
    raw_data = raw_data * np.load(f'{PATH_TO_INPUT_DATA}/uk_isle_shape.npy')[1:-1, 1:-1]
    raw_data_flat = raw_data.flatten()
    raw_data_flat = np.delete(raw_data_flat, np.isnan(raw_data_flat))
    treePlt.host_distribution_flat(raw_data_flat, )


def get_plot_host_density_map(species: str, cg_factor: int):
    """

    :param species:
    :param cg_factor:
    :return:
    """
    from landscape_control.domain_processing import coarse_grain

    raw_data = 0.01 * np.genfromtxt(f'{PATH_TO_INPUT_DATA}/{species}.csv', delimiter=',')
    raw_data = raw_data * np.load(f'{PATH_TO_INPUT_DATA}/uk_isle_shape.npy')[1:-1, 1:-1]

    cg_data = coarse_grain(raw_data, cg_factor)
    treePlt.host_density_map(cg_data)


if __name__ == '__main__':
    get_plot_host_dist('Fex')
    # get_plot_host_density_map('Fex', cg_factor=1)
import numpy as np
from typing import Iterable, Union
from scipy.optimize import curve_fit


def linear_func(xdata:Iterable, c:int):
    return c * xdata


def get_R0_gradient_fitting(species_distribution_map:np.ndarray, rhos:np.ndarray,
                            R0_v_rho_mapping:np.ndarray, print_fitting=False) -> np.ndarray:
    """
     For an array of R0 vs rho values, fit data to linear function. Then return tree-density mapped to R0-values.
    """
    popt, pcov = curve_fit(linear_func, rhos, R0_v_rho_mapping)
    if print_fitting:
        print(f'Fitted gradients {popt[0]}, Variance {pcov[0]}')
    return species_distribution_map * popt[0]


def coarse_grain(domain, cg_factor) -> 'float type, arr[n x m]':
    """
    Re-scale original dataset to a given granularity, re-shape to:
        cg_factor km^2 x cg_factor km^2
    """
    if 1 in np.isnan(domain):
        domain = np.where(np.isnan(domain), 0, domain)

    x_ind = 0
    new_xaxis = np.arange(0, domain.shape[0], cg_factor)
    new_yaxis = np.arange(0, domain.shape[1], cg_factor)
    cg_arr = np.zeros([len(new_xaxis), len(new_yaxis)])
    for row in new_xaxis:
        y_ind = 0
        for col in new_yaxis:
            patch = domain[row:row + cg_factor, col:col + cg_factor]
            av_value = np.sum(patch)
            cg_arr[x_ind][y_ind] = av_value

            y_ind += 1
        x_ind += 1
    cg_arr = cg_arr / cg_factor ** 2
    if 1 in np.isnan(domain):
        cg_arr[np.where(cg_arr == 0)] = np.nan

    return cg_arr


def get_R0_map(raw_species_data:np.ndarray, R0_vs_rho:np.ndarray,
               rhos:np.ndarray, coarse_grain_factor:Union[None, int]=None) -> np.ndarray:
    """
    Process a single domain, for one beta value, and return R0 map.
    """
    if max(R0_vs_rho) < 1:
        print('Warning: trivial data-set, max R0 < 1')

    if coarse_grain_factor is not None:
        raw_species_data_cg = coarse_grain(domain=raw_species_data, cg_factor=coarse_grain_factor)

    return get_R0_gradient_fitting(raw_species_data_cg, rhos, R0_vs_rho)
"""
Methods related to processing the input domain, that is, re-scale via coarse-graining and finding R0-maps.
"""
import numpy as np
from scipy.optimize import curve_fit

def linear_func(xdata, c):  # linear function to fit against rho mapping
    return c * xdata


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


def get_R0_map_boundary_mapping(species_distribution_map:np.ndarray, rhos:np.ndarray,
                                R0_v_rho_mapping:np.ndarray) -> np.ndarray:
    """
    From sub-grid mapping function, map field value to species distribution map density for spatial coordinate i,j
    """
    R0_map = np.zeros_like(species_distribution_map)
    # where key i is index in rho-space and value [rho_i, rho_i+1] is boundary
    rho_boundaries = {i : [rhos[i], rhos[i+1]] for i in range(len(rhos)-1)}
    max_density = rho_boundaries[len(rhos)-2][1]  # maximum density in data
    for i, row in enumerate(species_distribution_map):
        for j, col in enumerate(row):
            d_ij = species_distribution_map[i, j]  # density value at point i,j
            if np.isnan(d_ij):  # if sea, then pass
                pass
            else:  # if land region: map rho_ij to a velocity-space value
                for rho_box in rho_boundaries:  # iterate through rho-space $ check against if == density_ij
                    boundary = rho_boundaries[rho_box]
                    # If density in the range interval then set map location density_ij == velocity(density)
                    if boundary[0] <= d_ij < boundary[1]:
                        R0_map[i, j] = R0_v_rho_mapping[rho_box]
                    # check if density bigger than rho given space
                    # - cap at highest given rho space boundary mapping
                    elif d_ij > max_density:  # if density above max density, cap to max value
                        R0_map[i, j] = R0_v_rho_mapping[len(rho_boundaries) - 1]
    return R0_map


def get_R0_gradient_fitting(species_distribution_map:np.ndarray, rhos:np.ndarray,
                            R0_v_rho_mapping:np.ndarray, print_fitting=False) -> np.ndarray:
    """
     For an array of R0 vs rho values, fit data to linear function. Then return tree-density mapped to R0-values.
    """
    popt, pcov = curve_fit(linear_func, rhos, R0_v_rho_mapping)
    if print_fitting:
        print(f'Fitted gradients {popt[0]}, Variance {pcov[0]}')
    return species_distribution_map * popt[0]


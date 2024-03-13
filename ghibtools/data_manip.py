import numpy as np
import pandas as pd
import xarray as xr
import string

def midx_da(da , dim , midx_labels, midx_coords):
    midx = pd.MultiIndex.from_arrays(midx_coords, names=midx_labels)
    da_return = da.assign_coords(coords = {dim:midx})
    return da_return

def get_plot_letters(kind = 'upper'):
    if kind == 'upper':
        alphabet = list(string.ascii_uppercase)
    else:
        alphabet = list(string.ascii_lowercase)
    return [f'{letter})' for letter in alphabet]

def attribute_subplots(element_list, nrows, ncols):
    assert nrows * ncols >= len(element_list), f'Not enough subplots planned ({nrows*ncols} subplots but {len(element_list)} elements)'
    subplots_pos = {}
    counter = 0
    for r in range(nrows):
        for c in range(ncols):
            if counter == len(element_list):
                break
            subplots_pos[f'{element_list[counter]}'] = [r,c]
            counter += 1
    return subplots_pos  

def time_ratio_year(num, den, mode = 'from'):
    """
    num : int or float numerator in the ratio
    den : int or float denominator in the ratio
    mode : str ("from" compute date with 1 - ratio, "to" compute date with ratio
    """
    assert mode in ['from','to'], 'Mode should be "from" or "to"'
    date_vector = pd.date_range(start = '2000-01-01', end = '2001-01-01', freq = '30s')
    if mode == 'from':
        ratio = 1 - (num / den)
    elif mode == 'to':
        ratio = num / den
    ind_date = int(date_vector.size*(ratio))
    return date_vector[ind_date]
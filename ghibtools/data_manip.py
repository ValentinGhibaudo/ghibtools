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
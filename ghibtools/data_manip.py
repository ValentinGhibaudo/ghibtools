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


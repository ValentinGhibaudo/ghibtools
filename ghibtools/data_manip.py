import numpy as np
import pandas as pd
import xarray as xr

def midx_da(da , dim , midx_labels, midx_coords):
    midx = pd.MultiIndex.from_arrays(midx_coords, names=midx_labels)
    da_return = da.assign_coords(coords = {dim:midx})
    return da_return

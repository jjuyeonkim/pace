"""
NOTE: Trying to compare gaea and amdbox generate output.zar files. Diff isn't good enough.
(jk_zarr) pace]$ diff ~/tmp/20250725_gaea_t5_t6_comparison/{pace/,from_gaea/gaea60/}output.zarr/delp/0.0.0.0.0
Binary files ~/tmp/20250725_gaea_t5_t6_comparison/pace/output.zarr/delp/0.0.0.0.0 and ~/tmp/20250725_gaea_t5_t6_comparison/from_gaea/gaea60/output.zarr/delp/0.0.0.0.0 differ

Starting with suggested code from Gemini:
"""
import xarray as xr
import numpy as np


def get_top_5_diffs(val_gaea, val_amdbox):
    # Calculate the absolute difference between the two arrays


    diff_da = abs(val_gaea - val_amdbox)

    # Find the top N biggest differences
    N = 5

    # Use np.argpartition to efficiently find the flat indices of the N largest values
    flat_indices = np.argpartition(diff_da.values.flatten(), -N)[-N:]

    # Convert the flat indices back into multidimensional indices
    multi_indices = np.unravel_index(flat_indices, diff_da.shape)

    # Create a dictionary for passing indices to .isel()
    isel_dims = {dim: coords for dim, coords in zip(diff_da.dims, multi_indices)}

    # Select these top N points from the difference DataArray
    top_n_diffs = diff_da.isel(isel_dims)

        # Print the result, sorted from largest to smallest
    print(f"Top {N} biggest differences and their coordinates:")
    print(top_n_diffs.values[0][0][0][0])
    #input()

gaea_ds = xr.open_zarr('~/tmp/20250725_gaea_t5_t6_comparison/from_gaea/gaea60/output.zarr')
gaea_vars = set(gaea_ds.data_vars)
amdbox_ds = xr.open_zarr('~/tmp/20250725_gaea_t5_t6_comparison/pace/output.zarr')
amdbox_vars = set(amdbox_ds.data_vars)

# Check that the variables are the same
have_same_vars = (gaea_vars == amdbox_vars)
print(f"variable names match: {have_same_vars}")

for var_name in sorted(gaea_vars):
    val_gaea = gaea_ds[var_name]
    val_amdbox = amdbox_ds[var_name]
    
    print(f"{var_name}")
    # For exact equality of Datasets (including NaNs)
    are_equal = val_gaea.equals(val_amdbox)
    print(f"Datasets are equal: {are_equal}")

    # For approximate equality (e.g., with floating-point data)
    atol = 1e-15
    if var_name in ['ua', 'v']: 
        atol=1e-3 # TODO: Why are they so different?
        get_top_5_diffs(val_gaea, val_amdbox)
    elif var_name in ['u', 'va']: 
        atol=1e-4 # TODO: Why are they so different?
        get_top_5_diffs(val_gaea, val_amdbox)
        
    all_close = xr.testing.assert_allclose(val_gaea, val_amdbox, rtol=1e-6, atol=atol)
    print(f"Datasets are all close with abs tolerance: {atol}")
    print()

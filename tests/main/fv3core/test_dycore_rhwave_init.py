import os
import unittest.mock
from dataclasses import fields
from datetime import timedelta
from typing import Tuple
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import pyFV3.initialization.analytic_init as ai
from ndsl import (
    CompilationConfig,
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    DaceConfig,
    GridIndexing,
    NullComm,
    Quantity,
    QuantityFactory,
    StencilConfig,
    StencilFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from ndsl.grid import DampingCoefficients, GridData, MetricTerms
from ndsl.performance.timer import NullTimer, Timer
from ndsl.stencils.testing import assert_same_temporaries, copy_temporaries
from pyFV3 import DycoreState, DynamicalCore, DynamicalCoreConfig


DIR = os.path.abspath(os.path.dirname(__file__))
PACE_DIR = os.path.join(DIR, "../../../")

# TODO This is basically a copy of ~/pace/tests/main/fv3core/test_dycore_call.py
# NEED TO UPDATE THIS WHILE Porting RHWave 4 test from fortran

def setup_dycore() -> Tuple[DynamicalCore, DycoreState, Timer]:
    backend = "numpy"
    config = DynamicalCoreConfig(
        layout=(1, 1),
        npx=13,
        npy=13,
        npz=79,
        ntiles=6,
        nwat=6,            # TODO: Fortran test is 0, only nwat=6 is implemented in pace; How to change?
        dt_atmos=225,
        a_imp=1.0,         # TODO: What is fortran equiv?
        beta=0.0,
        consv_te=False,  # not implemented, needs allreduce
        d2_bg=0.0,
        d2_bg_k1=0.2,      # TODO: What is fortran equiv?
        d2_bg_k2=0.1,      # TODO: What is fortran equiv?
        d4_bg=0.15,
        d_con=1.0,         # TODO: What is fortran equiv?
        d_ext=0.0,         # TODO: What is fortran equiv?
        dddmp=0.0,
        delt_max=0.002,    # TODO: What is fortran equiv?
        do_sat_adj=True,   # TODO: What is fortran equiv?
        do_vort_damp=True, # TODO: What is fortran equiv?
        fill=True,         # TODO: What is fortran equiv?
        hord_dp=6,
        hord_mt=6,
        hord_tm=6,
        hord_tr=8,
        hord_vt=6,
        hydrostatic=False, # TODO: What is fortran equiv?
        k_split=1,         # TODO: What is fortran equiv?
        ke_bg=0.0,         # TODO: What is fortran equiv?
        kord_mt=9,         # TODO: What is fortran equiv?
        kord_tm=-9,        # TODO: What is fortran equiv?
        kord_tr=9,         # TODO: What is fortran equiv?
        kord_wz=9,         # TODO: What is fortran equiv?
        n_split=1,
        nord=3,
        p_fac=0.05,        # TODO: What is fortran equiv?
        rf_fast=True,      # TODO: What is fortran equiv?
        rf_cutoff=3000.0,  # TODO: What is fortran equiv?
        tau=10.0,          # TODO: What is fortran equiv?
        vtdm4=0.06,        # TODO: What is fortran equiv?
        z_tracer=True,     # TODO: What is fortran equiv?
        do_qa=True,        # TODO: What is fortran equiv?
    )
    mpi_comm = NullComm(
        rank=0, total_ranks=6 * config.layout[0] * config.layout[1], fill_value=0.0
    )
    partitioner = CubedSpherePartitioner(TilePartitioner(config.layout))
    communicator = CubedSphereCommunicator(mpi_comm, partitioner)
    dace_config = DaceConfig(communicator=communicator, backend=backend)
    stencil_config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend, rebuild=False, validate_args=True
        ),
        dace_config=dace_config,
    )
    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=config.npx - 1,
        ny_tile=config.npy - 1,
        nz=config.npz,
        n_halo=3,
        extra_dim_lengths={},
        layout=config.layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    grid_indexing = GridIndexing.from_sizer_and_communicator(
        sizer=sizer, comm=communicator
    )
    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)
    eta_file = "tests/main/input/eta79.nc"
    #eta_file = None
    metric_terms = MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
        eta_file=eta_file,
    )
    grid_data = GridData.new_from_metric_terms(metric_terms)

    # create an initial state for the Rossby Wave number 4 test case
    state = ai.init_analytic_state(
        analytic_init_case="rhwave",
        grid_data=grid_data,
        quantity_factory=quantity_factory,
        adiabatic=config.adiabatic,
        hydrostatic=config.hydrostatic,
        moist_phys=config.moist_phys,
        comm=communicator,
    )
    stencil_factory = StencilFactory(
        config=stencil_config,
        grid_indexing=grid_indexing,
    )

    dycore = DynamicalCore(
        comm=communicator,
        grid_data=grid_data,
        stencil_factory=stencil_factory,
        quantity_factory=quantity_factory,
        damping_coefficients=DampingCoefficients.new_from_metric_terms(metric_terms),
        config=config,
        timestep=timedelta(seconds=config.dt_atmos),
        phis=state.phis,
        state=state,
    )

    #return dycore, state, NullTimer()
    return None, state, NullTimer()


def copy_state(state1: DycoreState, state2: DycoreState):
    # copy all attributes of state1 to state2
    for attr_name in dir(state1):
        for _field in fields(type(state1)):
            if issubclass(_field.type, Quantity):
                attr = getattr(state1, attr_name)
                if isinstance(attr, Quantity):
                    getattr(state2, attr_name).data[:] = attr.data

def plot_wind_diff(diff_data, description, filename):
    """ Generated originally from Gemini and modified """
    print(f"{description} shape: {diff_data.shape}")

    # Calculate the number of rows and columns for the subplots
    num_rows = 8
    num_cols = 10
    num_plots = num_rows * num_cols

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(40, 40))

    # Loop through each subplot and add data
    for i in range(diff_data.shape[2]): 
        print(f"{i} shape: {diff_data[:,:,i].shape}")
        row_index = i // num_cols
        col_index = i % num_cols
        ax = axes[row_index, col_index]
    
        # Generate random data for each subplot
        x_data = np.linspace(0, 10, 100)
        y_data = np.random.rand(100)
    
        # Plot the data on the subplot
        ax.imshow(diff_data[:,:,i], cmap='viridis')
        #ax.colorbar()
    
        # Set the title for the subplot
        ax.set_title(f'z={i}')

    # Adjust spacing between subplots
    plt.tight_layout()

    #plt.savefig('rhwave_diff_u.png')
    plt.savefig(filename)
    plt.clf()

def plot_2d_diff(attribute, ds_values, state_values):
    plt.title(f"Diff NetCDF vs Pace Dycore State '{attribute}'")
    diff = ds_values - state_values
    plt.imshow(diff, cmap='viridis')
    plt.colorbar()
    plt.savefig(f"rhwave_diff_{attribute}_0.png") # TODO: directory somewhere?
    plt.clf()
    
    
def test_rhwave_init_validation():
    dycore, state, timer = setup_dycore()

    # Read in netcdf file.
    # Compare results for state's values for tile 1 to SHiELD build file
    #
    validation_dir = os.path.join(PACE_DIR, "tests", "main", "data", "rhwave_validation", "zero_time_v3")
    ds = xr.open_dataset(os.path.join(validation_dir, "fv_core.res.tile1.nc"))
    
    # TODO: In theory, these should match.... but they don't yet!!!!
    # We're only looking at time 0 in the netcdf file

    ''' TODO: Do we need plots of the whole thing?
    diff_delp = ds["delp"].values[0, :].transpose(2, 1, 0) - state.delp.view[:]
    plot_wind_diff(diff_delp, "delp", "rhwave_diff_delp_all.png")    
    '''
        
    max_eps_error = 1e-10 
    plot = True # TODO: Turn this off?

    # 3D Attributes
    for attribute in ["u", "v", "delp"]:
        ds_values = ds[attribute].values[0, :].transpose(2, 1, 0)
        state_values = getattr(state, attribute).view[:]
        
        if plot==True:
            plot_2d_diff(attribute, ds_values[ :, :, 0], state_values[ :, :, 0])
        
        max_error = np.max(np.absolute(ds_values - state_values))
        print(f"{attribute} max_error: {max_error}")
        assert max_error < max_eps_error # TODO: Can I use assert_almost_equal instead?
        np.testing.assert_almost_equal(state_values, ds_values, decimal=11)

    # 2D Attributes
    for attribute in ["phis"]: # TODO: Why is "ps" in DycoreState but  missing in RESTART?
        print(f"{attribute}:")
        ds_values = ds[attribute].values[0, :].transpose(1, 0)
        print(f"ds_values (ds_values.shape): {ds_values}")
        state_values = getattr(state, attribute).view[:]
        print(f"state_values (state_values.shape): {state_values}")
        
        if plot==True:
            plot_2d_diff(attribute, ds_values, state_values)
        
        max_error = np.max(np.absolute(ds_values - state_values))
        print(f"{attribute} max_error: {max_error}")
        assert max_error < max_eps_error # TODO: Can I use assert_almost_equal instead?
        np.testing.assert_almost_equal(state_values, ds_values, decimal=11)        
        
    ''' 
    TODO: Are these okay?
    u max_error: 1.2079226507921703e-13
    v max_error: 9.947598300641403e-14
    delp max_error: 1.4551915228366852e-11
    phis max_error: 0.0

    '''

    # TODO: from fv_core.res.nc: ak, bk
    # TODO: from fv_tracer.res.tile*.nc: cl, cl2, sphum
    # TODO: from fv_srf_wnd.res.tile*.nc: u_surf, v_surf
    # TODO: need to check all tiles 1-6?

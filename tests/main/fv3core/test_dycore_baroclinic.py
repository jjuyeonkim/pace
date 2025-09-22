"""Unit tests for Jablonowski & Williamson Baroclinic test cases
Corresponds to Fortran test #12 (Steady State) and #13 (Perturbation)
found in tools/test_cases.F90 of:
https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere.git
TODO This is a place holder class for more unit tests in the future.
TODO Consider changing this into or replacing with a PyFV3 Translate test
"""

from datetime import timedelta
from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
import xarray as xr
from typing import Tuple

import pyfv3.initialization.analytic_init as ai
from ndsl import (
    CompilationConfig,
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    DaceConfig,
    GridIndexing,
    NullComm,
    QuantityFactory,
    StencilConfig,
    StencilFactory,
    SubtileGridSizer,
    TileCommunicator,
    TilePartitioner,
)
from ndsl.grid import DampingCoefficients, GridData, MetricTerms
from ndsl.performance.timer import NullTimer
from pyfv3 import DycoreState, DynamicalCore, DynamicalCoreConfig

# TODO: DON'T Check this path in!!!
BC_DIR = "/home/Janice.Kim/old/pace/tests/main/data/baroclinic"

@pytest.fixture()
def setenv_pace64(monkeypatch: pytest.MonkeyPatch):
    with mock.patch.dict(os.environ):
        monkeypatch.setenv("PACE_FLOAT_PRECISION", "64")
        yield # Restore the environment after

def setup_dycore_config(test_case=ai.AnalyticCase.baroclinic_instability) -> DynamicalCoreConfig:
    config = DynamicalCoreConfig(
        layout=(1, 1),
        npx=49,
        npy=49,
        npz=32,
        ntiles=6,
        nwat=6,
        dt_atmos=225,
        a_imp=1.0,
        beta=0.0,
        consv_te=False,  # not implemented, needs allreduce
        d2_bg=0.0,
        d2_bg_k1=0.2,
        d2_bg_k2=0.1,
        d4_bg=0.15,
        d_con=0.0,       # Default is 0 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        d_ext=0.0,
        dddmp=0.5,
        #delt_max=0.002,  # TODO: What is equiv in SHiELD_build?
        do_sat_adj=True,
        do_vort_damp=True,
        fill=True,
        hord_dp=6,
        hord_mt=6,
        hord_tm=6,
        hord_tr=8,
        hord_vt=6,
        hydrostatic=False,
        k_split=1,
        ke_bg=0.0,       # Default is 0 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        kord_mt=9,
        kord_tm=-9,
        kord_tr=9,
        kord_wz=9,
        n_split=1,
        nord=3,
        p_fac=0.05,      # Default is 0.05 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        rf_fast=True,
        rf_cutoff=3000.0,
        tau=10.0,
        vtdm4=0.06,
        z_tracer=True,
        do_qa=True,
        moist_phys=True,
    )
    return config


def setup_dycore(rank=0, usesCubedSphereComm=True, test_case=ai.AnalyticCase.baroclinic_instability) -> DycoreState:
    """Sets up Dycore state for analytic initialization"""
    backend = "numpy"
    config = setup_dycore_config(test_case=test_case)
    mpi_comm = NullComm(
        rank=rank, total_ranks=6 * config.layout[0] * config.layout[1], fill_value=0.0
    )
    partitioner = CubedSpherePartitioner(TilePartitioner(config.layout))

    if usesCubedSphereComm:
        communicator = CubedSphereCommunicator(mpi_comm, partitioner)
    else:
        communicator = TileCommunicator(mpi_comm, partitioner)

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
    eta_file = "tests/main/input/eta32.nc" # TODO: where to document file creation for developers?
    metric_terms = MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
        eta_file=eta_file,
    )
    grid_data = GridData.new_from_metric_terms(metric_terms)
    damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)

    state = ai.init_analytic_state(
        analytic_init_case=test_case,
        grid_data=grid_data,
        quantity_factory=quantity_factory,
        adiabatic=config.adiabatic,
        hydrostatic=config.hydrostatic,
        moist_phys=config.moist_phys,
        sw_dynamics=config.sw_dynamics,
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
        damping_coefficients=damping_coefficients,
        config=config,
        timestep=timedelta(seconds=config.dt_atmos),
        phis=state.phis,
        state=state,
    )
    return dycore, state, NullTimer()


def plot_2d_diff(testname, rank, attribute, ds_values, state_values, plot_dir="."):
    os.makedirs(plot_dir, exist_ok=True)
    diff = ds_values - state_values
    plt.title(f"diff: Fortran - Pace for '{attribute}'")
    plt.imshow(diff, cmap="viridis")
    plt.colorbar()
    plt.savefig(
        os.path.join(
            plot_dir,
            f"test_{testname}_diff_r{rank}_{attribute}.png"
        )
    )
    plt.clf()

    # Normalize the differences. Take the regular diff when ds_value is 0
    norm_diff = np.where(
        ds_values != 0, 
        np.absolute((ds_values - state_values) / ds_values), 
        np.absolute(ds_values - state_values)
    )
    plt.title(f"norm_diff: abs(Fortran - Pace / Fortran) for '{attribute}'")
    plt.imshow(norm_diff, cmap="viridis")
    plt.colorbar()
    plt.savefig(
        os.path.join(
            plot_dir,
            f"test_{testname}_norm_diff_r{rank}_{attribute}.png"
        )
    )
    plt.clf()


def plot_2d(desc, rank, attribute, data, plot_dir="."):
    os.makedirs(plot_dir, exist_ok=True)
    plt.title(f"{desc} - rank:{rank}, '{attribute}'")
    plt.imshow(data, cmap="viridis")
    plt.colorbar()
    plt.savefig(
        os.path.join(
            plot_dir,
            f"test_{desc}_r{rank}_{attribute}.png"
        )
    )
    plt.clf()


def check_init(data_dir,
               attributes,
               max_eps_errors,
               gen_plots=False,
               plot_dir=".",
               step=False,
               test_case=ai.AnalyticCase.baroclinic_instability,
               desc="test13_64",
               rank_range=range(0,6)):
    """TODO: doc"""
    precision = "64"
    if 'PACE_FLOAT_PRECISION' in os.environ:
        precision = os.getenv("PACE_FLOAT_PRECISION", "SHOULD_NOT_BE_USED")
    print(f"precision: {precision}")
    # jk TODO: log precision instead?
    
    for rank in rank_range:
        dycore, state, timer = setup_dycore(rank=rank, test_case=test_case)

        fortran_rank = rank + 1
        if step: 
            dycore.step_dynamics(state, timer)
        core_ds = xr.open_dataset(
            os.path.join(data_dir, f"fv_core.res.tile{fortran_rank}.nc")
        )
        for attribute, max_eps_error in zip(attributes, max_eps_errors):
            print(f"rank {rank}, attribute {attribute}")
            # Dycore values/dimensions
            state_values = getattr(state, attribute.lower()).view[:]
            state_ndims = len(getattr(state, attribute.lower()).dims)

            # Dataset values for 3D/2D Attributes at time zero
            if state_ndims == 2:  # 2D
                core_ds_values = core_ds[attribute].values[0, :].transpose(1, 0)
                core_ds_values_2d, state_values_2d = core_ds_values, state_values
            elif state_ndims == 3:  # 3D
                core_ds_values = core_ds[attribute].values[0, :].transpose(2, 1, 0)
                core_ds_values_2d, state_values_2d = (
                    core_ds_values[:, :, 0],
                    state_values[:, :, 0],
                )
            else:
                assert False, f"Unexpected number of dims in DycoreState {attribute}"

            # TODO: Remove plotting eventually
            if gen_plots:
                step_prefix = "step1_" if step else "t0_"
                plot_2d(f"{step_prefix}pace.{desc}", rank, attribute, state_values_2d, plot_dir=plot_dir)
                plot_2d(f"{step_prefix}ds.{desc}", rank, attribute, core_ds_values_2d, plot_dir=plot_dir)
                plot_2d_diff(f"{step_prefix}{desc}", rank, attribute, core_ds_values_2d, state_values_2d, plot_dir=plot_dir)

            norm_diff = np.where(
                core_ds_values != 0, 
                np.absolute((core_ds_values - state_values) / core_ds_values), 
                np.absolute(core_ds_values - state_values)
            )
            max_error_norm_diff = np.max(norm_diff) # TODO: Use this when values don't blow up...?
            max_error_diff = np.max(np.abs(core_ds_values - state_values))
            assert max_error_diff < max_eps_error

    # NOTE: The original test_cases.F90 initialized tracers for cl and cl2,
    #       but we do not initialize or check for them in this test.


def test_baroclinic_init64(setenv_pace64: None):
    """Tests case #13 (Perturbation) initialization for 64bit precision
    Compare initialized DycoreState values with ground truth net-cdf files.

    Ground truth RESTART files were generated using the following script: 
    TODO (where to put the script?)
    Using SHiELD_build 8309e1151812f72dda41142a16da0eb1f2bc4f8a (5/22/2025)
    """
    desc = "test13_64_debug_rs"
    data_dir = os.path.join(BC_DIR, desc)
    attributes = ["phis", "delp", "u", "v"] # TODO: more attributes
    max_eps_errors = [5e-12, 1e-14, 2e-12, 2e-12]
    check_init(data_dir, attributes, max_eps_errors, test_case=ai.AnalyticCase.baroclinic_instability, gen_plots=True, desc=desc, plot_dir=desc)


def test_baroclinic_12_init64(setenv_pace64: None):
    """Tests case #12 (Steady State) initialization for 64bit precision
    Compare initialized DycoreState values with ground truth net-cdf files.
    """
    desc = "test12_64_debug_rs"
    data_dir = os.path.join(BC_DIR, desc)
    attributes = ["phis", "delp", "u", "v"] # TODO: more attributes
    max_eps_errors = [5e-12, 1e-14, 2e-12, 2e-12]
    check_init(data_dir, attributes, max_eps_errors, test_case=ai.AnalyticCase.baroclinic_steady, gen_plots=True, desc=desc, plot_dir=desc)


def test_baroclinic_steady_and_instability_are_different():
    """
    Simple test to check that the steady and instability tests produce
    different results.
    """
    _, state_steady, _ = setup_dycore(test_case=ai.AnalyticCase.baroclinic_steady)
    _, state_instability, _ = setup_dycore(
        test_case=ai.AnalyticCase.baroclinic_instability
    )
    assert not (state_instability.u.field == state_steady.u.field).all()

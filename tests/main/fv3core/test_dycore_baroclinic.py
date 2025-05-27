""" Unit tests for Jablonowski & Williamson Baroclinic test cases 
Corresponds to Fortran test #12 (Steady State) and #13 (Perturbation) 
found in tools/test_cases.F90 of:

https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere.git
"""

import os
from datetime import timedelta
from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import pyFV3.initialization.analytic_init as ai
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
    TilePartitioner,
    TileCommunicator,
)
from ndsl.grid import DampingCoefficients, GridData, MetricTerms
from ndsl.performance.timer import NullTimer, Timer
from pyFV3 import DycoreState, DynamicalCore, DynamicalCoreConfig


DIR = os.path.abspath(os.path.dirname(__file__))
PACE_DIR = os.path.join(DIR, "..", "..", "..")
BC_DIR = os.path.join(PACE_DIR, "tests", "main", "data", "baroclinic")


@pytest.fixture()
def setenv_pace32(monkeypatch):
    #TODO: doublecheck monkey patch fixture?
    #Monkeypatching: https://docs.pytest.org/en/stable/how-to/monkeypatch.html
    #Fixture example found here: https://stackoverflow.com/questions/77255758/how-can-i-mock-my-environment-variables-for-my-pytest
    #https://github.com/pytest-dev/pytest/issues/4576
    #https://docs.python.org/3/library/unittest.mock.html#unittest.mock.patch.dict
    with mock.patch.dict(os.environ):
        monkeypatch.setenv("PACE_FLOAT_PRECISION", "32")
        yield # Restore the environment after 


@pytest.fixture()
def setenv_pace64(monkeypatch):
    with mock.patch.dict(os.environ):
        monkeypatch.setenv("PACE_FLOAT_PRECISION", "64")
        yield # Restore the environment after


def setup_dycore_config() -> DynamicalCoreConfig:
    config = DynamicalCoreConfig(
        layout=(1, 1),
        npx=13,
        npy=13,
        npz=79,
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


def setup_dycore(rank=0, usesCubedSphereComm=True) -> DycoreState:
    """Sets up Dycore state for Rossby analytic initialization"""
    backend = "numpy"
    config = setup_dycore_config()
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
    eta_file = "tests/main/input/eta79.nc"
    metric_terms = MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
        eta_file=eta_file,
    )
    grid_data = GridData.new_from_metric_terms(metric_terms)

    state = ai.init_analytic_state(
        analytic_init_case=ai.Cases.baroclinic.value,
        grid_data=grid_data,
        quantity_factory=quantity_factory,
        config=config,
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
    return dycore, state, NullTimer()


def plot_2d_diff(testname, rank, attribute, ds_values, state_values):
    plt.title(f"Fortran - Pace / '{attribute}'")
    diff = ds_values - state_values
    plt.imshow(diff, cmap="viridis")
    plt.colorbar()
    plt.savefig(
        f"test_{testname}_baroclinic_diff_r{rank}_{attribute}.png"
    )  # TODO: directory somewhere?
    plt.clf()


def plot_2d(desc, rank, attribute, data):
    plt.title(f"{desc} - rank:{rank}, '{attribute}'")
    plt.imshow(data, cmap="viridis")
    plt.colorbar()
    plt.savefig(
        f"test_{desc}_baroclinic_r{rank}_{attribute}.png"
    )  # TODO: directory somewhere?
    plt.clf()


def check_init(data_dir, attributes, max_eps_error, gen_plots=False, step=False):
    """TODO: doc"""

    precision = "64"
    if 'PACE_FLOAT_PRECISION' in os.environ:
        precision = os.getenv("PACE_FLOAT_PRECISION", "SHOULD_NOT_BE_USED")

    for rank in range(0, 1):
        fortran_rank = rank + 1
        dycore, state, timer = setup_dycore(rank=rank)
        if step:
            dycore.step_dynamics(state, timer)
        core_ds = xr.open_dataset(
            os.path.join(data_dir, f"fv_core.res.tile{fortran_rank}.nc")
        )
        for attribute in attributes:
            print(f"rank {rank}, attribute {attribute}")
            # Dycore values/dimensions
            state_values = getattr(state, attribute).view[:]
            state_ndims = len(getattr(state, attribute).dims)

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
                step_prefix = "step1_" if step else ""
                plot_2d(f"{step_prefix}pace-init{precision}", rank, attribute, state_values_2d)
                plot_2d(f"{step_prefix}ds-init{precision}", rank, attribute, core_ds_values_2d)
                plot_2d_diff(f"{step_prefix}init{precision}", rank, attribute, core_ds_values_2d, state_values_2d)

            max_error_diff = np.max(
                np.absolute((core_ds_values - state_values) / core_ds_values)
            )
            if np.isnan(max_error_diff): # e.g., from zero division
                max_error_diff = np.max(np.absolute((core_ds_values - state_values)))

            assert max_error_diff < max_eps_error

    # NOTE: The original test_cases.F90 initialized tracers for cl and cl2,
    #       but we do not initialize or check for them in this test.


# TODO: Why doesn't this work instead of setenv_pace64?
#@mock.patch.dict("os.environ", {"PACE_FLOAT_PRECISION", "64"})
#def test_rossby_init64():

def test_baroclinic_init64(setenv_pace64):
    """Tests case #13 (Perturbation) initialization for 64bit precision
    Compare initialized DycoreState values with ground truth net-cdf files.
    """

    data_dir = os.path.join(BC_DIR, "C96.solo.BCmoist.pace_64")

    attributes = ["u", "v"] # TODO: more attributes
    max_eps_error = 1e-13
    check_init(data_dir, attributes, max_eps_error, gen_plots=True)


def test_baroclinic_init32(setenv_pace32):
    """Tests case #13 (Perturbation) initialization for 32bit precision
    Compare initialized DycoreState values with ground truth net-cdf files.
    """

    data_dir = os.path.join(BC_DIR, "C96.solo.BCmoist.pace_32")

    # attributes = ["u", "v", "delp", "phis"]
    attributes = ["u", "v"] # TODO: more attributes
    max_eps_error = 1e-7
    check_init(data_dir, attributes, max_eps_error)


def test_baroclinic_12_init64(setenv_pace64):
    """Tests case #12 (Steady State) initialization for 64bit precision
    Compare initialized DycoreState values with ground truth net-cdf files.
    """

    data_dir = os.path.join(BC_DIR, "C96.solo.BCmoist.pace_12_64")

    attributes = ["u", "v"] # TODO: more attributes
    max_eps_error = 1e-13
    check_init(data_dir, attributes, max_eps_error, gen_plots=True)


def test_baroclinic_12_init32(setenv_pace32):
    """Tests case #12 (Steady State) initialization for 32bit precision
    Compare initialized DycoreState values with ground truth net-cdf files.
    """
    data_dir = os.path.join(BC_DIR, "C96.solo.BCmoist.pace_12_32")

    attributes = ["u", "v"] # TODO: more attributes
    max_eps_error = 1e-7
    check_init(data_dir, attributes, max_eps_error)



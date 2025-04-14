import os
import unittest.mock
from dataclasses import fields
from datetime import timedelta
from typing import Tuple
import xarray as xr
import numpy as np

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

'''
def test_temporaries_are_deterministic():
    """
    This is a precursor test to the next one, ensuring that two
    identically-initialized dycores called on identically-initialized
    states produce identical temporaries.

    This will fail if there is non-determinism in the initialization,
    for example from using `empty` instead of `zeros` to initialize data.
    """
    dycore1, state1, timer1 = setup_dycore()
    dycore2, state2, timer2 = setup_dycore()

    dycore1.step_dynamics(state1, timer1)
    first_temporaries = copy_temporaries(dycore1, max_depth=10)
    assert len(first_temporaries) > 0
    dycore2.step_dynamics(state2, timer2)
    second_temporaries = copy_temporaries(dycore2, max_depth=10)
    assert_same_temporaries(second_temporaries, first_temporaries)

def test_call_on_same_state_same_dycore_produces_same_temporaries():
    """
    Assuming the precursor test passes, this test indicates whether
    the dycore retains and re-uses internal state on subsequent calls.
    If it does not, then subsequent calls on identical input should
    produce identical results.
    """
    dycore, state_1, timer_1 = setup_dycore()
    _, state_2, timer_2 = setup_dycore()

    # state_1 and state_2 are identical, if the dycore is stateless then they
    # should produce identical dycore final states when used to call
    dycore.step_dynamics(state_1, timer_1)
    first_temporaries = copy_temporaries(dycore, max_depth=10)
    assert len(first_temporaries) > 0
    # TODO: The orchestrated code pushed us to make the dycore stateful for halo
    # exchange, so we must copy into state_1 instead of using state_2.
    # We should call with state_2 directly when this is fixed.
    copy_state(state_2, state_1)
    dycore.step_dynamics(state_1, timer_2)
    second_temporaries = copy_temporaries(dycore, max_depth=10)
    assert_same_temporaries(second_temporaries, first_temporaries)


def test_call_does_not_allocate_storages():
    dycore, state, timer = setup_dycore()

    def error_func(*args, **kwargs):
        raise AssertionError("call not allowed")

    with unittest.mock.patch("gt4py.storage.zeros", new=error_func):
        with unittest.mock.patch("gt4py.storage.empty", new=error_func):
            dycore.step_dynamics(state, timer)


def test_call_does_not_define_stencils():
    dycore, state, timer = setup_dycore()

    def error_func(*args, **kwargs):
        raise AssertionError("call not allowed")

    with unittest.mock.patch("gt4py.cartesian.gtscript.stencil", new=error_func):
        dycore.step_dynamics(state, timer)
'''

def test_validation():
    dycore, state, timer = setup_dycore()

    # Read in netcdf file.
    # Compare results for state's values for tile 1 to SHiELD build file
    #
    validation_dir = os.path.join(PACE_DIR, "tests", "main", "data", "rhwave_validation", "zero_time_v3")
    ds = xr.open_dataset(os.path.join(validation_dir, "fv_core.res.tile1.nc"))

    # Dataset
    desc = 'dataset'
    ds_u_transposed = ds["u"].values[0, :].transpose(2, 1, 0)
    print(f"{desc} u\n{ds_u_transposed.shape}\n{desc} u\n{ds_u_transposed}")
    ds_v_transposed = ds["v"].values[0, :].transpose(2, 1, 0)
    print(f"{desc} v\n{ds_v_transposed.shape}\n{desc} v\n{ds_v_transposed}")
    
    #print(f"{desc} delp\n{ds['delp'].values[0, :].shape}):\n{desc} delp\n{ds['delp'].values[0, :]}")
    
    # Dycore
    desc = 'dycore state'
    print(f"{desc} u\n{state.u.view[:].shape}\n{desc} u\n{state.u.view[:]}")
    print(f"{desc} v\n{state.v.view[:].shape}\n{desc} v\n{state.v.view[:]}")
    # TODO:  WHY IS THIS VIEW SO DIFFERENT?
    #print(f"{desc} delp\n:\n{desc} delp\n{state.delp}")
    #print(f"{desc} delp\n{state.delp.view[:].shape}):\n{desc} delp\n{state.delp.view[:]}")

    # TODO: In theory, these should match.... but they don't yet!!!!
    # We're only looking at time 0 in the netcdf file

    diff_u = np.sum((ds["u"].values[0, :].transpose(2, 1, 0) - state.u.view[:]) ** 2)
    print(f"diff_u: {diff_u}")
    diff_v = np.sum((ds["v"].values[0, :].transpose(2, 1, 0) - state.v.view[:]) ** 2)
    print(f"diff_v: {diff_v}")
    
    np.testing.assert_array_equal(
        ds["u"].values[0, :].transpose(2, 1, 0), state.u.view[:]
    )    
    np.testing.assert_array_equal(
        ds["v"].values[0, :].transpose(2, 1, 0), state.v.view[:]
    )    
    #np.testing.assert_array_equal(ds["delp"].values[0, :], state.delp.view[:])
    # TODO: ADD MORE!


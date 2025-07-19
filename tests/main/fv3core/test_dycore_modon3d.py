""" Unit tests for the 3D Modon Soliton analytic test case
Corresponds to Fortran test #45 found in tools/test_cases.F90 of:

https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere.git

TODO This is a place holder class for more unit tests in the future.
"""

from datetime import timedelta

import pyfv3.initialization.analytic_init as ai
from ndsl import (
    CompilationConfig,
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    DaceConfig,
    GridIndexing,
    Namelist,
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


def setup_dycore_config_NO_GOOD() -> DynamicalCoreConfig:
    config = DynamicalCoreConfig(
        layout=(4, 4),
        npx=48,
        npy=48,
        npz=79, # TODO: eventually set to 5?
        ntiles=6,
        nwat=6,
        dt_atmos=1200,
        #a_imp=1.0,  # not in Joseph's
        #beta=0.0,  # not in Joseph's
        consv_te=False,  # not implemented, needs allreduce
        d2_bg=0.0,
        d2_bg_k1=0.0,
        d2_bg_k2=0.0,
        d4_bg=0.08,
        d_con=0.0,  # Default is 0 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        d_ext=0.0,
        #dddmp=0.5,  # not in Joseph's
        # delt_max=0.002,  # TODO: What is equiv in SHiELD_build?
        #do_sat_adj=True, # not in Joseph's
        do_vort_damp=False,
        fill=False,
        hord_dp=8,
        hord_mt=8,
        hord_tm=8,
        hord_tr=8,
        hord_vt=8,
        hydrostatic=False, # True in Joseph's, keeping false for now
        k_split=2,
        ke_bg=0.0,  # Default is 0 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        kord_mt=9,
        kord_tm=-9,
        kord_tr=9,
        kord_wz=9,
        n_split=8,
        nord=2,
        p_fac=0.05,  # Default is 0.05 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        #rf_fast=True,  # not in Joseph's 
        # rf_cutoff=3000.0,  # not in Joseph's
        #tau=10.0,  # not in Joseph's
        vtdm4=0.00,
        #z_tracer=True, # not in Joseph's
        do_qa=True, # not in Joseph's
        # moist_phys=True, # not in Joseph's
    )
    return config

def setup_dycore_config_from_namelist() -> DynamicalCoreConfig:
    import f90nml # TODO: jk move up if needed
    from collections import OrderedDict # TODO: jk move up if needed

    namelist_path = "/home/Janice.Kim/SHiELD_dev/SCRATCH/soloCI_amdbox_FV3-202411-public/CI/BATCH-CI/3dmodon/C128.solo.modon_zeroday/rundir/input.nml"
    namelist_od = f90nml.read(namelist_path)
    nml = Namelist.from_f90nml(namelist_od)
    config_from_namelist = DynamicalCoreConfig.from_namelist(nml)

    # Pace testing tweaks:
    config_from_namelist.nwat = 6
    #config_from_namelist.hydrostatic = False # Maybe?

    # TODO: Need to reconfigure somehow: 
    # NotImplementedError: D-Grid Shallow Water Lagrangian Dynamics (D_SW): damp_vt misconfiguration, some are above a d_con of 1e-05.
    # config = DGridShallowWaterLagrangianDynamicsConfig(dddmp=0.0, d2_bg=0.0, d2_bg_k1=0.0, d2_bg_k2=0.0, d4_bg=0.08, ke_bg=0.0, nor...f3d=False, do_skeb=False, d_con=0.0, vtdm4=0.0, inline_q=False, convert_ke=False, do_vort_damp=False, hydrostatic=True

    # TODO: Need to reconfigure somehow:
    # self.acoustic_dynamics = AcousticDynamics(
    # pyFV3/pyfv3/stencils/dyn_core.py:505: in __init__
    # self.update_height_on_d_grid = updatedzd.UpdateHeightOnDGrid(...
    # damping_coefficients = DampingCoefficients(...
    #         if any(column_namelist["damp_vt"].view[:] <= Float(1e-5)):
    # >           raise NotImplementedError(
    # "damp <= 1e-5 in column_namelist is not implemented")
    # E           NotImplementedError: damp <= 1e-5 in column_namelist is not implemented

    # TODO: Maybe this isn't the best way --- maybe I should configure like rossby and baroclinic and not from namelist

    return config_from_namelist


def setup_dycore_config() -> DynamicalCoreConfig:
    config = DynamicalCoreConfig(
        layout=(1, 1),
        npx=48,
        npy=48,
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
        d_con=0.0,  # Default is 0 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        d_ext=0.0,
        dddmp=0.5,
        # delt_max=0.002,  # TODO: What is equiv in SHiELD_build?
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
        ke_bg=0.0,  # Default is 0 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        kord_mt=9,
        kord_tm=-9,
        kord_tr=9,
        kord_wz=9,
        n_split=1,
        nord=3,
        p_fac=0.05,  # Default is 0.05 in GFDL_atmos_cubed_sphere/model/fv_arrays.F90
        rf_fast=True,
        rf_cutoff=3000.0,
        tau=10.0,
        vtdm4=0.06,
        z_tracer=True,
        do_qa=True,
        moist_phys=True,
    )
    return config


def setup_dycore(
    rank=0, usesCubedSphereComm=True, test_case=ai.AnalyticCase.baroclinic_instability
) -> DycoreState:
    """Sets up Dycore state for analytic initialization"""

    backend = "numpy"
    #config = setup_dycore_config()
    config = setup_dycore_config_from_namelist()
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
    eta_file = "tests/main/input/eta5.nc"
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


def test_modon_initialization():
    """
    TODO desc
    """
    # TODO jk Testing out namelist functionality while I'm at it.
    _, state_instability, _ = setup_dycore(
        test_case=ai.AnalyticCase.modon3d
    )
    # TODO actually figure out a tests here
    pass

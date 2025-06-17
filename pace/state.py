import dataclasses
from dataclasses import fields
from typing import List

import xarray as xr

import ndsl.dsl.gt4py_utils as gt_utils
from ndsl import Quantity, QuantityFactory, SubtileGridSizer
from ndsl.constants import N_HALO_DEFAULT, X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import Float
from ndsl.filesystem import get_fs
from ndsl.grid import DampingCoefficients, DriverGridData, GridData
from ndsl.typing import Communicator
from pyFV3 import DycoreState
from pySHiELD import PHYSICS_PACKAGES, PhysicsState

# jk TODO REMOVE
import matplotlib.pyplot as plt
import os
import numpy as np

@dataclasses.dataclass()
class TendencyState:
    """
    Accumulated tendencies from physical parameterizations to be applied
    to the dynamical core model state.
    """

    u_dt: Quantity = dataclasses.field(
        metadata={
            "name": "eastward_wind_tendency_due_to_physics",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "m/s**2",
            "intent": "inout",
        }
    )
    v_dt: Quantity = dataclasses.field(
        metadata={
            "name": "northward_wind_tendency_due_to_physics",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "m/s**2",
            "intent": "inout",
        }
    )
    pt_dt: Quantity = dataclasses.field(
        metadata={
            "name": "temperature_tendency_due_to_physics",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "K/s",
            "intent": "inout",
        }
    )

    @classmethod
    def init_zeros(cls, quantity_factory: QuantityFactory) -> "TendencyState":
        initial_quantities = {}
        for _field in dataclasses.fields(cls):
            initial_quantities[_field.name] = quantity_factory.zeros(
                _field.metadata["dims"],
                _field.metadata["units"],
                dtype=Float,
            )
        return cls(**initial_quantities)


# jk TODO REMOVE plot functions
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

    norm_diff = np.absolute((ds_values - state_values) / ds_values)
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


@dataclasses.dataclass
class DriverState:
    dycore_state: DycoreState
    physics_state: PhysicsState
    tendency_state: TendencyState
    grid_data: GridData
    damping_coefficients: DampingCoefficients
    driver_grid_data: DriverGridData

    # TODO: the driver_config argument here isn't type hinted from
    # import due to a circular dependency. This can be fixed by refactoring
    # for example by moving this method into some restart.py module
    @classmethod
    def load_state_from_restart(
        cls,
        restart_path: str,
        driver_config,
        damping_coefficients: DampingCoefficients,
        driver_grid_data: DriverGridData,
        grid_data: GridData,
        schemes: List[PHYSICS_PACKAGES],
    ) -> "DriverState":
        comm = driver_config.comm_config.get_comm()
        communicator = Communicator.from_layout(comm=comm, layout=driver_config.layout)
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=driver_config.nx_tile,
            ny_tile=driver_config.nx_tile,
            nz=driver_config.nz,
            n_halo=N_HALO_DEFAULT,
            extra_dim_lengths={},
            layout=driver_config.layout,
            tile_partitioner=communicator.partitioner.tile,
            tile_rank=communicator.tile.rank,
        )
        quantity_factory = QuantityFactory.from_backend(
            sizer, backend=driver_config.stencil_config.compilation_config.backend
        )

        state = _restart_driver_state(
            restart_path,
            communicator.rank,
            quantity_factory,
            communicator,
            damping_coefficients=damping_coefficients,
            driver_grid_data=driver_grid_data,
            grid_data=grid_data,
            schemes=schemes,
        )
        return state


    def save_state(self, comm, restart_path: str = "RESTART"):
        from pathlib import Path

        Path(restart_path).mkdir(parents=True, exist_ok=True)
        current_rank = str(comm.Get_rank())
        self.dycore_state.xr_dataset.to_netcdf(
            f"{restart_path}/restart_dycore_state_{current_rank}.nc"
        )
        self.physics_state.xr_dataset.to_netcdf(
            f"{restart_path}/restart_physics_state_{current_rank}.nc"
        )
        # we can also convert the state to Fortran's restart format using
        # code similar to this commented code. We don't need this feature right
        # now so we haven't implemented it, but this is a good starter.
        """
        xr.Dataset(
            data_vars={
                "cld_amt": state.dycore_state.qcld.data_array,
                "graupel": state.dycore_state.qgraupel.data_array,
                "ice_wat": state.dycore_state.qice.data_array,
                "liq_wat": state.dycore_state.qliquid.data_array,
                "o3mr": state.dycore_state.qo3mr.data_array,
                "rainwat": state.dycore_state.qrain.data_array,
                "sgs_tke": state.dycore_state.qsgs_tke.data_array,
                "snowwat": state.dycore_state.qsnow.data_array,
                "sphum": state.dycore_state.qvapor.data_array,
            }
        ).rename(
            {
                "z": "zaxis_1",
                "x": "xaxis_1",
                "y": "yaxis_1",
            }
        ).transpose(
            "zaxis_1", "yaxis_1", "xaxis_1"
        ).expand_dims(
            dim="Time", axis=0
        ).to_netcdf(os.path.join(path, f"fv_tracer.res.tile{rank + 1}.nc"))
        """
        # jk TODO REMOVE plotting eventually (for testing only)
        rank = comm.Get_rank()
        state = self.dycore_state
        fortran_rank = rank + 1
        data_dir = '/home/Janice.Kim/pace/tests/main/data/baroclinic/pace_test13_64_debug_rs/'
        core_ds = xr.open_dataset(data_dir + f"fv_core.res.tile{fortran_rank}.nc")
        for attribute in ["u", "v", "delp", "phis"]:
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

            gen_plots = True
            if gen_plots:
                plot_2d(f"pace", rank, attribute, state_values_2d, plot_dir=restart_path)
                plot_2d(f"ds", rank, attribute, core_ds_values_2d, plot_dir=restart_path)
                plot_2d_diff(f"", rank, attribute, core_ds_values_2d, state_values_2d, plot_dir=restart_path)


def _overwrite_state_from_restart(
    path: str,
    rank: int,
    state: DycoreState,
    restart_file_prefix: str,
):
    """
    Args:
        path: path to restart files
        rank: current rank number
        state: an empty state
        restart_file_prefix: file prefix name to read
    """
    ds = xr.open_dataset(path + f"/{restart_file_prefix}_{rank}.nc")

    for _field in fields(type(state)):
        if "units" in _field.metadata.keys():
            state.__dict__[_field.name].data[:] = gt_utils.asarray(
                ds[_field.name].data[:], to_type=state.__dict__[_field.name].np.ndarray
            )


def _restart_driver_state(
    path: str,
    rank: int,
    quantity_factory: QuantityFactory,
    communicator: Communicator,
    damping_coefficients: DampingCoefficients,
    driver_grid_data: DriverGridData,
    grid_data: GridData,
    schemes: List[PHYSICS_PACKAGES],
):
    fs = get_fs(path)

    restart_files = fs.ls(path)
    is_fortran_restart = any(
        fname.endswith("fv_core.res.nc") for fname in restart_files
    )

    if is_fortran_restart:
        dycore_state = DycoreState.from_fortran_restart(
            quantity_factory=quantity_factory, communicator=communicator, path=path
        )
    else:
        dycore_state = DycoreState.init_zeros(quantity_factory=quantity_factory)
        _overwrite_state_from_restart(
            path,
            rank,
            dycore_state,
            "restart_dycore_state",
        )

    physics_state = PhysicsState.init_zeros(
        quantity_factory=quantity_factory, schemes=schemes
    )

    physics_state.__post_init__(quantity_factory, schemes)
    tendency_state = TendencyState.init_zeros(
        quantity_factory=quantity_factory,
    )

    return DriverState(
        dycore_state=dycore_state,
        physics_state=physics_state,
        tendency_state=tendency_state,
        grid_data=grid_data,
        damping_coefficients=damping_coefficients,
        driver_grid_data=driver_grid_data,
    )

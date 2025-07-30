import collections
import dataclasses
import os
import pytest
import typing
import f90nml
import yaml

import pyfv3._config
from ndsl import Namelist


CONFIG_CLASSES = [
    pyfv3._config.SatAdjustConfig,
    pyfv3._config.AcousticDynamicsConfig,
    pyfv3._config.RiemannConfig,
    pyfv3._config.DGridShallowWaterLagrangianDynamicsConfig,
    pyfv3._config.DynamicalCoreConfig,
]


@dataclasses.dataclass
class FirstConfigClass:
    value: float


@dataclasses.dataclass
class CompatibleConfigClass:
    value: float


@dataclasses.dataclass
class IncompatibleConfigClass:
    value: int


@dataclasses.dataclass
class IncompatiblePropertyConfigClass:
    @property
    def value(self) -> int:
        return 0


def assert_types_match(classes):
    types = collections.defaultdict(set)
    for cls in classes:
        for name, field in cls.__dataclass_fields__.items():
            types[name].add(field.type)
        for name, attr in cls.__dict__.items():
            if isinstance(attr, property):
                types[name].add(
                    typing.get_type_hints(attr.fget).get("return", typing.Any)
                )
    assert not any(len(type_list) > 1 for type_list in types.values()), {
        key: value for key, value in types.items() if len(value) > 1
    }


def assert_defaults_match(classes):
    types = collections.defaultdict(set)
    for cls in classes:
        for name, field in cls.__dataclass_fields__.items():
            types[name].add(field.default)
    assert not any(len(type_list) > 1 for type_list in types.values()), {
        key: value for key, value in types.items() if len(value) > 1
    }


def test_assert_types_match_compatible_types():
    assert_types_match([FirstConfigClass, CompatibleConfigClass])


def test_assert_types_match_incompatible_types():
    with pytest.raises(AssertionError):
        assert_types_match([FirstConfigClass, IncompatibleConfigClass])


def test_assert_types_match_incompatible_property_type():
    with pytest.raises(AssertionError):
        assert_types_match([FirstConfigClass, IncompatiblePropertyConfigClass])


def test_types_match():
    """
    Test that when an attribute exists on two or more configuration dataclasses,
    their type hints are the same.

    Checks both dataclass attributes and property methods.
    """
    assert_types_match(CONFIG_CLASSES)


def test_dycore_config_from_yaml():
    """ TODO """
    yaml_path = os.path.join("examples", "configs", "baroclinic_c12.yaml")
    with open(os.path.abspath(yaml_path), "r") as f:
        yaml_config = yaml.safe_load(f)
    dcconfig1 = pyfv3.DynamicalCoreConfig.from_yaml(yaml_path)

    # TODO: Is it too simplistic to do random spot checks here? Leaving it for now

    # Check matching parameters
    assert(yaml_config["dt_atmos"] == getattr(dcconfig1, "dt_atmos"))
    assert(yaml_config["nz"] == getattr(dcconfig1, "npz"))

    dycore_specific_params = ["hydrostatic"]
    for param in dycore_specific_params:
        assert(yaml_config["dycore_config"][param] == getattr(dcconfig1, param))

    # Check default parameters, not specified in the yaml
    dcconfig_default = pyfv3.DynamicalCoreConfig()

    default_params = ["adiabatic"]
    for param in default_params:
        assert(param not in yaml_config.keys())
        assert(param not in yaml_config["dycore_config"].keys())
        assert(getattr(dcconfig1, param) == getattr(dcconfig_default, param))


def test_dycore_config_from_f90nml():
    """ TODO """
    # TODO: Is it too simplistic to do random spot checks here?
    f90_namelist_path = os.path.join("examples", "configs", "baroclinic_stable_c48_input.nml")

    f90_namelist = f90nml.read(f90_namelist_path)
    dcconfig1 = pyfv3.DynamicalCoreConfig.from_f90nml(f90_namelist)

    namelist = Namelist.from_f90nml(f90_namelist)
    dcconfig2 = pyfv3.DynamicalCoreConfig.from_namelist(namelist)

    assert(dcconfig1.__dataclass_fields__ == dcconfig2.__dataclass_fields__)
    assert(dcconfig1.dt_atmos == dcconfig2.dt_atmos)
    assert(dataclasses.asdict(dcconfig1) == dataclasses.asdict(dcconfig2))

import collections
import dataclasses
import typing

import pytest

import pyfv3._config


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

def test_from_nml():
    """TODO: throw this away once you understand this better"""
    from ndsl import Namelist
    import f90nml

    f90_namelist_path = "/home/Janice.Kim/SHiELD_dev/SCRATCH/soloCI_amdbox_FV3-202411-public/CI/BATCH-CI/C48.BCmoist.pace_test12_64_debug/input.nml"

    f90_namelist = f90nml.read(f90_namelist_path)
    for key, value in f90_namelist.items():
        print(f"Key: {key}, Value: {value}")
    dcconfig1 = pyfv3.DynamicalCoreConfig.from_f90nml(f90_namelist)

    namelist = Namelist.from_f90nml(f90_namelist)
    dcconfig2 = pyfv3.DynamicalCoreConfig.from_namelist(namelist)

    # Compare the two dcconfigs
    assert(dcconfig1.__dataclass_fields__ == dcconfig2.__dataclass_fields__)
    # NOTE: If they're both the same, then we might not need the actual Namelist.from_f90nml anymore

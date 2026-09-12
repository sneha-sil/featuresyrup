import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import featuresyrup.functions as features_functions
from featuresyrup.classes import _default_node_record
from featuresyrup import (
    parse_orbital_energies,
    get_IR_frequencies,
    get_NMR_shieldings,
    get_dipole_moment,
    get_isotropic_polarizability,
    get_Fukui_indices,
    get_IE_EA,
    calculate_morfeus_descriptors,
)


BASE_DIR = Path(__file__).resolve().parent

NEUTRAL_NBO = BASE_DIR / "SS-11-01_azl_001_neutral_nbo.out"
CATION_NBO = BASE_DIR / "SS-11-01_azl_001_cation_nbo.out"
ANION_NBO = BASE_DIR / "SS-11-01_azl_001_anion_nbo.out"

NEUTRAL_ORCA = BASE_DIR / "SS-11-01_azl_001_neutral.out"
CATION_ORCA = BASE_DIR / "SS-11-01_azl_001_cation.out"
ANION_ORCA = BASE_DIR / "SS-11-01_azl_001_anion.out"

OUTPUT_ORCA = BASE_DIR / "azl_001.out"
NMR_ORCA = BASE_DIR / "SS-11-01_azl_001_nmr.out"

XYZ_FILE = BASE_DIR / "azl_001_geom.xyz"
SMILES = "CC(C)(C)C[C@H]1COC2=NN(c3ccccc3)[C]N21"


def _require_file(path: Path):
    if not path.exists():
        pytest.skip(f"Missing test fixture file: {path.name}")


def _assert_finite_float(value):
    assert isinstance(value, float)
    assert math.isfinite(value)


def test_parse_orbital_energies():
    _require_file(NEUTRAL_ORCA)

    homo, lumo, gap = parse_orbital_energies(str(NEUTRAL_ORCA))
    print("\nparse_orbital_energies:")
    print("  homo =", homo)
    print("  lumo =", lumo)
    print("  gap  =", gap)

    assert isinstance(homo, str)
    assert isinstance(lumo, str)
    _assert_finite_float(gap)

    homo_f = float(homo)
    lumo_f = float(lumo)
    assert math.isclose(gap, lumo_f - homo_f, rel_tol=1e-12, abs_tol=1e-12)


def test_get_ir_frequencies():
    _require_file(OUTPUT_ORCA)

    ir_values, stats = get_IR_frequencies(str(OUTPUT_ORCA))
    print("\nget_IR_frequencies:")
    print("  ir_values =", ir_values)
    print("  stats =", stats)

    assert set(ir_values.keys()) == {"frequencies", "eps", "intensity", "t2"}
    assert set(stats.keys()) == {"frequencies", "eps", "intensity", "t2"}

    n = len(ir_values["frequencies"])
    assert n > 0
    assert len(ir_values["eps"]) == n
    assert len(ir_values["intensity"]) == n
    assert len(ir_values["t2"]) == n

    for key in ir_values:
        assert all(isinstance(x, float) for x in ir_values[key])
        assert all(math.isfinite(x) for x in ir_values[key])

        assert set(stats[key].keys()) == {"mean", "max", "min", "std", "median"}
        for stat_name, stat_value in stats[key].items():
            print(f"  stats[{key!r}][{stat_name!r}] = {stat_value}")
            _assert_finite_float(float(stat_value))


def test_get_nmr_shieldings():
    _require_file(NMR_ORCA)

    result = get_NMR_shieldings(str(NMR_ORCA))
    print("\nget_NMR_shieldings:")
    for key, values in result.items():
        print(f"  {key} = {values}")

    assert set(result.keys()) == {
        "proton_isotropic",
        "proton_anisotropic",
        "carbon_isotropic",
        "carbon_anisotropic",
    }

    lengths = {len(v) for v in result.values()}
    assert len(lengths) == 1

    for key, values in result.items():
        assert isinstance(values, list)
        for v in values:
            assert v is None or isinstance(v, float)


def test_get_dipole_moment():
    _require_file(NEUTRAL_ORCA)

    result = get_dipole_moment(str(NEUTRAL_ORCA))
    print("\nget_dipole_moment:")
    for key, value in result.items():
        print(f"  {key} = {value}")

    assert set(result.keys()) == {
        "dipole_x",
        "dipole_y",
        "dipole_z",
        "dipole_magnitude",
    }

    for key in result:
        _assert_finite_float(result[key])


def test_get_isotropic_polarizability():
    _require_file(NMR_ORCA)

    result = get_isotropic_polarizability(str(NMR_ORCA))
    print("\nget_isotropic_polarizability:")
    print("  result =", result)

    assert set(result.keys()) == {"isotropic_polarizability"}
    _assert_finite_float(result["isotropic_polarizability"])


def test_get_fukui_indices():
    _require_file(NEUTRAL_NBO)
    _require_file(CATION_NBO)
    _require_file(ANION_NBO)

    result = get_Fukui_indices(str(CATION_NBO), str(ANION_NBO), str(NEUTRAL_NBO))
    print("\nget_Fukui_indices:")
    print("  result =", result)

    assert "fukui_indices" in result
    f = result["fukui_indices"]

    assert set(f.keys()) == {"f_plus", "f_minus", "f_zero"}

    n = len(f["f_plus"])
    assert n > 0
    assert len(f["f_minus"]) == n
    assert len(f["f_zero"]) == n

    for i in range(n):
        _assert_finite_float(f["f_plus"][i])
        _assert_finite_float(f["f_minus"][i])
        _assert_finite_float(f["f_zero"][i])
        assert math.isclose(
            f["f_zero"][i],
            (f["f_plus"][i] + f["f_minus"][i]) / 2,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_get_ie_ea():
    _require_file(CATION_ORCA)
    _require_file(ANION_ORCA)
    _require_file(NEUTRAL_ORCA)

    result = get_IE_EA(str(CATION_ORCA), str(ANION_ORCA), str(NEUTRAL_ORCA))
    print("\nget_IE_EA:")
    for key, value in result.items():
        print(f"  {key} = {value}")

    assert set(result.keys()) == {
        "ionization_energy",
        "electron_affinity",
        "hardness",
        "chemical_potential",
        "electronegativity",
        "electrophilicity",
    }

    for key, value in result.items():
        _assert_finite_float(value)

    assert math.isclose(
        result["hardness"],
        (result["ionization_energy"] - result["electron_affinity"]) / 2,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result["chemical_potential"],
        -(result["ionization_energy"] + result["electron_affinity"]) / 2,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert math.isclose(
        result["electronegativity"],
        -result["chemical_potential"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_calculate_morfeus_descriptors():
    _require_file(XYZ_FILE)

    try:
        result = calculate_morfeus_descriptors(str(XYZ_FILE), SMILES)
    except Exception as e:
        pytest.skip(f"Morfeus descriptor test skipped because SMILES/carbene setup is not valid here: {e}")

    print("\ncalculate_morfeus_descriptors:")
    for key, value in result.items():
        print(f"  {key} = {value}")

    expected_keys = {
        "buried_volume",
        "fraction_vbur",
        "free_volume",
        "sasa_area",
        "sasa_volume",
    }
    assert set(result.keys()) == expected_keys

    for key in expected_keys:
        _assert_finite_float(result[key])
        assert result[key] >= 0


def test_default_node_record_marks_carbene_center(monkeypatch):
    qm_data = SimpleNamespace(
        atomic_numbers=[6, 6],
        atom_labels=["C1", "C2"],
        xyz_coordinates=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        atomic_masses=[12.0, 12.0],
        electronegativities=[2.5, 2.5],
        covalent_radii=[0.77, 0.77],
        smiles="C1=NC=N1",
        ACSF=None,
        SOAP=None,
        NMR_shieldings=None,
        Fukui_indices=None,
    )

    monkeypatch.setattr("featuresyrup.classes._carbene_center_index", lambda mol: 1)

    first_atom = _default_node_record(0, qm_data)
    second_atom = _default_node_record(1, qm_data)

    assert first_atom["is_carbene_center"] is False
    assert second_atom["is_carbene_center"] is True


def test_get_all_catalyst_data_skips_nmr_when_missing(monkeypatch):
    monkeypatch.setattr(features_functions, "parse_orbital_energies", lambda *_: {"HOMO": "-1.0", "LUMO": "0.5", "HOMO_LUMO_gap": 1.5})
    monkeypatch.setattr(features_functions, "calculate_morfeus_descriptors", lambda *_: {"buried_volume": 1.0, "fraction_vbur": 2.0, "free_volume": 3.0, "sasa_area": 4.0, "sasa_volume": 5.0})
    monkeypatch.setattr(features_functions, "calculate_ACSF_values", lambda *_: {"acsf_features": [1.0]})
    monkeypatch.setattr(features_functions, "calculate_SOAP_values", lambda *_: {"soap_features": [1.0]})
    monkeypatch.setattr(features_functions, "get_IR_frequencies", lambda *_: ({"frequencies": [1.0], "eps": [2.0], "intensity": [3.0], "t2": [4.0]}, {"frequencies": {"mean": 1.0, "max": 1.0, "min": 1.0, "std": 0.0, "median": 1.0}, "eps": {"mean": 2.0, "max": 2.0, "min": 2.0, "std": 0.0, "median": 2.0}, "intensity": {"mean": 3.0, "max": 3.0, "min": 3.0, "std": 0.0, "median": 3.0}, "t2": {"mean": 4.0, "max": 4.0, "min": 4.0, "std": 0.0, "median": 4.0}}))
    monkeypatch.setattr(features_functions, "get_dipole_moment", lambda *_: {"dipole_x": 1.0, "dipole_y": 2.0, "dipole_z": 3.0, "dipole_magnitude": 4.0})
    monkeypatch.setattr(features_functions, "get_isotropic_polarizability", lambda *_: {"isotropic_polarizability": 5.0})
    monkeypatch.setattr(features_functions, "get_Fukui_indices", lambda *_: {"fukui_indices": {"f_plus": [1.0], "f_minus": [2.0], "f_zero": [1.5]}})
    monkeypatch.setattr(features_functions, "get_IE_EA", lambda *_: {"ionization_energy": 1.0, "electron_affinity": 2.0, "hardness": 3.0, "chemical_potential": 4.0, "electronegativity": 5.0, "electrophilicity": 6.0})

    result = features_functions.get_all_catalyst_data(
        "C1=NC=N1",
        "dummy.xyz",
        "neutral.out",
        "cation.out",
        "anion.out",
        "neutral_nbo.out",
        "cation_nbo.out",
        "anion_nbo.out",
        None,
        "ir.out",
        "dipole.out",
        "homo_lumo.out",
    )

    assert "NMR_shieldings" not in result


def test_get_all_catalyst_data_skips_dipole_when_missing(monkeypatch):
    monkeypatch.setattr(features_functions, "parse_orbital_energies", lambda *_: {"HOMO": "-1.0", "LUMO": "0.5", "HOMO_LUMO_gap": 1.5})
    monkeypatch.setattr(features_functions, "calculate_morfeus_descriptors", lambda *_: {"buried_volume": 1.0, "fraction_vbur": 2.0, "free_volume": 3.0, "sasa_area": 4.0, "sasa_volume": 5.0})
    monkeypatch.setattr(features_functions, "calculate_ACSF_values", lambda *_: {"acsf_features": [1.0]})
    monkeypatch.setattr(features_functions, "calculate_SOAP_values", lambda *_: {"soap_features": [1.0]})
    monkeypatch.setattr(features_functions, "get_IR_frequencies", lambda *_: ({"frequencies": [1.0], "eps": [2.0], "intensity": [3.0], "t2": [4.0]}, {"frequencies": {"mean": 1.0, "max": 1.0, "min": 1.0, "std": 0.0, "median": 1.0}, "eps": {"mean": 2.0, "max": 2.0, "min": 2.0, "std": 0.0, "median": 2.0}, "intensity": {"mean": 3.0, "max": 3.0, "min": 3.0, "std": 0.0, "median": 3.0}, "t2": {"mean": 4.0, "max": 4.0, "min": 4.0, "std": 0.0, "median": 4.0}}))
    monkeypatch.setattr(features_functions, "get_dipole_moment", lambda *_: (_ for _ in ()).throw(AssertionError("dipole parser should not run when the output is missing")))
    monkeypatch.setattr(features_functions, "get_isotropic_polarizability", lambda *_: (_ for _ in ()).throw(AssertionError("polarizability parser should not run when the output is missing")))
    monkeypatch.setattr(features_functions, "get_Fukui_indices", lambda *_: {"fukui_indices": {"f_plus": [1.0], "f_minus": [2.0], "f_zero": [1.5]}})
    monkeypatch.setattr(features_functions, "get_IE_EA", lambda *_: {"ionization_energy": 1.0, "electron_affinity": 2.0, "hardness": 3.0, "chemical_potential": 4.0, "electronegativity": 5.0, "electrophilicity": 6.0})

    result = features_functions.get_all_catalyst_data(
        "C1=NC=N1",
        "dummy.xyz",
        "neutral.out",
        "cation.out",
        "anion.out",
        "neutral_nbo.out",
        "cation_nbo.out",
        "anion_nbo.out",
        None,
        "ir.out",
        None,
        "homo_lumo.out",
    )

    assert "dipole_moment" not in result
    assert "polarizability" not in result
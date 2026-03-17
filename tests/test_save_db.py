import shutil
import pytest
from pathlib import Path

from mchem.fileformats import load_pdb, read_pdb_box
from mchem.forcefield import ForceField
from mchem.system import Box, System

DATA_DIR = Path(__file__).resolve().parent / "data"
TMP_DIR = Path(__file__).resolve().parent / "tmp"
FF_NAME = "amoebabio18.xml"


def _pdb_to_db(pdb_path: Path, db_path: Path):
    top = load_pdb(str(pdb_path))
    ff = ForceField(FF_NAME)
    system = ff.createSystem(top)
    box = read_pdb_box(str(pdb_path))
    if box is not None:
        system.addTerm(Box(*box), "box")
    system.save(str(db_path), overwrite=True)

    reloaded = System(str(db_path))
    assert len(reloaded.data) > 0, "No term tables in saved database"
    assert len(reloaded.meta) > 0, "No metadata in saved database"
    for name, terms in reloaded.data.items():
        assert len(terms) > 0, f"Table '{name}' is empty"
    return reloaded


@pytest.fixture(scope="module", autouse=True)
def setup_tmp_dir():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir()
    yield
    # shutil.rmtree(TMP_DIR)


def test_save_water216():
    system = _pdb_to_db(DATA_DIR / "water216.pdb", TMP_DIR / "water216.db")
    assert "Particle" in system.data
    assert len(system["Particle"]) == 648
    assert "box" not in system.data  # water216.pdb has no CRYST1


def test_save_dhfr():
    system = _pdb_to_db(DATA_DIR / "DHFR.pdb", TMP_DIR / "DHFR.db")
    assert "Particle" in system.data
    assert len(system["Particle"]) == 23558
    assert "box" in system.data
    assert len(system["box"]) == 1
    box = system["box"][0]
    assert box.a == box.b == box.c == pytest.approx(61.645)
    assert box.alpha == box.beta == box.gamma == pytest.approx(90.0)


def test_save_aladi_water_box():
    system = _pdb_to_db(DATA_DIR / "aladi_water_box.pdb", TMP_DIR / "aladi_water_box.db")
    assert "Particle" in system.data
    assert len(system["Particle"]) > 0
    assert "box" in system.data
    assert len(system["box"]) == 1
    box = system["box"][0]
    assert box.a == box.b == box.c == pytest.approx(30.0)
    assert box.alpha == box.beta == box.gamma == pytest.approx(90.0)


def test_save_dhfr_torsiontorsion():
    system = _pdb_to_db(DATA_DIR / "DHFR.pdb", TMP_DIR / "DHFR_tt.db")
    assert "AmoebaTorsionTorsion" in system.data
    assert len(system["AmoebaTorsionTorsion"]) > 0
    assert "AmoebaTorsionTorsionGrid" in system.data
    assert len(system["AmoebaTorsionTorsionGrid"]) > 0

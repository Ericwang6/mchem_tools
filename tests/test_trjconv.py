import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mchem.trjconv import (
    DBTopologyProvider,
    HDF5TrajectoryReader,
    trjconv,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "generate_hdf5")
DB_PATH = os.path.join(DATA_DIR, "aladi_water_box.db")
H5_PATH = os.path.join(DATA_DIR, "md_traj_1.h5")

NATOMS = 2602
LAST_FRAME = 10


def _hdf5_frame_coords(frame: int) -> np.ndarray:
    """Read coordinates for a given 1-based frame directly from the HDF5 file."""
    with h5py.File(H5_PATH, "r") as f:
        grp = f[f"job/1/molecular_dynamics/time_step/{frame}"]
        return np.asarray(grp["coordinates"][:], dtype=float).T


def _read_pdb_atoms(path: str):
    """Return (name, resname, resnum, x, y, z) tuples from a PDB file."""
    atoms = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                atoms.append(
                    (
                        line[12:16].strip(),
                        line[17:20].strip(),
                        int(line[22:26]),
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )
                )
    return atoms


def _read_cryst1(path: str):
    with open(path) as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                return (
                    float(line[6:15]),
                    float(line[15:24]),
                    float(line[24:33]),
                )
    return None


def test_db_topology_provider_atoms():
    provider = DBTopologyProvider(DB_PATH)
    assert provider.natoms == NATOMS
    first = provider.atoms[0]
    assert (first.idx, first.name, first.element, first.resname) == (
        0, "CH3", "C", "ACE",
    )


def test_hdf5_reader_frames():
    with HDF5TrajectoryReader(H5_PATH) as reader:
        assert reader.natoms == NATOMS
        assert reader.n_frames == LAST_FRAME
        assert reader.frame_indices == list(range(1, LAST_FRAME + 1))
        frame = reader.read_frame(1)
        assert frame.coordinates.shape == (NATOMS, 3)
        assert frame.box == (30.0, 30.0, 30.0)


def test_trjconv_default_last_frame(tmp_path):
    out = tmp_path / "last.pdb"
    written = trjconv(DB_PATH, H5_PATH, str(out))
    assert written.index == LAST_FRAME

    atoms = _read_pdb_atoms(str(out))
    assert len(atoms) == NATOMS

    expected = _hdf5_frame_coords(LAST_FRAME)
    written_coords = np.array([a[3:] for a in atoms])
    np.testing.assert_allclose(written_coords, expected, atol=1e-3)

    assert _read_cryst1(str(out)) == (30.0, 30.0, 30.0)


def test_trjconv_explicit_frame(tmp_path):
    out = tmp_path / "frame1.pdb"
    written = trjconv(DB_PATH, H5_PATH, str(out), frame=1)
    assert written.index == 1

    atoms = _read_pdb_atoms(str(out))
    written_coords = np.array([a[3:] for a in atoms])
    np.testing.assert_allclose(written_coords, _hdf5_frame_coords(1), atol=1e-3)

    # First atom metadata comes from the topology.
    assert atoms[0][0] == "CH3"
    assert atoms[0][1] == "ACE"


def test_trjconv_invalid_frame(tmp_path):
    out = tmp_path / "bad.pdb"
    with pytest.raises(IndexError):
        trjconv(DB_PATH, H5_PATH, str(out), frame=999)


def test_trjconv_unsupported_output(tmp_path):
    out = tmp_path / "bad.xtc"
    with pytest.raises(ValueError):
        trjconv(DB_PATH, H5_PATH, str(out))

"""Tests for solvation (Python API and CLI)."""

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mchem.fileformats import load_pdb, write_pdb
from mchem.fileformats.pdb import _box_vectors_to_lengths_angles
from mchem.solvate import (
    solvate,
    BoxShape,
    POSITIVE_IONS,
    NEGATIVE_IONS,
    _compute_box_vectors,
    _compute_charge,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
CLI_CMD = [sys.executable, "-m", "mchem.main"]
HAS_PACKMOL = shutil.which("packmol") is not None

requires_packmol = pytest.mark.skipif(
    not HAS_PACKMOL, reason="packmol not installed"
)


# ---------------------------------------------------------------------------
# Unit tests: box vectors
# ---------------------------------------------------------------------------

class TestComputeBoxVectors:
    def test_cube(self):
        w = 30.0
        v1, v2, v3 = _compute_box_vectors(w, "cube")
        np.testing.assert_allclose(v1, [w, 0, 0])
        np.testing.assert_allclose(v2, [0, w, 0])
        np.testing.assert_allclose(v3, [0, 0, w])

    def test_dodecahedron(self):
        w = 30.0
        v1, v2, v3 = _compute_box_vectors(w, "dodecahedron")
        np.testing.assert_allclose(v1, [w, 0, 0])
        np.testing.assert_allclose(v2, [0, w, 0])
        np.testing.assert_allclose(v3, [0.5 * w, 0.5 * w, 0.5 * math.sqrt(2) * w])
        vol_diag = v1[0] * v2[1] * v3[2]
        vol_det = np.linalg.det(np.array([v1, v2, v3]))
        assert abs(vol_diag - vol_det) < 1e-10

    def test_octahedron(self):
        w = 30.0
        v1, v2, v3 = _compute_box_vectors(w, "octahedron")
        np.testing.assert_allclose(v1, [w, 0, 0])
        np.testing.assert_allclose(
            v2, [w / 3.0, 2.0 * math.sqrt(2) * w / 3.0, 0.0]
        )
        np.testing.assert_allclose(
            v3,
            [-w / 3.0, math.sqrt(2) * w / 3.0, math.sqrt(6) * w / 3.0],
        )
        vol_diag = v1[0] * v2[1] * v3[2]
        vol_det = np.linalg.det(np.array([v1, v2, v3]))
        assert abs(vol_diag - vol_det) < 1e-10

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="Unknown box shape"):
            _compute_box_vectors(30.0, "sphere")


# ---------------------------------------------------------------------------
# Unit tests: charge calculation
# ---------------------------------------------------------------------------

class TestChargeCalculation:
    def test_ace_ala_nme_neutral(self):
        top = load_pdb(str(DATA_DIR / "ace_ala_nme.pdb"))
        charge = _compute_charge(top)
        assert charge == 0


# ---------------------------------------------------------------------------
# Unit tests: write_pdb / CRYST1
# ---------------------------------------------------------------------------

class TestWritePdb:
    def test_cryst1_cube(self):
        top = load_pdb(str(DATA_DIR / "ace_ala_nme.pdb"))
        positions = top.coordinates
        vectors = _compute_box_vectors(30.0, "cube")

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_path = f.name
        try:
            write_pdb(tmp_path, top, positions, box_vectors=vectors)
            with open(tmp_path) as f:
                first_line = f.readline()
            assert first_line.startswith("CRYST1")
            a = float(first_line[6:15])
            b = float(first_line[15:24])
            c = float(first_line[24:33])
            alpha = float(first_line[33:40])
            beta = float(first_line[40:47])
            gamma = float(first_line[47:54])
            assert abs(a - 30.0) < 0.01
            assert abs(b - 30.0) < 0.01
            assert abs(c - 30.0) < 0.01
            assert abs(alpha - 90.0) < 0.01
            assert abs(beta - 90.0) < 0.01
            assert abs(gamma - 90.0) < 0.01
        finally:
            os.unlink(tmp_path)

    def test_cryst1_dodecahedron(self):
        vectors = _compute_box_vectors(30.0, "dodecahedron")
        a, b, c, alpha, beta, gamma = _box_vectors_to_lengths_angles(vectors)
        assert abs(a - 30.0) < 0.01
        assert abs(b - 30.0) < 0.01
        assert abs(alpha - 60.0) < 0.5 or abs(alpha - 90.0) < 0.5

    def test_cryst1_octahedron(self):
        vectors = _compute_box_vectors(30.0, "octahedron")
        a, b, c, alpha, beta, gamma = _box_vectors_to_lengths_angles(vectors)
        assert a > 0 and b > 0 and c > 0

    def test_no_cryst1_without_vectors(self):
        top = load_pdb(str(DATA_DIR / "ace_ala_nme.pdb"))
        positions = top.coordinates
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_path = f.name
        try:
            write_pdb(tmp_path, top, positions)
            with open(tmp_path) as f:
                first_line = f.readline()
            assert not first_line.startswith("CRYST1")
        finally:
            os.unlink(tmp_path)

    def test_roundtrip(self):
        """Write then re-read: atom count must match."""
        top = load_pdb(str(DATA_DIR / "ace_ala_nme.pdb"))
        positions = top.coordinates
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_path = f.name
        try:
            write_pdb(tmp_path, top, positions)
            top2 = load_pdb(tmp_path)
            assert top2.numAtoms == top.numAtoms
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Integration tests: solvation (require Packmol)
# ---------------------------------------------------------------------------

@requires_packmol
class TestSolvateIntegration:
    def test_solvate_cube(self):
        top = load_pdb(str(DATA_DIR / "ace_ala_nme.pdb"))
        positions = top.coordinates
        n_solute = top.numAtoms

        solv_top, solv_pos, box_vectors = solvate(
            top, positions, box_shape="cube", buffer=10.0
        )
        assert solv_top.numAtoms > n_solute
        assert solv_pos.shape == (solv_top.numAtoms, 3)
        a, b, c, alpha, beta, gamma = _box_vectors_to_lengths_angles(box_vectors)
        assert abs(alpha - 90.0) < 0.01
        assert abs(beta - 90.0) < 0.01
        assert abs(gamma - 90.0) < 0.01

    def test_solvate_dodecahedron(self):
        top = load_pdb(str(DATA_DIR / "ace_ala_nme.pdb"))
        positions = top.coordinates

        solv_top, solv_pos, box_vectors = solvate(
            top, positions, box_shape="dodecahedron", buffer=10.0
        )
        assert solv_top.numAtoms > top.numAtoms

    def test_solvate_octahedron(self):
        top = load_pdb(str(DATA_DIR / "ace_ala_nme.pdb"))
        positions = top.coordinates

        solv_top, solv_pos, box_vectors = solvate(
            top, positions, box_shape="octahedron", buffer=10.0
        )
        assert solv_top.numAtoms > top.numAtoms

    def test_solvate_with_ionic_strength(self):
        top = load_pdb(str(DATA_DIR / "ace_ala_nme.pdb"))
        positions = top.coordinates

        solv_top, solv_pos, box_vectors = solvate(
            top,
            positions,
            box_shape="cube",
            buffer=10.0,
            neutralize=True,
            ionic_strength=0.15,
        )
        assert solv_top.numAtoms > top.numAtoms
        res_names = [r.name for r in solv_top.residues]
        assert "NA" in res_names or "Cl" in res_names

    def test_solvate_write_and_reload(self):
        """Solvate, write PDB, re-read: atom counts must match."""
        top = load_pdb(str(DATA_DIR / "ace_ala_nme.pdb"))
        positions = top.coordinates

        solv_top, solv_pos, box_vectors = solvate(
            top, positions, box_shape="cube", buffer=10.0
        )
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_path = f.name
        try:
            write_pdb(tmp_path, solv_top, solv_pos, box_vectors=box_vectors)
            reloaded = load_pdb(tmp_path)
            assert reloaded.numAtoms == solv_top.numAtoms
        finally:
            os.unlink(tmp_path)

    def test_solvate_then_parameterize(self):
        """Ultimate test: solvate then parameterize with amoebabio18_solvate.xml."""
        from mchem.forcefield import ForceField

        top = load_pdb(str(DATA_DIR / "ace_ala_nme.pdb"))
        positions = top.coordinates

        solv_top, solv_pos, box_vectors = solvate(
            top,
            positions,
            box_shape="cube",
            buffer=10.0,
            ionic_strength=0.15,
        )
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_path = f.name
        try:
            write_pdb(tmp_path, solv_top, solv_pos, box_vectors=box_vectors)
            reloaded = load_pdb(tmp_path)
            ff = ForceField("amoebabio18_solvate.xml")
            system = ff.createSystem(reloaded)
            assert system is not None
            assert "Multipole" in system.data
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

class TestCli:
    def test_help(self):
        result = subprocess.run(
            CLI_CMD + ["--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert "solvate" in result.stdout
        assert "convert" in result.stdout

    def test_solvate_help(self):
        result = subprocess.run(
            CLI_CMD + ["solvate", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert "buffer" in result.stdout
        assert "box-shape" in result.stdout

    @requires_packmol
    def test_solvate_invocation(self):
        inp = str(DATA_DIR / "ace_ala_nme.pdb")
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            out = f.name
        try:
            result = subprocess.run(
                CLI_CMD + [
                    "solvate",
                    "-i", inp,
                    "-o", out,
                    "--box-shape", "cube",
                    "--buffer", "10.0",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parent.parent,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            assert os.path.exists(out)
            assert os.path.getsize(out) > 0
            with open(out) as f:
                first_line = f.readline()
            assert first_line.startswith("CRYST1")
        finally:
            if os.path.exists(out):
                os.unlink(out)

"""
Solvation: add water and ions to a solute in a periodic box.

Uses Packmol as the external placement engine and computes box dimensions
following OpenMM's :meth:`openmm.app.Modeller.addSolvent` conventions.
"""

import logging
import math
import os
import subprocess
import tempfile
from typing import Literal, Tuple

import numpy as np

from .fileformats import load_pdb, write_pdb
from .topology import Topology

logger = logging.getLogger(__name__)

BoxShape = Literal["cube", "dodecahedron", "octahedron"]
"""Allowed periodic box shapes."""

POSITIVE_IONS = ("Cs+", "K+", "Li+", "Na+", "Rb+")
NEGATIVE_IONS = ("Cl-", "Br-", "F-", "I-")

ION_NAME_MAP = {
    "Li+": "LI",
    "Na+": "NA",
    "K+":  "K",
    "Rb+": "RB",
    "Cs+": "CS",
    "F-":  "F",
    "Cl-": "Cl",
    "Br-": "Br",
    "I-":  "I",
}

_WATER_PDB = """\
ATOM      1  O   HOH A   1       4.125  13.679  13.761  1.00  0.00           O
ATOM      2  H1  HOH A   1       4.025  14.428  14.348  1.00  0.00           H
ATOM      3  H2  HOH A   1       4.670  13.062  14.249  1.00  0.00           H
END
"""

_ION_PDB = {
    "LI": "HETATM    1  LI   LI A   1       0.000   0.000   0.000  1.00  0.00          Li\nEND\n",
    "NA": "HETATM    1  NA   NA A   1       0.000   0.000   0.000  1.00  0.00          Na\nEND\n",
    "K":  "HETATM    1  K     K A   1       0.000   0.000   0.000  1.00  0.00           K\nEND\n",
    "RB": "HETATM    1  RB   RB A   1       0.000   0.000   0.000  1.00  0.00          Rb\nEND\n",
    "CS": "HETATM    1  CS   CS A   1       0.000   0.000   0.000  1.00  0.00          Cs\nEND\n",
    "F":  "HETATM    1  F     F A   1       0.000   0.000   0.000  1.00  0.00           F\nEND\n",
    "Cl": "HETATM    1  Cl   Cl A   1       0.000   0.000   0.000  1.00  0.00          Cl\nEND\n",
    "Br": "HETATM    1  Br   Br A   1       0.000   0.000   0.000  1.00  0.00          Br\nEND\n",
    "I":  "HETATM    1  I     I A   1       0.000   0.000   0.000  1.00  0.00           I\nEND\n",
}

# Bulk water number density (molecules/A^3)
_WATER_DENSITY = 0.0334


def _compute_box_vectors(
    width: float, box_shape: BoxShape
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute periodic box vectors in reduced (lower-triangular) form.

    Parameters
    ----------
    width : float
        Box width in Angstroms.
    box_shape : BoxShape
        One of ``"cube"``, ``"dodecahedron"``, or ``"octahedron"``.

    Returns
    -------
    tuple of np.ndarray
        Three box vectors ``(v1, v2, v3)``.
    """
    if box_shape == "cube":
        v1 = np.array([width, 0.0, 0.0])
        v2 = np.array([0.0, width, 0.0])
        v3 = np.array([0.0, 0.0, width])
    elif box_shape == "dodecahedron":
        v1 = np.array([width, 0.0, 0.0])
        v2 = np.array([0.0, width, 0.0])
        v3 = np.array([0.5 * width, 0.5 * width, 0.5 * math.sqrt(2) * width])
    elif box_shape == "octahedron":
        v1 = np.array([width, 0.0, 0.0])
        v2 = np.array([width / 3.0, 2.0 * math.sqrt(2) * width / 3.0, 0.0])
        v3 = np.array([
            -width / 3.0,
            math.sqrt(2) * width / 3.0,
            math.sqrt(6) * width / 3.0,
        ])
    else:
        raise ValueError(f"Unknown box shape: {box_shape!r}")
    return v1, v2, v3


def _compute_charge(topology: Topology) -> int:
    """Parameterize the solute and return the rounded net monopole charge."""
    from .forcefield import ForceField

    ff = ForceField("amoebabio18_solvate.xml")
    system = ff.createSystem(topology)
    multipoles = system.data["Multipole"]
    total = sum(term.c0 for term in multipoles)
    return int(math.floor(total + 0.5))


def _write_solute_pdb(topology: Topology, positions: np.ndarray, path: str) -> None:
    """Write solute atoms as a plain PDB (no CRYST1)."""
    write_pdb(path, topology, positions)


def solvate(
    topology: Topology,
    positions: np.ndarray,
    *,
    box_shape: BoxShape = "cube",
    buffer: float = 10.0,
    neutralize: bool = True,
    ionic_strength: float = 0.0,
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
) -> Tuple[Topology, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Solvate a solute with water and optional ions in a periodic box.

    Parameters
    ----------
    topology : Topology
        Solute topology (no solvent).
    positions : np.ndarray
        Atom positions, shape ``(N, 3)``, in Angstroms.
    box_shape : BoxShape, optional
        Periodic box shape. Default is ``"cube"``.
    buffer : float, optional
        Minimum padding in Angstroms between solute and box edge.
        Default is ``10.0``.
    neutralize : bool, optional
        If True, add counterions to neutralize the system. Default is True.
    ionic_strength : float, optional
        Target ionic strength in mol/L. Default is ``0.0``.
    positive_ion : str, optional
        Positive ion type (e.g. ``"Na+"``). Default is ``"Na+"``.
    negative_ion : str, optional
        Negative ion type (e.g. ``"Cl-"``). Default is ``"Cl-"``.

    Returns
    -------
    tuple
        ``(topology, positions, box_vectors)`` where *box_vectors* is a tuple
        of three ``np.ndarray`` vectors.

    Raises
    ------
    ValueError
        If an unsupported ion type is given.
    RuntimeError
        If Packmol fails.
    """
    pos_ion_res = ION_NAME_MAP.get(positive_ion)
    neg_ion_res = ION_NAME_MAP.get(negative_ion)
    if pos_ion_res is None:
        raise ValueError(f"Unsupported positive ion: {positive_ion!r}")
    if neg_ion_res is None:
        raise ValueError(f"Unsupported negative ion: {negative_ion!r}")

    # --- charge calculation ---
    total_charge = _compute_charge(topology)
    logger.info("System net charge: %d", total_charge)

    # --- bounding sphere and box width ---
    center = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    radius = float(np.max(np.linalg.norm(positions - center, axis=1)))
    width = max(2.0 * radius + buffer, 2.0 * buffer)

    vectors = _compute_box_vectors(width, box_shape)
    bbox = np.array([vectors[0][0], vectors[1][1], vectors[2][2]])
    logger.info("Box shape: %s, width: %.3f A", box_shape, width)
    logger.info("Box vectors: v1=%s, v2=%s, v3=%s", vectors[0], vectors[1], vectors[2])
    logger.info("Orthorhombic box: %.3f x %.3f x %.3f A", *bbox)

    # --- shift solute to orthorhombic box center ---
    box_center = bbox / 2.0
    shift = box_center - center
    shifted_positions = positions + shift

    # --- estimate water count ---
    cell_volume = float(bbox[0] * bbox[1] * bbox[2])
    solute_volume = (4.0 / 3.0) * math.pi * radius ** 3
    n_water = int((cell_volume - solute_volume) * _WATER_DENSITY)

    # --- ion counts ---
    n_positive = 0
    n_negative = 0
    if neutralize:
        if total_charge > 0:
            n_negative += total_charge
        else:
            n_positive += abs(total_charge)

    n_pairs = 0
    if ionic_strength > 0:
        n_pairs = int(math.floor(
            (n_water - n_positive - n_negative) * ionic_strength / 55.4 + 0.5
        ))
        n_positive += n_pairs
        n_negative += n_pairs

    n_water -= (n_positive + n_negative)
    logger.info("Water molecules: %d", n_water)
    logger.info("Positive ions (%s): %d", positive_ion, n_positive)
    logger.info("Negative ions (%s): %d", negative_ion, n_negative)
    if (n_water + n_positive + n_negative) > 0:
        effective_ionic = n_pairs * 55.4 / (n_water + n_positive + n_negative)
    else:
        effective_ionic = 0.0
    logger.info("Effective ionic strength: %.4f M", effective_ionic)

    # --- run Packmol ---
    with tempfile.TemporaryDirectory() as tmpdir:
        solute_path = os.path.join(tmpdir, "solute.pdb")
        water_path = os.path.join(tmpdir, "water.pdb")
        output_path = os.path.join(tmpdir, "output.pdb")
        _write_solute_pdb(topology, shifted_positions, solute_path)

        with open(water_path, "w") as f:
            f.write(_WATER_PDB)

        inp_lines = [
            "tolerance 2.0",
            "filetype pdb",
            f"output {output_path}",
            "",
            f"structure {solute_path}",
            "  number 1",
            "  fixed 0. 0. 0. 0. 0. 0.",
            "end structure",
            "",
            f"structure {water_path}",
            f"  number {n_water}",
            f"  inside box 0. 0. 0. {bbox[0]:.3f} {bbox[1]:.3f} {bbox[2]:.3f}",
            "end structure",
        ]

        for ion_res, ion_count, label in [
            (pos_ion_res, n_positive, "positive"),
            (neg_ion_res, n_negative, "negative"),
        ]:
            if ion_count > 0:
                ion_path = os.path.join(tmpdir, f"{ion_res}.pdb")
                with open(ion_path, "w") as f:
                    f.write(_ION_PDB[ion_res])
                inp_lines += [
                    "",
                    f"structure {ion_path}",
                    f"  number {ion_count}",
                    f"  inside box 0. 0. 0. {bbox[0]:.3f} {bbox[1]:.3f} {bbox[2]:.3f}",
                    "end structure",
                ]

        inp_content = "\n".join(inp_lines) + "\n"
        inp_path = os.path.join(tmpdir, "packmol.inp")
        with open(inp_path, "w") as f:
            f.write(inp_content)
        logger.info("Running Packmol...")

        result = subprocess.run(
            ["packmol", "-i", inp_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0 or "Success!" not in result.stdout:
            raise RuntimeError(
                f"Packmol failed (exit code {result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )
        logger.info("Packmol completed successfully")

        solvated_top = load_pdb(output_path)

    solvated_positions = solvated_top.coordinates
    return solvated_top, solvated_positions, vectors

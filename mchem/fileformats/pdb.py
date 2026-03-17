"""PDB file I/O: read/write :class:`Topology` from/to ATOM/HETATM records."""

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..topology import Bond, Atom, Topology, Residue


def read_pdb_box(fname: os.PathLike) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    Read periodic box dimensions from the first CRYST1 record in a PDB file.

    Parameters
    ----------
    fname : os.PathLike
        Path to the PDB file.

    Returns
    -------
    tuple of (a, b, c, alpha, beta, gamma) or None
        Cell lengths a, b, c in Angstroms and angles alpha, beta, gamma in
        degrees. None if no CRYST1 record is present.
    """
    with open(fname, "r") as f:
        for line in f:
            if line[:6].strip() == "CRYST1":
                a = float(line[6:15])
                b = float(line[15:24])
                c = float(line[24:33])
                alpha = float(line[33:40])
                beta = float(line[40:47])
                gamma = float(line[47:54])
                return (a, b, c, alpha, beta, gamma)
    return None


def load_pdb(fname: os.PathLike) -> Topology:
    """
    Load a topology from a PDB file (ATOM/HETATM), match residue templates, and return editable topology.

    Parameters
    ----------
    fname : os.PathLike
        Path to the PDB file.

    Returns
    -------
    Topology
        Topology with residues and atoms; templates are matched so bonds and standard names are set.
    """
    top = Topology(name=Path(fname).stem)
    residues = []
    with open(fname, 'r') as f:
        prevSig = ""
        for line in f:
            record = line[:6].strip()
            if record == "ATOM" or record == "HETATM":
                atIdx = int(line[6:11])
                atName = line[12:16].strip()
                altLoc = line[16].strip()
                resName = line[17:20].strip()
                chain = line[21]
                resNum = int(line[22:26])
                iCode = line[26]
                xx = float(line[30:38])
                xy = float(line[38:46])
                xz = float(line[46:54])
                occ = float(line[54:60])
                tempFactor = float(line[60:66])
                element = line[76:78].strip()
                
                try:
                    charge = float(line[78:80])
                except ValueError:
                    charge = 0.0
                
                sig = f"{resName}{resNum}{iCode}/{chain}"
                if sig != prevSig:
                    residues.append(Residue(resName, resNum, chain, iCode))
                if (not altLoc) or (altLoc == "A"):
                    res = residues[-1]
                    atom = Atom(atName, element)
                    atom.setPosition([xx, xy, xz])
                    res.addAtom(atom)
                prevSig = sig
    # TODO (Eric): Add processing bonds from CONECT record
    with top.setEditable():
        for res in residues:
            top.addResidue(res)
        top.matchTemplates()
    
    return top


# Single-atom residues that get HETATM records
_HETATM_RESIDUES = {"NA", "K", "Cl", "LI", "CS", "RB", "F", "Br", "I",
                    "BE", "MG", "CA", "ZN"}


def _box_vectors_to_lengths_angles(
    vectors: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[float, float, float, float, float, float]:
    """Convert box vectors to (a, b, c, alpha, beta, gamma) in Angstroms/degrees."""
    v1, v2, v3 = [np.asarray(v, dtype=float) for v in vectors]
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    c = np.linalg.norm(v3)
    alpha = math.degrees(math.acos(np.clip(np.dot(v2, v3) / (b * c), -1, 1)))
    beta = math.degrees(math.acos(np.clip(np.dot(v3, v1) / (c * a), -1, 1)))
    gamma = math.degrees(math.acos(np.clip(np.dot(v1, v2) / (a * b), -1, 1)))
    return a, b, c, alpha, beta, gamma


def write_pdb(
    fname: os.PathLike,
    topology: Topology,
    positions: np.ndarray,
    box_vectors: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    """
    Write a PDB file from a topology and positions.

    Parameters
    ----------
    fname : os.PathLike
        Output PDB file path.
    topology : Topology
        Molecular topology.
    positions : np.ndarray
        Atom positions, shape ``(N, 3)``, in Angstroms.
    box_vectors : tuple of np.ndarray, optional
        Three box vectors ``(v1, v2, v3)``.  If provided, a ``CRYST1`` record
        is written at the top of the file.
    """
    with open(fname, "w") as f:
        if box_vectors is not None:
            a, b, c, alpha, beta, gamma = _box_vectors_to_lengths_angles(box_vectors)
            f.write(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
                    f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n")

        atom_serial = 1
        for res in topology.residues:
            record = "HETATM" if res.name in _HETATM_RESIDUES else "ATOM  "
            for atom in res.atoms:
                x, y, z = positions[atom.idx]
                name_field = f" {atom.name:<3s}" if len(atom.name) < 4 else atom.name
                f.write(
                    f"{record}{atom_serial:5d} {name_field:4s}"
                    f" {res.name:>3s} {res.chain}{res.number:4d}"
                    f"    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {atom.symbol:>2s}\n"
                )
                atom_serial += 1
        f.write("END\n")

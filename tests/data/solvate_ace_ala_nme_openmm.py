#!/opt/anaconda3/envs/llama/bin/python
"""
Generate reference solvated systems using OpenMM Modeller.addSolvent.

Reads ace_ala_nme.pdb from this directory, solvates with 2.5 nm buffer,
0.15 M ionic strength, for cube, dodecahedron, and octahedron box shapes.
Centers the system in the box and writes PDB (and CRYST1) to this directory.

Requires: openmm (e.g. conda install -c conda-forge openmm).
"""

import numpy as np
from pathlib import Path

from openmm import unit
from openmm.app import ForceField, Modeller, PDBFile

DATA_DIR = Path(__file__).resolve().parent
INPUT_PDB = DATA_DIR / "ace_ala_nme.pdb"

PADDING = 2.5 * unit.nanometers
IONIC_STRENGTH = 0.15 * unit.molar
BOX_SHAPES = ("cube", "dodecahedron", "octahedron")


def _center_in_box(positions, box_vectors):
    """Shift positions so the geometric center of the system is at the box center."""
    if hasattr(positions, "value_in_unit"):
        pos_nm = positions.value_in_unit(unit.nanometers)
    else:
        pos_nm = np.array([[p.x, p.y, p.z] for p in positions])
    pos_nm = np.asarray(pos_nm)
    current_center = pos_nm.mean(axis=0)
    v1, v2, v3 = box_vectors
    box_center = (
        np.array([v1.x, v1.y, v1.z])
        + np.array([v2.x, v2.y, v2.z])
        + np.array([v3.x, v3.y, v3.z])
    ) / 2
    shift = box_center - current_center
    new_pos_nm = pos_nm + shift
    return unit.Quantity(new_pos_nm, unit.nanometers)


def main() -> None:
    if not INPUT_PDB.exists():
        raise FileNotFoundError(f"Input PDB not found: {INPUT_PDB}")

    pdb = PDBFile(str(INPUT_PDB))
    forcefield = ForceField("amber99sb.xml", "tip3p.xml")

    for box_shape in BOX_SHAPES:
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=PADDING,
            boxShape=box_shape,
            ionicStrength=IONIC_STRENGTH,
            neutralize=True,
        )
        topology = modeller.getTopology()
        positions = modeller.getPositions()
        box = topology.getPeriodicBoxVectors()
        if box is not None:
            positions = _center_in_box(positions, box)
        out_name = f"ace_ala_nme_solvated_openmm_{box_shape}.pdb"
        out_path = DATA_DIR / out_name
        with open(out_path, "w") as f:
            PDBFile.writeFile(topology, positions, f, keepIds=True)
        n_atoms = topology.getNumAtoms()
        if box is not None:
            print(f"Wrote {out_path} ({n_atoms} atoms, box vectors set, centered)")
        else:
            print(f"Wrote {out_path} ({n_atoms} atoms)")


if __name__ == "__main__":
    main()

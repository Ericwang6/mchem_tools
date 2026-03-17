"""
AMOEBA energy decomposition for the DHFR benchmark system.

Loads DHFR.pdb, applies the AMOEBA 2018 force field with PME,
and reports the energy of each force component along with all
PME / cutoff settings.
"""

import json
from pathlib import Path

import openmm as mm
import openmm.app as app
from openmm.unit import nanometer, kilojoule_per_mole, angstrom


DATA_DIR = Path(__file__).resolve().parent / "data"
PDB_PATH = DATA_DIR / "DHFR.pdb"
OUTPUT_PATH = DATA_DIR / "dhfr_amoeba_energy.json"

CUTOFF = 7.0 * angstrom
POLAR_METHOD = mm.AmoebaMultipoleForce.Mutual
EWALD_TOL = 1e-5
VDW_CUTOFF = 9.0 * angstrom


def main():
    pdb = app.PDBFile(str(PDB_PATH))
    ff = app.ForceField("amoeba2018.xml")

    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=CUTOFF,
        ewaldErrorTolerance=EWALD_TOL,
        polarization="mutual",
        mutualInducedTargetEpsilon=1e-5,
        vdwCutoff=VDW_CUTOFF,
        constraints=None,
    )

    force_names = []
    for idx, force in enumerate(system.getForces()):
        force.setForceGroup(idx)
        name = force.getName()
        force_names.append(name)

    integrator = mm.VerletIntegrator(0.001)
    platform = mm.Platform.getPlatformByName("CPU")
    context = mm.Context(system, integrator, platform)
    context.setPositions(pdb.positions)

    # ── Collect PME / cutoff settings ──────────────────────────────
    _POLAR_NAMES = {0: "Mutual", 1: "Direct", 2: "Extrapolated"}
    _VDW_POT = {0: "Buffered-14-7", 1: "Lennard-Jones"}

    settings = {}
    for force in system.getForces():
        if isinstance(force, mm.AmoebaMultipoleForce):
            alpha, nx, ny, nz = force.getPMEParametersInContext(context)
            cutoff_nm = force.getCutoffDistance() / nanometer
            settings["multipole_cutoff_nm"] = cutoff_nm
            settings["multipole_PME_alpha_per_nm"] = alpha
            settings["multipole_PME_nx"] = nx
            settings["multipole_PME_ny"] = ny
            settings["multipole_PME_nz"] = nz
            settings["ewald_error_tolerance"] = force.getEwaldErrorTolerance()
            pol_id = force.getPolarizationType()
            settings["polarization_type"] = _POLAR_NAMES.get(pol_id, str(pol_id))
        elif isinstance(force, mm.AmoebaVdwForce):
            vdw_cut_nm = force.getCutoffDistance() / nanometer
            settings["vdw_cutoff_nm"] = vdw_cut_nm
            pot_id = force.getPotentialFunction()
            settings["vdw_potential_function"] = _VDW_POT.get(pot_id, str(pot_id))
            settings["vdw_sigma_combining_rule"] = force.getSigmaCombiningRule()
            settings["vdw_epsilon_combining_rule"] = force.getEpsilonCombiningRule()
            settings["vdw_dispersion_correction"] = force.getUseDispersionCorrection()

    # ── Total energy ───────────────────────────────────────────────
    state = context.getState(getEnergy=True)
    total_energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    # ── Per-force energy decomposition ─────────────────────────────
    energies = {}
    for idx, name in enumerate(force_names):
        st = context.getState(getEnergy=True, groups=2**idx)
        e = st.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        energies[name] = e

    # ── Print results ──────────────────────────────────────────────
    print("=" * 60)
    print("AMOEBA Energy Decomposition – DHFR")
    print("=" * 60)
    print(f"\nPDB file          : {PDB_PATH}")
    print(f"Number of atoms   : {system.getNumParticles()}")
    print()

    print("─── Settings ───")
    for k, v in settings.items():
        print(f"  {k:40s} = {v}")
    print()

    print("─── Energy Components (kJ/mol) ───")
    for name, e in energies.items():
        print(f"  {name:40s} = {e:20.6f}")
    print()
    print(f"  {'Total':40s} = {total_energy:20.6f}")
    print("=" * 60)

    # ── Write output JSON ──────────────────────────────────────────
    output = {
        "pdb_file": str(PDB_PATH),
        "num_atoms": system.getNumParticles(),
        "settings": settings,
        "energies_kJ_per_mol": energies,
        "total_energy_kJ_per_mol": total_energy,
    }
    with open(OUTPUT_PATH, "w") as fp:
        json.dump(output, fp, indent=2)
    print(f"\nOutput written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

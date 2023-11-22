import pytest
from pathlib import Path

from mchem.system import System
from mchem.common import TermList
from mchem.terms import AmoebaBond, CMAPTable


def test_system():
    path = Path(__file__).resolve().parent

    system = System()
    system.addMeta("name", "test")
    system.addMeta("date", "08/26/2023")
    
    bonds = TermList(AmoebaBond)
    bonds.append(AmoebaBond(0, 1, 1.0, 2.0, -1.0, -2.0))
    bonds.append(AmoebaBond(1, 2, 1.0, 2.0, -1.0, -2.0))
    system.addTerms(bonds)

    cmapTable = TermList(CMAPTable)
    cmapTable.append(CMAPTable(phi=-180.0, psi=-180.0, ene=0.0))
    cmapTable.append(CMAPTable(phi=-180.0, psi=-150.0, ene=0.1))
    system.addTerms(cmapTable, name="CMAPTable1")

    db = path / "test.db"
    system.save(db, overwrite=True)

    systemRead = System(db)
    assert len(systemRead['AmoebaBond']) == 2
    assert len(systemRead['CMAPTable1']) == 2
    db.unlink()


def test_water_system():
    from mchem.terms.bonded import AmoebaBond, AmoebaAngle 
    from mchem.terms.nonbonded import Particle, AmoebaVdw147, Multipole, IsotropicPolarization, MultipoleAxisType
    bCubic = -2.55
    bQuartic = 3.793125
    aCubic = -0.014
    aQuartic = 0.000056
    aPentic = -0.0000007
    aSextic = 0.000000022

    water = System()
    water.addMeta("name", "water")
    water.addMeta("date", "09/06/2023")

    particles = TermList(Particle)
    particles.append(Particle(0, "OW", "O", 15.9994, 0, "HOH", 0.0, 0.0, 0.0))
    particles.append(Particle(1, "HW1", "H", 1.0079, 0, "HOH", 0.0, 0.0, 1.0))
    particles.append(Particle(2, "HW2", "H", 1.0079, 0, "HOH", 0.0, 0.0, -1.0))
    water.addTerms(particles)

    bonds = TermList(AmoebaBond)
    bonds.append(AmoebaBond(0, 1, 0.9572, 556.85, bCubic, bQuartic))
    bonds.append(AmoebaBond(0, 2, 0.9572, 556.85, bCubic, bQuartic))
    water.addTerms(bonds)

    angles = TermList(AmoebaAngle)
    angles.append(AmoebaAngle(1, 0, 2, 108.50, 48.70, aCubic, aQuartic, aPentic, aSextic))
    water.addTerms(angles)

    multipoles = TermList(Multipole)
    multipoles.append(Multipole(0, -0.51966, 0.0, 0.0, 0.14279, 0.37928, 0.0, 0.0, -0.41809, 0.0, 0.03881, MultipoleAxisType.Bisector.value, -1, -2))
    multipoles.append(Multipole(1, 0.25983, -0.03859, 0.0, -0.05818, -0.03673, 0.0, -0.00203, -0.10739, 0.0, 0.14412, MultipoleAxisType.ZOnly.value, 0, 2))
    multipoles.append(Multipole(2, 0.25983, -0.03859, 0.0, -0.05818, -0.03673, 0.0, -0.00203, -0.10739, 0.0, 0.14412, MultipoleAxisType.ZOnly.value, 0, 1))
    water.addTerms(multipoles)

    pols = TermList(IsotropicPolarization)
    pols.append(IsotropicPolarization(0, 0.8370, 0.3900, [0, 1, 2]))
    pols.append(IsotropicPolarization(1, 0.4960, 0.3900, [0, 1, 2]))
    pols.append(IsotropicPolarization(2, 0.4960, 0.3900, [0, 1, 2]))
    water.addTerms(pols)

    vdws = TermList(AmoebaVdw147)
    vdws.append(AmoebaVdw147(0, 3.4050, 0.1100))
    vdws.append(AmoebaVdw147(1, 2.6550, 0.0135, 0, 0.910))
    vdws.append(AmoebaVdw147(2, 2.6550, 0.0135, 0, 0.910))
    water.addTerms(vdws)

    water.save("./water.db", overwrite=True)


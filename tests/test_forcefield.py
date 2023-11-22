import pytest
from mchem.system import System
from mchem.fileformats import load_pdb
from mchem.forcefield import ForceField

import openmm as mm
import openmm.app as app

from pprint import pprint


def generate_ref_omm_sys(path) -> mm.System:
    ff = app.ForceField("amoeba2018.xml")
    pdb = app.PDBFile(path)
    system = ff.createSystem(pdb.topology)
    return system


def generate_mchem_sys(path) -> System:
    ff = ForceField("amoebabio18.xml")
    top = load_pdb(path)
    system = ff.createSystem(top)
    return system


def _test_bond(ommSystem: mm.System, mchemSystem: System):
    bondsRef = []
    for force in ommSystem.getForces():
        if isinstance(force, mm.CustomBondForce):
            numBonds = force.getNumBonds()
            for i in range(numBonds):
                param = force.getBondParameters(i)
                bondsRef.append((param[0], param[1]))
            break
    
    bonds = [(bo.p0, bo.p1) for bo in mchemSystem.getTermsByName("AmoebaBond")]
    assert len(bonds) == len(bondsRef), "Number of bonds does not match"
    for bo in bonds:
        assert bo in bondsRef or tuple(reversed(bo)) in bondsRef, f"Bond {bo} not in ref"


def _test_angle(ommSystem: mm.System, mchemSystem: System):
    anglesRef = []
    for force in ommSystem.getForces():
        if isinstance(force, mm.CustomAngleForce):
            numAngles = force.getNumAngles()
            for i in range(numAngles):
                param = force.getAngleParameters(i)
                anglesRef.append((param[0], param[1], param[2]))
        
    angles = [(an.p0, an.p1, an.p2) for an in mchemSystem.getTermsByName("AmoebaAngle")]
    assert len(angles) == len(anglesRef), "Number of angles does not match"
    for angle in angles:
        assert angle in anglesRef or tuple(reversed(angle)) in anglesRef, f"Angle {angle} not in ref"


def _test_angle_in_plane(ommSystem: mm.System, mchemSystem: System):
    anglesInPlaneRef = []
    for force in ommSystem.getForces():
        if force.getName() == "AmoebaInPlaneAngle":
            num = force.getNumBonds()
            for i in range(num):
                param = force.getBondParameters(i)
                anglesInPlaneRef.append(param[0])
    
    anglesInPlane = [(an.p0, an.p1, an.p2, an.p3) for an in mchemSystem.getTermsByName("AmoebaAngleInPlane")]
    assert len(anglesInPlaneRef) == len(anglesInPlane), "Number of in-plane angles does not match"
    for angle in anglesInPlane:
        assert angle in anglesInPlaneRef or (angle[2], angle[1], angle[0], angle[3]) in anglesInPlaneRef, f"AngleInPlane {angle} not in ref"


def _test_ub(ommSystem: mm.System, mchemSystem: System):
    ubsRef = []
    for force in ommSystem.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            numUb = force.getNumBonds()
            for i in range(numUb):
                param = force.getBondParameters(i)
                ubsRef.append((param[0], param[1]))
    
    ubs = [(u.p0, u.p2) for u in mchemSystem.getTermsByName("AmoebaUreyBradley")]
    assert len(ubsRef) == len(ubs), "Number of Urey-Bradley does not match"
    for ub in ubs:
        assert ub in ubsRef or tuple(reversed(ub)) in ubsRef, f"Urey-Bradley {ub} not in ref"
        

def _test_strbnd(ommSystem: mm.System, mchemSystem: System):
    strbndsRef = []
    for force in ommSystem.getForces():
        if force.getName() == "AmoebaStretchBend":
            for i in range(force.getNumBonds()):
                param = force.getBondParameters(i)
                strbndsRef.append(param[0])
    
    strbnds = [(sb.p0, sb.p1, sb.p2) for sb in mchemSystem.getTermsByName("AmoebaStretchBend")]
    assert len(strbnds) == len(strbndsRef), "Number of Stetch-Bend Bend does not match"
    for sb in strbnds:
        assert sb in strbndsRef or tuple(reversed(sb)) in strbndsRef, f"Stretch-Bend {sb} not in ref"


def _test_oop(ommSystem: mm.System, mchemSystem: System):
    oopsRef = []
    for force in ommSystem.getForces():
        if force.getName() == "AmoebaOutOfPlaneBend":
            for i in range(force.getNumBonds()):
                param = force.getBondParameters(i)
                oopsRef.append(param[0])
    
    oops = [(oop.p0, oop.p1, oop.p2, oop.p3) for oop in mchemSystem.getTermsByName("AmoebaOutOfPlaneBend")]
    assert len(oops) == len(oopsRef), "Number of OutOfPlaneBend does not match"
    for oop in oops:
        assert oop in oopsRef or (oop[0], oop[1], oop[3], oop[2]) in oopsRef, f"OutOfPlaneBend {oop} not in ref"


def _test_pitor(ommSystem: mm.System, mchemSystem: System):
    pitorsRef = []
    for force in ommSystem.getForces():
        if force.getName() == "AmoebaPiTorsion":
            for i in range(force.getNumBonds()):
                param = force.getBondParameters(i)
                pitorsRef.append(param[0])
    
    pitors = [(pt.p0, pt.p1, pt.p2, pt.p3, pt.p4, pt.p5) for pt in mchemSystem.getTermsByName("AmoebaPiTorsion")]
    assert len(pitors) == len(pitorsRef), "Number of PiTorsion does not match"
    for pt in pitors:
        combinations = [
            pt,
            (pt[1], pt[0], pt[2], pt[3], pt[4], pt[5]),
            (pt[0], pt[1], pt[2], pt[3], pt[5], pt[4]),
            (pt[1], pt[0], pt[2], pt[3], pt[5], pt[4]),
            (pt[5], pt[4], pt[3], pt[2], pt[1], pt[0]),
            (pt[5], pt[4], pt[3], pt[2], pt[0], pt[1]),
            (pt[4], pt[5], pt[3], pt[2], pt[1], pt[0]),
            (pt[4], pt[5], pt[3], pt[2], pt[0], pt[1])
        ]
        assert any([p in pitorsRef for p in combinations]), f"PiTorsion {pt} not in ref"
        

def _test_torsion(ommSystem: mm.System, mchemSystem: System):
    torsRef = []
    for force in ommSystem.getForces():
        if isinstance(force, mm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                param = force.getTorsionParameters(i)
                torsRef.append((param[0], param[1], param[2], param[3], param[4]))
    
    tors = []
    for term in mchemSystem.getTermsByName("PeriodicTorsion"):
        for i in range(1, 7):
            if getattr(term, f'k{i}') != 0.0:
                tors.append((term.p0, term.p1, term.p2, term.p3, i))
    
    assert len(torsRef) == len(tors), "Number of Torsions does not match"
    for tor in tors:
        assert tor in torsRef or (tor[3], tor[2], tor[1], tor[0], tor[4]) in torsRef, f"Torsion {tor} not in ref"


def _test_multipoles(ommSystem: mm.System, mchemSystem: System):
    for force in ommSystem.getForces():
        if isinstance(force, mm.AmoebaMultipoleForce):
            break
    for term in mchemSystem.getTermsByName("Multipole"):
        param = force.getMultipoleParameters(term.idx)
        axisType, kz, kx, ky = param[3], param[4], param[5], param[6]
        assert term.axistype == axisType and term.kz == kz and term.kx == kx and term.ky == ky, f"Multipole {term.idx} not match ref"


def test_mchem():
    path = "tests/data/ace_ala_nme_water.pdb"
    ommSystem = generate_ref_omm_sys(path)
    mchemSystem = generate_mchem_sys(path)
    _test_bond(ommSystem, mchemSystem)
    _test_angle(ommSystem, mchemSystem)
    _test_angle_in_plane(ommSystem, mchemSystem)
    _test_ub(ommSystem, mchemSystem)
    _test_strbnd(ommSystem, mchemSystem)
    _test_oop(ommSystem, mchemSystem)
    _test_pitor(ommSystem, mchemSystem)
    _test_multipoles(ommSystem, mchemSystem)
    _test_torsion(ommSystem, mchemSystem)
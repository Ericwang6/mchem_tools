import numpy as np
from .base import Term
from dataclasses import dataclass

@dataclass
class HarmonicBond(Term):
    """Class for a bond with normal harmonic potential"""
    p0: int
    p1: int
    b0: float
    kb: float
    

@dataclass
class AmoebaBond(Term):
    """Class for a bond in AMOEBA force field, i.e. up to quartic polynomials"""
    p0: int
    p1: int
    b0: float
    kb: float
    cubic: float
    quartic: float


@dataclass
class HarmonicAngle(Term):
    """Class for an angle with harmonic potential"""
    p0: int
    p1: int
    p2: int
    th0: float
    kth: float
    

@dataclass
class AmoebaAngle(Term):
    """Class for an angle in AMOEBA force field (up to sextic polynomials)"""
    p0: int
    p1: int
    p2: int
    th0: float
    kth: float
    cubic: float
    quartic: float
    pentic: float
    sextic: float
    

@dataclass
class AmoebaAngleInPlane(Term):
    """Class for an in-plane angle in AMOEBA force field (up to sextic polynomials)"""
    p0: int
    p1: int
    p2: int
    p3: int # auxliary atom index for evaluate projected angle
    th0: float
    kth: float
    cubic: float
    quartic: float
    pentic: float
    sextic: float


@dataclass
class AmoebaStretchBend(Term):
    """Class for stretch-bend coupling term in AMOEBA force field"""
    p0: int
    p1: int
    p2: int
    th0: float
    b01: float
    b02: float
    kb1: float
    kb2: float
    

@dataclass
class AmoebaUreyBradley(Term):
    """Class for Urey-Bradley term used in AMOEBA force field"""
    p0: int
    p1: int
    p2: int
    r0: float
    fc: float
    

@dataclass
class AmoebaOutOfPlaneBend(Term):
    """Class for out-of-plane bending term in AMOEBA force field"""
    p0: int
    p1: int
    p2: int
    p3: int
    fc: float
    

@dataclass
class PeriodicTorsion(Term):
    p0: int
    p1: int
    p2: int
    p3: int
    phase1: float
    phase2: float
    phase3: float
    phase4: float
    phase5: float
    phase6: float
    k1: float
    k2: float
    k3: float
    k4: float
    k5: float
    k6: float


@dataclass
class AmoebaStretchTorsion(Term):
    """Class for AMOEBA stretch-torsion coupling term"""
    p0: int
    p1: int
    p2: int
    p3: int
    k11: float
    k12: float
    k13: float
    k21: float
    k22: float
    k23: float
    k31: float
    k32: float
    k33: float
    b01: float
    b02: float
    b03: float
    phi01: float
    phi02: float
    phi03: float


@dataclass
class AmoebaAngleTorsion(Term):
    """Class for AMOEBA angle-torsion coupling term"""
    p0: int
    p1: int
    p2: int
    p3: int
    k11: float
    k12: float
    k13: float
    k21: float
    k22: float
    k23: float
    th01: float
    th02: float
    phi01: float
    phi02: float
    phi03: float


@dataclass
class AmoebaPiTorsion(Term):
    """
    Class for pi-torsion term in Amoeba
    """
    p0: int
    p1: int
    p2: int
    p3: int
    p4: int
    p5: int
    k: float


@dataclass
class CMAPTable(Term):
    """Class for a CMAP tabulated torsion-torsion coupling term"""
    phi: float
    psi: float
    ene: float


@dataclass
class CMAP(Term):
    """Class for CMAP torsion-torsion coupling term"""
    cmap: int
    p0: int
    p1: int
    p2: int
    p3: int
    p4: int
    p5: int
    p6: int
    p7: int
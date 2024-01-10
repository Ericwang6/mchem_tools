import numpy as np
from dataclasses import dataclass


@dataclass
class HarmonicBond:
    """Class for a bond with normal harmonic potential"""
    p0: int
    p1: int
    b0: float
    kb: float
    paramIdx: int = -1
    

@dataclass
class AmoebaBond:
    """Class for a bond in AMOEBA force field, i.e. up to quartic polynomials"""
    p0: int
    p1: int
    b0: float
    kb: float
    cubic: float
    quartic: float
    paramIdx: int = -1


@dataclass
class HarmonicAngle:
    """Class for an angle with harmonic potential"""
    p0: int
    p1: int
    p2: int
    th0: float
    kth: float
    paramIdx: int = -1
    

@dataclass
class AmoebaAngle:
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
    paramIdx: int = -1
    

@dataclass
class AmoebaAngleInPlane:
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
    paramIdx: int = -1


@dataclass
class AmoebaStretchBend:
    """Class for stretch-bend coupling term in AMOEBA force field"""
    p0: int
    p1: int
    p2: int
    th0: float
    b01: float
    b02: float
    kb1: float
    kb2: float
    paramIdx: int = -1
    

@dataclass
class AmoebaUreyBradley:
    """Class for Urey-Bradley term used in AMOEBA force field"""
    p0: int
    p1: int
    p2: int
    r0: float
    fc: float
    paramIdx: int = -1
    

@dataclass
class AmoebaOutOfPlaneBend:
    """Class for out-of-plane bending term in AMOEBA force field"""
    p0: int
    p1: int
    p2: int
    p3: int
    fc: float
    paramIdx: int = -1


@dataclass
class PeriodicTorsion:
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
    paramIdx: int = -1


@dataclass
class AmoebaStretchTorsion:
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
class AmoebaAngleTorsion:
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
    paramIdx: int = -1


@dataclass
class AmoebaPiTorsion:
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
    paramIdx: int = -1


@dataclass
class CMAPTable:
    """Class for a CMAP tabulated torsion-torsion coupling term"""
    phi: float
    psi: float
    ene: float
    paramIdx: int = -1


@dataclass
class CMAP:
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
from dataclasses import dataclass, field
from enum import Enum


class MultipoleAxisType(Enum):
    ZThenX            = 0
    Bisector          = 1
    ZBisect           = 2
    ThreeFold         = 3
    ZOnly             = 4
    NoAxisType        = 5
    LastAxisTypeIndex = 6

MultipoleAxisTypeInt2Str = {
    at.value: at.name for at in MultipoleAxisType
}


@dataclass
class Particle:
    idx: int
    name: str
    element: str
    mass: float
    resnum: int
    resname: str
    xx: float
    xy: float
    xz: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


@dataclass
class AmoebaVdw147:
    idx: int
    epsilon: float
    r0: float
    parentIdx: int = -1
    reduction: float = 0.0
    paramIdx: int = -1


@dataclass
class Multipole:
    idx: int
    c0: float
    dx: float
    dy: float
    dz: float
    qxx: float
    qxy: float
    qxz: float
    qyy: float
    qyz: float
    qzz: float
    axistype: int
    kz: int = -1
    kx: int = -1
    ky: int = -1
    paramIdx: int = -1


@dataclass
class IsotropicPolarization:
    idx: int
    alpha: float
    thole: float
    grp: list
    paramIdx: int = -1

    def __post_init__(self):
        if isinstance(self.grp, str):
            self.grp = [int(x) for x in self.grp.split()]


@dataclass
class AnisotropicPolarization:
    idx: int
    alphaxx: float
    alphaxy: float
    alphaxz: float
    alphayy: float
    alphayz: float
    alphazz: float
    thole: float
    grp: list = field(default_factory=list)
    paramIdx: int = -1

    def __post_init__(self):
        if isinstance(self.grp, str):
            self.grp = [int(x) for x in self.grp.split()]


@dataclass
class MBUCBChargePenetration:
    idx: int
    alpha: float
    beta: float
    paramIdx: int = -1


@dataclass
class MBUCBChargeTransfer:
    idx: int
    d: float
    b: float
    alpha: float
    paramIdx: int = -1


@dataclass
class PairList:
    """Class for atom pairs that has to specially treated in non-bonded calculations"""
    p0: int
    p1: int

    def energy(self) -> float:
        return 0.0
    

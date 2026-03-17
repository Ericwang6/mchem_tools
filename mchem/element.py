"""Periodic table data: element symbol, mass, and name."""

from dataclasses import dataclass


@dataclass
class Element:
    """
    Dataclass for a chemical element.

    Attributes
    ----------
    atomicNum : int
        Atomic number.
    symbol : str
        Element symbol (e.g. ``'C'``, ``'H'``).
    mass : float
        Atomic mass in amu.
    name : str
        Full element name.
    """

    atomicNum: int
    symbol: str
    mass: float
    name: str
    

H  = Element(  1, 'H' , 1.0079, 'Hydrogen')
He = Element(  2, 'He', 4.0026, 'Helium')
Li = Element(  3, 'Li', 6.9410, 'Lithium')
Be = Element(  4, 'Be', 9.0122, 'Beryllium')
B  = Element(  5, 'B' , 10.8110, 'Boron')
C  = Element(  6, 'C' , 12.0107, 'Carbon')
N  = Element(  7, 'N' , 14.0067, 'Nitrogen')
O  = Element(  8, 'O' , 15.9994, 'Oxygen')
F  = Element(  9, 'F' , 18.9984, 'Fluorine')
Ne = Element( 10, 'Ne', 20.1797, 'Neon')
Na = Element( 11, 'Na', 22.9898, 'Sodium')
Mg = Element( 12, 'Mg', 24.3050, 'Magnesium')
Al = Element( 13, 'Al', 26.9815, 'Aluminum')
Si = Element( 14, 'Si', 28.0855, 'Silicon')
P  = Element( 15, 'P' , 30.9738, 'Phosphorus')
S  = Element( 16, 'S' , 32.0650, 'Sulfur')
Cl = Element( 17, 'Cl', 35.4530, 'Chlorine')
Ar = Element( 18, 'Ar', 39.9480, 'Argon')
K  = Element( 19, 'K' , 39.0983, 'Potassium')
Ca = Element( 20, 'Ca', 40.0780, 'Calcium')
Sc = Element( 21, 'Sc', 44.9559, 'Scandium')
Ti = Element( 22, 'Ti', 47.8670, 'Titanium')
V  = Element( 23, 'V' , 50.9415, 'Vanadium')
Cr = Element( 24, 'Cr', 51.9961, 'Chromium')
Mn = Element( 25, 'Mn', 54.9380, 'Manganese')
Fe = Element( 26, 'Fe', 55.8450, 'Iron')
Co = Element( 27, 'Co', 58.9331, 'Cobalt')
Ni = Element( 28, 'Ni', 58.6934, 'Nickel')
Cu = Element( 29, 'Cu', 63.5460, 'Copper')
Zn = Element( 30, 'Zn', 65.4090, 'Zinc')
Ga = Element( 31, 'Ga', 69.7230, 'Gallium')
Ge = Element( 32, 'Ge', 72.6400, 'Germanium')
As = Element( 33, 'As', 74.9216, 'Arsenic')
Se = Element( 34, 'Se', 78.9600, 'Selenium')
Br = Element( 35, 'Br', 79.9040, 'Bromine')
Kr = Element( 36, 'Kr', 83.7980, 'Krypton')
Rb = Element( 37, 'Rb', 85.4678, 'Rubidium')
Sr = Element( 38, 'Sr', 87.6200, 'Strontium')
Y  = Element( 39, 'Y' , 88.9059, 'Yttrium')
Zr = Element( 40, 'Zr', 91.2240, 'Zirconium')
Nb = Element( 41, 'Nb', 92.9064, 'Niobium')
Mo = Element( 42, 'Mo', 95.9400, 'Molybdenum')
Tc = Element( 43, 'Tc', 98.0000, 'Technetium')
Ru = Element( 44, 'Ru', 101.0700, 'Ruthenium')
Rh = Element( 45, 'Rh', 102.9055, 'Rhodium')
Pd = Element( 46, 'Pd', 106.4200, 'Palladium')
Ag = Element( 47, 'Ag', 107.8682, 'Silver')
Cd = Element( 48, 'Cd', 112.4110, 'Cadmium')
In = Element( 49, 'In', 114.8180, 'Indium')
Sn = Element( 50, 'Sn', 118.7100, 'Tin')
Sb = Element( 51, 'Sb', 121.7600, 'Antimony')
Te = Element( 52, 'Te', 127.6000, 'Tellurium')
I  = Element( 53, 'I' , 126.9045, 'Iodine')
Xe = Element( 54, 'Xe', 131.2930, 'Xenon')
Cs = Element( 55, 'Cs', 132.9055, 'Cesium')
Ba = Element( 56, 'Ba', 137.3270, 'Barium')
La = Element( 57, 'La', 138.9055, 'Lanthanum')
Ce = Element( 58, 'Ce', 140.1160, 'Cerium')
Pr = Element( 59, 'Pr', 140.9077, 'Praseodymium')
Nd = Element( 60, 'Nd', 144.2420, 'Neodymium')
Pm = Element( 61, 'Pm', 145.0000, 'Promethium')
Sm = Element( 62, 'Sm', 150.3600, 'Samarium')
Eu = Element( 63, 'Eu', 151.9640, 'Europium')
Gd = Element( 64, 'Gd', 157.2500, 'Gadolinium')
Tb = Element( 65, 'Tb', 158.9254, 'Terbium')
Dy = Element( 66, 'Dy', 162.5000, 'Dysprosium')
Ho = Element( 67, 'Ho', 164.9303, 'Holmium')
Er = Element( 68, 'Er', 167.2590, 'Erbium')
Tm = Element( 69, 'Tm', 168.9342, 'Thulium')
Yb = Element( 70, 'Yb', 173.0400, 'Ytterbium')
Lu = Element( 71, 'Lu', 174.9670, 'Lutetium')
Hf = Element( 72, 'Hf', 178.4900, 'Hafnium')
Ta = Element( 73, 'Ta', 180.9479, 'Tantalum')
W  = Element( 74, 'W' , 183.8400, 'Tungsten')
Re = Element( 75, 'Re', 186.2070, 'Rhenium')
Os = Element( 76, 'Os', 190.2300, 'Osmium')
Ir = Element( 77, 'Ir', 192.2170, 'Iridium')
Pt = Element( 78, 'Pt', 195.0840, 'Platinum')
Au = Element( 79, 'Au', 196.9666, 'Gold')
Hg = Element( 80, 'Hg', 200.5900, 'Mercury')
Tl = Element( 81, 'Tl', 204.3833, 'Thallium')
Pb = Element( 82, 'Pb', 207.2000, 'Lead')
Bi = Element( 83, 'Bi', 208.9804, 'Bismuth')
Po = Element( 84, 'Po', 209.0000, 'Polonium')
At = Element( 85, 'At', 210.0000, 'Astatine')
Rn = Element( 86, 'Rn', 222.0000, 'Radon')
Fr = Element( 87, 'Fr', 223.0000, 'Francium')
Ra = Element( 88, 'Ra', 226.0000, 'Radium')
Ac = Element( 89, 'Ac', 227.0000, 'Actinium')
Th = Element( 90, 'Th', 232.0381, 'Thorium')
Pa = Element( 91, 'Pa', 231.0359, 'Proactinium')
U  = Element( 92, 'U' , 238.0289, 'Uranium')
Np = Element( 93, 'Np', 237.0000, 'Neptunium')
Pu = Element( 94, 'Pu', 244.0000, 'Plutonium')
Am = Element( 95, 'Am', 243.0000, 'Americium')
Cm = Element( 96, 'Cm', 247.0000, 'Curium')
Bk = Element( 97, 'Bk', 247.0000, 'Berkelium')
Cf = Element( 98, 'Cf', 251.0000, 'Californium')
Es = Element( 99, 'Es', 252.0000, 'Einsteinium')
Fm = Element(100, 'Fm', 257.0000, 'Fermium')
Md = Element(101, 'Md', 258.0000, 'Mendelevium')
No = Element(102, 'No', 259.0000, 'Nobelium')
Lr = Element(103, 'Lr', 262.0000, 'Lawrencium')
Rf = Element(104, 'Rf', 261.0000, 'Rutherfordium')
Db = Element(105, 'Db', 262.0000, 'Dubnium')
Sg = Element(106, 'Sg', 266.0000, 'Seaborgium')
Bh = Element(107, 'Bh', 264.0000, 'Bohrium')
Hs = Element(108, 'Hs', 277.0000, 'Hassium')
Mt = Element(109, 'Mt', 268.0000, 'Meitnerium')
Ds = Element(110, 'Ds', 281.0000, 'Darmstadtium')
Rg = Element(111, 'Rg', 272.0000, 'Roentgenium')
Cn = Element(112, 'Cn', 285.0000, 'Copernicium')
Nh = Element(113, 'Nh', 286.0000, 'Nihonium')
Fl = Element(114, 'Fl', 289.0000, 'Flerovium')
Mc = Element(115, 'Mc', 289.0000, 'Moscovium')
Lv = Element(116, 'Lv', 293.0000, 'Livermorium')
Ts = Element(117, 'Ts', 294.0000, 'Tennessine')
Og = Element(118, 'Og', 294.0000, 'Oganesson')


ELEMENTS = {
    'H': H,
    'He': He,
    'Li': Li,
    'Be': Be,
    'B': B,
    'C': C,
    'N': N,
    'O': O,
    'F': F,
    'Ne': Ne,
    'Na': Na,
    'Mg': Mg,
    'Al': Al,
    'Si': Si,
    'P': P,
    'S': S,
    'Cl': Cl,
    'Ar': Ar,
    'K': K,
    'Ca': Ca,
    'Sc': Sc,
    'Ti': Ti,
    'V': V,
    'Cr': Cr,
    'Mn': Mn,
    'Fe': Fe,
    'Co': Co,
    'Ni': Ni,
    'Cu': Cu,
    'Zn': Zn,
    'Ga': Ga,
    'Ge': Ge,
    'As': As,
    'Se': Se,
    'Br': Br,
    'Kr': Kr,
    'Rb': Rb,
    'Sr': Sr,
    'Y': Y,
    'Zr': Zr,
    'Nb': Nb,
    'Mo': Mo,
    'Tc': Tc,
    'Ru': Ru,
    'Rh': Rh,
    'Pd': Pd,
    'Ag': Ag,
    'Cd': Cd,
    'In': In,
    'Sn': Sn,
    'Sb': Sb,
    'Te': Te,
    'I': I,
    'Xe': Xe,
    'Cs': Cs,
    'Ba': Ba,
    'La': La,
    'Ce': Ce,
    'Pr': Pr,
    'Nd': Nd,
    'Pm': Pm,
    'Sm': Sm,
    'Eu': Eu,
    'Gd': Gd,
    'Tb': Tb,
    'Dy': Dy,
    'Ho': Ho,
    'Er': Er,
    'Tm': Tm,
    'Yb': Yb,
    'Lu': Lu,
    'Hf': Hf,
    'Ta': Ta,
    'W': W,
    'Re': Re,
    'Os': Os,
    'Ir': Ir,
    'Pt': Pt,
    'Au': Au,
    'Hg': Hg,
    'Tl': Tl,
    'Pb': Pb,
    'Bi': Bi,
    'Po': Po,
    'At': At,
    'Rn': Rn,
    'Fr': Fr,
    'Ra': Ra,
    'Ac': Ac,
    'Th': Th,
    'Pa': Pa,
    'U': U,
    'Np': Np,
    'Pu': Pu,
    'Am': Am,
    'Cm': Cm,
    'Bk': Bk,
    'Cf': Cf,
    'Es': Es,
    'Fm': Fm,
    'Md': Md,
    'No': No,
    'Lr': Lr,
    'Rf': Rf,
    'Db': Db,
    'Sg': Sg,
    'Bh': Bh,
    'Hs': Hs,
    'Mt': Mt,
    'Ds': Ds,
    'Rg': Rg,
    'Cn': Cn,
    'Nh': Nh,
    'Fl': Fl,
    'Mc': Mc,
    'Lv': Lv,
    'Ts': Ts,
    'Og': Og,
}

#for key in ELEMENTS:
#    if len(key) > 1:
#        ELEMENTS[key.upper()] = ELEMENTS[key]

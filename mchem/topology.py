import uuid
from typing import Optional, Union, List

import contextlib
import numpy as np

from .element import ELEMENTS
from .template import ResidueTemplate


class Residue:
    def __init__(self, name: str, number: int, chain: str = "A", insertionCode: str = ""):
        self._atoms = []
        self._atoms_by_name = {}
        self._name = name
        self.number = number
        self.chain = chain
        self.insertionCode = insertionCode
        self._topology = None
        self._idx = -1
        self._stdname = name
    
    @property
    def name(self) -> str:
        return self._name
    
    def setName(self, newName: str):
        self._name = newName
    
    @property
    def stdName(self) -> str:
        return self._stdname
    
    def setStdName(self, newName: str):
        self._stdname = newName

    @property
    def topology(self):
        return self._topology
    
    def setTopology(self, topology):
        self._topology = topology
    
    def hasTopology(self):
        return self.topology is not None

    @property
    def idx(self) -> int:
        if self.hasTopology():
            return self._idx
        else:
            return -1
    
    def setIdx(self, newIdx: int):
        self._idx = newIdx

    def addAtom(self, atom):
        atom.setResidue(self)
        self._atoms.append(atom)
        if self.getAtomWithName(atom.name) is not None:
            raise RuntimeError(f"Atom {atom.name} already exists.")
        else:
            self._atoms_by_name[atom.name] = atom
    
    def getAtomWithName(self, name: str):
        if name in self._atoms_by_name:
            return self._atoms_by_name[name]
        else:
            return None
    
    def removeAtom(self, atom):
        atom.setResidue(None)
        self._atoms.remove(atom)
        del self._atoms_by_name[atom.name]
    
    @property
    def numAtoms(self):
        return len(self._atoms)
    
    @property
    def atoms(self):
        return self._atoms
    
    def __repr__(self):
        rep = f"<{self.__class__.__name__} {self.name}[{self.number}]; chain={self.chain}"
        if self.insertionCode:
            rep += f"; insertionCode={self.insertionCode}"
        if self.idx != -1:
            rep += f"; id={self.idx}"
        rep += ">"
        return rep
    
    def matchTemplate(self, template: ResidueTemplate, stdResidueName: bool = True):
        # match residue name
        if self.name != template.name:
            if self.name in template.altNames and stdResidueName:
                self.setName(template.name)
            else:
                return False
        self.setStdName(template.name)
        
        # match atoms
        if len(self.atoms) != len(template.atoms):
            return False
        
        for atomName, atomInfo in template.atoms.items():
            if self.getAtomWithName(atomName) is not None:
                continue
            else:
                for altname in atomInfo['altNames']:
                    atomMatch = self.getAtomWithName(altname)
                    if atomMatch is not None:
                        # standardize atom name
                        del self._atoms_by_name[altname]
                        self._atoms_by_name[atomName] = atomMatch
                        atomMatch.setName(atomName)
                        break
                else:
                    return False

        # match bonds
        for bond in template.bonds:
            if bond['atom1'].startswith("-"):
                prevRes = self.topology.getResidueWithIdx(self.idx - 1)
                atom1 = prevRes.getAtomWithName(bond['atom1'][1:])
                atom2 = self.getAtomWithName(bond['atom2'])
            elif bond['atom2'].startswith("-"):
                prevRes = self.topology.getResidueWithIdx(self.idx - 1)
                atom1 = self.getAtomWithName(bond['atom1'])
                atom2 = prevRes.getAtomWithName(bond['atom2'][1:])
            else:
                atom1 = self.getAtomWithName(bond['atom1'])
                atom2 = self.getAtomWithName(bond['atom2'])
            bondToAdd = Bond(atom1, atom2, bond['order'])
            self.topology.addBond(bondToAdd)
        
        return True


class Atom:
    def __init__(self, name: str, element: str):
        self._name = name
        self.element = ELEMENTS[element]
        self._residue = None
        self._bonds = []
        self._idx = -1
        self._neighbors = []
        self._atype = None
        self._aclass = None
        self._uuid = uuid.uuid4()
        self._top_info = None
        self._polarization_group = set([self])
        self.xx = 0.0
        self.xy = 0.0
        self.xz = 0.0
    
    def setPosition(self, pos):
        self.xx = pos[0]
        self.xy = pos[1]
        self.xz = pos[2]
    
    @property
    def polarizationGroup(self):
        return self._polarization_group
    
    def setPolarizationGroup(self, setOfAtoms):
        self._polarization_group = setOfAtoms
    
    def addToPolarizationGroup(self, other):
        grp = self.polarizationGroup.union(other.polarizationGroup)
        self._polarization_group = grp
        other._polairzation_group = grp
    
    def __hash__(self):
        return hash(self._uuid)
    
    @property
    def mass(self) -> float:
        return self.element.mass
    
    @property
    def symbol(self) -> str:
        return self.element.symbol
    
    @property
    def atomicNum(self) -> int:
        return self.element.atomicNum
    
    @property
    def atomType(self):
        return self._atype
    
    @property
    def atomClass(self):
        return self._aclass
    
    def setAtomType(self, atype):
        self._atype = atype

    def setAtomClass(self, aclass):
        self._aclass = aclass
    
    @property
    def parametrized(self):
        return self.atomType is not None
    
    @property
    def name(self) -> str:
        return self._name
    
    def setName(self, newname: str):
        self._name = newname
    
    def __repr__(self) -> str:
        if self.atomType is not None:
            typeStr = f" type={self.atomType}"
        else:
            typeStr = ""
        return f"<Atom {self.name} [{self.idx}]{typeStr}>"
    
    @property
    def idx(self) -> int:
        if self.hasTopology():
            return self._idx
        else:
            return -1
    
    def setIdx(self, newIdx: int):
        self._idx = newIdx

    @property
    def residue(self):
        return self._residue
    
    def hasResidue(self):
        return self.residue is not None
    
    def setResidue(self, residue: Union[None, Residue]):
        self._residue = residue

    @property
    def topology(self):
        if self.hasResidue():
            return self.residue.topology
        else:
            return None
    
    def hasTopology(self):
        if self.hasResidue():
            return self.residue.hasTopology()
        else:
            return False
    
    @property
    def bonds(self):
        return self._bonds
    
    def generateNeighbors(self):
        neighbors = []
        for bond in self.bonds:
            if bond.atom1 is self:
                neighbors.append(bond.atom2)
            else:
                neighbors.append(bond.atom1)
        self._neighbors = neighbors
    
    def getNeighbors(self):
        return self._neighbors
    
    def getHighOrderNeighbors(self, order: int):
        assert order > 0
        hoNeis = [set(self.getNeighbors())]
        for _ in range(order - 1):
            nblist = set()
            for atom in hoNeis[-1]:
                for nei in atom.getNeighbors():
                    nblist.add(nei)
            for prevList in hoNeis:
                nblist -= prevList
            nblist -= set([self])
            hoNeis.append(nblist)

        hoNeisResult = []
        for i, atoms in enumerate(hoNeis):
            for atom in atoms:
                hoNeisResult.append((atom, i+1))
        return hoNeisResult
    
    def generateTopologicalInfo(self, maxConnect: int):
        paths = {i+1: set() for i in range(maxConnect)}
        paths[1] = set(BondedAtoms([self, nei]) for nei in self.getNeighbors())
        for i in range(2, maxConnect + 1):
            for path in paths[i-1]:
                for atom in path[-1].getNeighbors():
                    if not (atom is path[-2]):
                        paths[i].add(path.extend([atom]))
        
        nbList = {}
        for numConnect, path in paths.items():
            for p in path:
                if p[-1] not in nbList:
                    nbList[p[-1]] = numConnect
        
        self._top_info = {
            "bondedAtoms": nbList,
            "pathToBondedAtoms": paths
        }
    
    @property
    def pathsToBondedAtoms(self):
        return self.getTopologicalInfo()['pathToBondedAtoms']
    
    @property
    def bondedAtoms(self):
        return self.getTopologicalInfo()['bondedAtoms']

    def getTopologicalInfo(self):
        if self._top_info is None:
            raise RuntimeError("Topological Information is None")
        else:
            return self._top_info

    def addBond(self, bond):
        if (bond.atom1 is not self) and (bond.atom2 is not self):
            raise RuntimeError("The bond to add does not contain this atom")
        elif (bond.atom1 is self) and (bond.atom2 is self):
            raise RuntimeError("Atoms in a bond are the same")
        self._bonds.append(bond)
    
    def pruneBonds(self):
        rmlist = []
        for i, bo in enumerate(self.bonds):
            if not bo.hasTopology():
                rmlist.append(i)
        rmlist.reverse()
        for i in rmlist:
            del self._bonds[i]
        self.generateNeighbors()


class Bond:
    def __init__(self, atom1: Atom, atom2: Atom, order: float):
        self.atom1 = atom1
        self.atom2 = atom2
        self.order = order
        self._idx = -1
    
    def hasTopology(self) -> bool:
        return self.atom1.hasTopology() and self.atom2.hasTopology()
    

def require_editable(func):
    def new_func(self, *args, **kwargs):
        if not self.editable:
            raise RuntimeError("Topolgy is not editable")
        else:
            return func(self, *args, **kwargs)
    return new_func


def require_noneditable(func):
    def new_func(self, *args, **kwargs):
        if self.editable:
            raise RuntimeError("Topology is editable")
        else:
            return func(self, *args, **kwargs)
    return new_func



class BondedAtoms:
    def __init__(self, atoms: List[Atom]):
        self.atoms = atoms if isinstance(atoms, list) else list(atoms)
        tuple1 = tuple(atom for atom in self.atoms)
        tuple2 = tuple(self.atoms[i-1] for i in range(len(self), 0, -1))
        hash1 = hash(tuple1)
        hash2 = hash(tuple2)
        self._hash = min(hash1, hash2)
    
    def __hash__(self):
        return self._hash
    
    def __len__(self):
        return len(self.atoms)
    
    def __eq__(self, other):
        if self.atoms == other.atoms:
            return True
        elif self.atoms == [other.atoms[i-1] for i in range(len(other.atoms), 0, -1)]:
            return True
        else:
            return False
    
    def __getitem__(self, idx: int):
        return self.atoms[idx]
    
    def extend(self, atoms):
        return BondedAtoms(self.atoms + atoms)
    
    def __repr__(self):
        return repr(self.atoms)
        

class Topology:
    def __init__(self, name: str, maxConnect: int = 5):
        self._residues = []
        self._bonds = []
        self.name = name
        self._editable = False
        assert maxConnect > 0, "maxConnect must be positive"
        self._maxConnect = maxConnect
        self._topInfo = None
    
    @property
    def editable(self):
        return self._editable
    
    @contextlib.contextmanager
    def setEditable(self):
        self._topInfo = None
        self._editable = True
        yield
        self.assignIndex()
        self.pruneBonds()
        self.generateTopologicalInfo()
        self._editable = False

    def assignIndex(self):
        atomId = 0
        for resId, res in enumerate(self.residues):
            res.setIdx(resId)
            for atom in res.atoms:
                atom.setIdx(atomId)
                atomId += 1
    
    def pruneBonds(self):
        rmlist = []
        for i, bo in enumerate(self.bonds):
            if not bo.hasTopology():
                rmlist.append(i)
        rmlist.reverse()
        for i in rmlist:
            del self._bonds[i]
        
        for atom in self.atoms():
            atom.pruneBonds()

    @property
    def residues(self) -> List[Residue]:
        return self._residues
    
    @property
    def numResidues(self):
        return len(self._residues)
    
    def getResidueWithIdx(self, idx: int) -> Residue:
        return self.residues[idx]
    
    @require_editable
    def addResidue(self, residue: Residue):
        residue.setIdx(self.numResidues)
        residue.setTopology(self)
        self._residues.append(residue)
    
    @require_editable
    def insertResidue(self, residue: Residue, resId: int):
        residue.setIdx(resId)
        residue.setTopology(self)
        for res in self.residues[resId:]:
            res.setIdx(res.id + 1)
        self._residues.insert(resId, residue)
    
    @require_editable
    def removeResidue(self, resId: int):
        residue = self.getResidueWithIdx(resId)
        residue.setTopology(None)
        for res in self.residues[resId:]:
            res.setIdx(res.id - 1)
        self._residues.remove(residue)

    @property
    def numAtoms(self):
        return sum([res.numAtoms for res in self.residues])
    
    @property
    def numBonds(self):
        return len(self.bonds)
    
    def atoms(self) -> Atom:
        for res in self.residues:
            for atom in res.atoms:
                yield atom
    
    def getAtomWithIdx(self, idx: int) -> Atom:
        if idx >= self.numAtoms:
            raise IndexError("Index out of range")
        else:
            resId = 0
            while idx >= self.residues[resId].numAtoms:
                resId += 1
                idx -= self.residues[resId].numAtoms
            return self.residues[resId].atoms[idx]
    
    @require_editable
    def addAtomToResidue(self, atom: Atom, resId: int):
        res = self.getResidueWithIdx(resId)
        res.addAtom(atom)

    def __repr__(self):
        rep = f"<Topology[{self.name}]; {self.numAtoms} atoms; {self.numBonds} bonds>"
        return rep
    
    @property
    def bonds(self):
        return self._bonds
    
    @require_editable
    def addBond(self, bond: Bond):
        bond.atom1.addBond(bond)
        bond.atom2.addBond(bond)
        self._bonds.append(bond)
    
    @property
    def connTable(self):
        return self._topInfo['connectivity']
    
    @property
    def bondedAtoms(self):
        return self._topInfo['bondedAtoms']
    
    def generateTopologicalInfo(self):
        topInfo = {
            "connectivity": {},
            "bondedAtoms": {i+1: set() for i in range(self._maxConnect)},
        }
        for atom in self.atoms():
            atom.generateTopologicalInfo(self._maxConnect)
            atomTopInfo = atom.getTopologicalInfo()
            for numConnect, paths in atomTopInfo['pathToBondedAtoms'].items():
                topInfo['bondedAtoms'][numConnect] = topInfo['bondedAtoms'][numConnect].union(paths)
            for nei, numConnect in atom.bondedAtoms.items():
                atomPair = BondedAtoms([atom, nei])
                topInfo['connectivity'][atomPair] = numConnect
        
        conn = []
        for p, numConnect in topInfo['connectivity'].items():
            if p.atoms[0].idx < p.atoms[1].idx:
                conn.append([p.atoms[0].idx, p.atoms[1].idx, numConnect])
            else:
                conn.append([p.atoms[1].idx, p.atoms[0].idx, numConnect])
        conn.sort(key=lambda x: (x[-1], x[0], x[1]))
        conn = np.array(conn, dtype=int)

        topInfo["connectivity"] = conn
        self._topInfo = topInfo
                
    def matchTemplates(self):
        from .template import TEMPLATES

        resCYX = []
        for residue in self.residues:
            if residue.name not in TEMPLATES:
                raise KeyError(f"ResidueTemplate {residue.name} not defined")
            succ = residue.matchTemplate(TEMPLATES[residue.name])
            if not succ:
                succ = residue.matchTemplate(TEMPLATES[f'N{residue.name}'], stdResidueName=False)
            if not succ:
                succ = residue.matchTemplate(TEMPLATES[f'C{residue.name}'], stdResidueName=False)
            if not succ:
                protonatedStates = [
                    "HIS", "HIP", "HID", "HIE", 
                    "ASP", "ASH", 
                    "GLU", "GLH", 
                    "LYS", "LYD", 
                    "CYS", "CYD", 
                    "TYR", "TYD"
                ]
                for key in protonatedStates:
                    succ = residue.matchTemplate(TEMPLATES[key])
                    if succ:
                        break
            if not succ:
                raise RuntimeError(f"Fail to match template for residue #{residue.idx} {residue.name}")
            if residue.name == "CYX":
                resCYX.append(residue.idx)
        
        if len(resCYX) > 0:
            self.processDisulfideBond()

    def processDisulfideBond(self):
        raise NotImplementedError()
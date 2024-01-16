import itertools
import xml.etree.ElementTree as ET

from mchem.topology import Topology

from ..topology import Topology
from .base import ForceField, Generator, Parsers, str2float, str2bool, str2int, float2str
from ..terms import (
    TermList,
    AmoebaBond, 
    AmoebaAngle, 
    AmoebaAngleInPlane,
    AmoebaStretchTorsion, 
    AmoebaStretchBend, 
    AmoebaAngleTorsion,
    AmoebaOutOfPlaneBend,
    AmoebaUreyBradley,
    AmoebaPiTorsion,
    HarmonicAngle,
    HarmonicBond,
    PeriodicTorsion,
    AmoebaVdw147,
    Multipole,
    MultipoleAxisType,
    MultipoleAxisTypeInt2Str,
    IsotropicPolarization,
    MBUCBChargePenetration,
    AnisotropicPolarization,
    MBUCBChargeTransfer
)


class AmoebaBondGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, ["b0", "kb"], False)

    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(AmoebaBondGenerator)
        generator.setMetadata("bondCubic", str2float(element.get("bond-cubic")))
        generator.setMetadata("bondQuartic", str2float(element.get("bond-quartic")))
        for bond in element.findall("Bond"):
            generator.addBond(bond)
    
    def addBond(self, bondElement: ET.Element):
        paramDict = {
            "b0": str2float(bondElement.get("length")),
            "kb": str2float(bondElement.get("k"))
        }
        if "smirks" not in bondElement.attrib:
            atypes = self.ff.findAtomTypes(bondElement, 2)
            self.addParameterWithAtomTypes(atypes, paramDict)
        else:
            self.addParameterWithSmirks(
                bondElement.get("smirks"),
                paramDict
            )

    def createTerms(self, topology: Topology, **kwargs):
        bondTerms = TermList(AmoebaBond)
        bCubic = self.getMetadata("bondCubic")
        bQuartic = self.getMetadata("bondQuartic")

        useSmirks = kwargs.get("useSmirks", False)
        if useSmirks:
            raise NotImplementedError()
        else:
            for bond in topology.bondedAtoms[1]:
                atom1, atom2 = bond.atoms[0], bond.atoms[1]

                paramIdx = self.getParameterIdxWithAtomType((atom1.atomType, atom2.atomType))
                if paramIdx is None:
                    paramIdx = self.getParameterIdxWithAtomType((atom2.atomType, atom1.atomType))
                
                if paramIdx is None:
                    self.raise_exception(f"Bond between {atom1.idx} and {atom2.idx} not matched")
                
                param = self.getParameterWithIdx(paramIdx)
                term = AmoebaBond(
                    atom1.idx, 
                    atom2.idx, 
                    param['b0'], 
                    param['kb'], 
                    bCubic, 
                    bQuartic,
                    paramIdx=paramIdx
                )
                bondTerms.append(term)
        # bondTerms.sort(key=lambda t: (t.p0, t.p1))
        return bondTerms


Parsers["AmoebaBondForce"] = AmoebaBondGenerator


class AmoebaAngleGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, ['th0', 'inPlane', 'kth'], False)
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(AmoebaAngleGenerator)
        generator.setMetadata("angleCubic", str2float(element.get("angle-cubic")))
        generator.setMetadata("angleQuartic", str2float(element.get("angle-quartic")))
        generator.setMetadata("anglePentic", str2float(element.get('angle-pentic')))
        generator.setMetadata("angleSextic", str2float(element.get("angle-sextic")))
        for angle in element.findall("Angle"):
            generator.addAngle(angle)
        
    def addAngle(self, angleElement: ET.Element):
        paramDict = {
            "th0": [
                str2float(angleElement.get("angle1")),
                str2float(angleElement.get("angle2", 0.0)),
                str2float(angleElement.get("angle3", 0.0))
            ],
            "kth": str2float(angleElement.get("k")),
            "inPlane": str2bool(angleElement.get("inPlane"))
        }
        if "smirks" not in angleElement.attrib:
            atypes = self.ff.findAtomTypes(angleElement, 3)
            self.addParameterWithAtomTypes(atypes, paramDict)
        else:
            self.addParameterWithSmirks(
                angleElement.get("smirks"),
                paramDict
            )
    
    def createTerms(self, topology: Topology, **kwargs):
        angleTerms = TermList(AmoebaAngle)
        angleInPlaneTerms = TermList(AmoebaAngleInPlane)
        aCubic = self.getMetadata("angleCubic")
        aQuartic = self.getMetadata("angleQuartic")
        aPentic = self.getMetadata("anglePentic")
        aSextic = self.getMetadata("angleSextic")

        useSmirks = kwargs.get("useSmirks", False)
        if useSmirks:
            raise NotImplementedError()
        else:
            for angle in topology.bondedAtoms[2]:
                atom1, atom2, atom3 = angle.atoms[0], angle.atoms[1], angle.atoms[2]
                # count non-hydrogens on the central atom -> determine 'angle1' or 'angle2' or 'angle3'
                # adapted from openmm/app/forcefield.py#L3585
                
                paramIdx = self.getParameterIdxWithAtomType((atom1.atomType, atom2.atomType, atom3.atomType))
                if paramIdx is None:
                    paramIdx = self.getParameterIdxWithAtomType((atom3.atomType, atom2.atomType, atom1.atomType))
                
                if paramIdx is None:
                    self.raise_exception(f"Bond between {atom1.idx} and {atom2.idx} not matched")
                
                param = self.getParameterWithIdx(paramIdx)
                if len(param['th0']) > 1:
                    numHydrogens = 0
                    for nei in atom2.getNeighbors():
                        if (nei is not atom1) and (nei is not atom3) and (nei.element.atomicNum == 1):
                            numHydrogens += 1
                    th0 = param['th0'][numHydrogens]
                else:
                    th0 = param['th0'][0]
                
                if not param['inPlane']:
                    term = AmoebaAngle(
                        atom1.idx, atom2.idx, atom3.idx,
                        th0, param['kth'],
                        aCubic, aQuartic, aPentic, aSextic,
                        paramIdx=paramIdx
                    )
                    angleTerms.append(term)
                else:
                    neighbors = angle.atoms[1].getNeighbors()
                    assert len(neighbors) == 3
                    auxAtom = [nei for nei in neighbors if (nei is not atom1) and (nei is not atom3)][0]

                    term = AmoebaAngleInPlane(
                        atom1.idx, atom2.idx, atom3.idx, auxAtom.idx,
                        th0, param['kth'], 
                        aCubic, aQuartic, aPentic, aSextic,
                        paramIdx=paramIdx
                    )
                    angleInPlaneTerms.append(term)
        # angleTerms.sort(key=lambda t: (t.p0, t.p1, t.p2))
        # angleInPlaneTerms.sort(key=lambda t: (t.p0, t.p1, t.p2))
        return angleTerms, angleInPlaneTerms

Parsers["AmoebaAngleForce"] = AmoebaAngleGenerator


class AmoebaUreyBradleyGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, ["fc", "r0"], False)
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(AmoebaUreyBradleyGenerator)
        generator.setMetadata("ubCubic", str2float(element.get("cubic", 0.0)))
        generator.setMetadata("ubQuartic", str2float(element.get("quartic", 0.0)))
        for ub in element.findall("UreyBradley"):
            generator.addUreyBrad(ub)
        
    def addUreyBrad(self, ubElement: ET.Element):
        paramDict = {
            "fc": str2float(ubElement.get("k")),
            "r0": str2float(ubElement.get("d"))
        }
        if "smirks" not in ubElement.attrib:
            atypes = self.ff.findAtomTypes(ubElement, 3)
            self.addParameterWithAtomTypes(atypes, paramDict)
        else:
            self.addParameterWithSmirks(
                ubElement.get("smirks"),
                paramDict
            )
    
    def createTerms(self, topology: Topology, **kwargs):
        ubTerms = TermList(AmoebaUreyBradley)
        ubCubic = self.getMetadata("ubCubic")
        ubQuartic = self.getMetadata("ubQuartic")

        useSmirks = kwargs.get("useSmirks", False)
        if useSmirks:
            raise NotImplementedError()
        else:
            for angle in topology.bondedAtoms[2]:
                atom1, atom2, atom3 = angle.atoms[0], angle.atoms[1], angle.atoms[2]
                
                paramIdx = self.getParameterIdxWithAtomType((atom1.atomType, atom2.atomType, atom3.atomType))
                if paramIdx is None:
                    paramIdx = self.getParameterIdxWithAtomType((atom3.atomType, atom2.atomType, atom1.atomType))
                
                if paramIdx is None:
                    continue
                
                param = self.getParameterWithIdx(paramIdx)
                term = AmoebaUreyBradley(
                    atom1.idx, atom2.idx, atom3.idx,
                    param['r0'], param['fc'],
                    paramIdx=paramIdx
                )
                ubTerms.append(term)
        # ubTerms.sort(key=lambda t: (t.p0, t.p1))
        return ubTerms
    
Parsers["AmoebaUreyBradleyForce"] = AmoebaUreyBradleyGenerator


class MultipoleGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, [
            'kz', 'kx', 'ky', 'axisType', 'multipoles'
        ], False)
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(MultipoleGenerator)
        for mpole in element.findall("Multipole"):
            generator.addMultipole(mpole)
    
    @staticmethod
    def setAxisType(kz: int, kx: int, ky: int):
        # from OpenMM
        axisType = MultipoleAxisType.ZThenX.value
        if (kz == 0):
            axisType = MultipoleAxisType.NoAxisType.value
        if (kz != 0 and kx == 0):
            axisType = MultipoleAxisType.ZOnly.value
        if (kz < 0 or kx < 0):
            axisType = MultipoleAxisType.Bisector.value
        if (kx < 0 and ky < 0):
            axisType = MultipoleAxisType.ZBisect.value
        if (kz < 0 and kx < 0 and ky < 0):
            axisType = MultipoleAxisType.ThreeFold.value
        return axisType

    def addMultipole(self, mpoleElement: ET.Element):
        kz = str2int(mpoleElement.get("kz", 0))
        kx = str2int(mpoleElement.get("kx", 0))
        ky = str2int(mpoleElement.get("ky", 0))
        
        if "axistype" in mpoleElement.attrib:
            axisType = MultipoleAxisType[mpoleElement.get("axistype")].value
        else:
            axisType = MultipoleGenerator.setAxisType(kz, kx, ky)
        
        paramDict = {
            "kz": abs(kz) if kz else -1, 
            "kx": abs(kx) if kx else -1,
            "ky": abs(ky) if ky else -1,
            "axisType": axisType,
            "multipoles": [
                str2float(mpoleElement.get("c0")),
                str2float(mpoleElement.get("d1")),
                str2float(mpoleElement.get("d2")),
                str2float(mpoleElement.get("d3")),
                str2float(mpoleElement.get("q11")),
                str2float(mpoleElement.get("q21")),
                str2float(mpoleElement.get("q31")),
                str2float(mpoleElement.get("q22")),
                str2float(mpoleElement.get("q32")),
                str2float(mpoleElement.get("q33")),
            ]
        }
        if "smirks" not in mpoleElement.attrib:
            atypes = self.ff.findAtomTypes(mpoleElement, 1)
            assert len(atypes) == 1
            typeQuery = [str(atypes[0][0])]
            for kString in ['kz', 'kx', 'ky']:
                if paramDict[kString] == -1:
                    break
                typeQuery.append(str(paramDict[kString]))
            typeQuery = tuple(typeQuery)
            self.addParameterWithAtomTypes(typeQuery, paramDict)
        else:
            raise NotImplementedError()
    
    def exportParameterToStr(self):
        mpoleStrs = ['c0', 'd1', 'd2', 'd3', 'q11', 'q21', 'q31', 'q22', 'q32', 'q33']
        strings = []
        for key, value in self._with_atom_types.items():
            atype = key[0]
            kz = self._parameters['kz'][value]
            kx = self._parameters['kx'][value]
            ky = self._parameters['ky'][value]
            typestr = f'type="{atype}"'
            kzstr = f'kz="{kz if kz != -1 else 0}"'
            kxstr = f'kx="{kx if kx != -1 else 0}"'
            kystr = f'ky="{ky if ky != -1 else 0}"'
            axisType = 'axistype="{}"'.format(MultipoleAxisTypeInt2Str[int(self._parameters['axisType'][value])])
            mpoles = self._parameters['multipoles'][value]
            elestr = f'\t\t<Multipole {typestr:<10} {kzstr:<8} {kxstr:<8} {kystr:<8} {axisType:<22}'
            for i, mstr in enumerate(mpoleStrs):
                mstr = f'{mstr}="{float2str(mpoles[i])}"'
                elestr += f"{mstr:<23} "
            elestr += "/>"
            strings.append(elestr)
        
        return '\n'.join(strings)
    
    def createTerms(self, topology: Topology, **kwargs):
        mpoleTerms = TermList(Multipole)

        useSmirks = kwargs.get("useSmirks", False)
        if useSmirks:
            raise NotImplementedError()
        else:
            for atom in topology.atoms():
                kz, kx, ky = -1, -1, -1
                paramIdx = self.getParameterIdxWithAtomType((atom.atomType,))
                
                if paramIdx is None:
                    neighbors = atom.getNeighbors()
                    for nei in neighbors:
                        paramIdx = self.getParameterIdxWithAtomType((atom.atomType, nei.atomType))
                        if paramIdx is not None:
                            kz = nei.idx
                            break
                
                if paramIdx is None:
                    for nei1, nei2 in itertools.permutations(neighbors, 2):
                        paramIdx = self.getParameterIdxWithAtomType((atom.atomType, nei1.atomType, nei2.atomType))
                        if paramIdx is not None:
                            kz, kx = nei1.idx, nei2.idx
                            break
                
                if paramIdx is None:
                    for nei1, nei2, nei3 in itertools.permutations(neighbors, 3):
                        paramIdx = self.getParameterIdxWithAtomType((atom.atomType, nei1.atomType, nei2.atomType, nei3.atomType))
                        if paramIdx is not None:
                            kz, kx, ky = nei1.idx, nei2.idx, nei3.idx
                            break
                
                if paramIdx is None:
                    for nei in atom.getNeighbors():
                        for nnei in nei.getNeighbors():
                            if nnei is not nei and nnei is not atom:
                                paramIdx = self.getParameterIdxWithAtomType((atom.atomType, nei.atomType, nnei.atomType))
                            if paramIdx is not None: break
                        if paramIdx is not None: break
                    kz, kx = nei.idx, nnei.idx
                
                if paramIdx is None:
                    self.raise_exception(f"Atom {atom.idx} not matched for multipoles")
                
                param = self.getParameterWithIdx(paramIdx)
                
                term = Multipole(
                    atom.idx,
                    param["multipoles"][0],
                    param["multipoles"][1], param["multipoles"][2], param["multipoles"][3],
                    param["multipoles"][4], param["multipoles"][5], param["multipoles"][6], 
                    param["multipoles"][7], param["multipoles"][8], param["multipoles"][9],
                    param['axisType'],
                    kz, kx, ky,
                    paramIdx=paramIdx
                )
                mpoleTerms.append(term)
        
        return mpoleTerms


class IsotropicPolarizationGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, ["thole", "alpha", "grp"], True)
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(IsotropicPolarizationGenerator)
        for polar in element.findall("Polarize"):
            generator.addPolarize(polar)
    
    def addPolarize(self, polarElement: ET.Element):
        paramDict = {
            "alpha": str2float(polarElement.get("polarizability")),
            "thole": str2float(polarElement.get("thole")),
            "grp": set(polarElement.get(attr) for attr in polarElement.attrib if attr.startswith("pgrp"))
        }
        if "smirks" not in polarElement.attrib:
            atypes = self.ff.findAtomTypes(polarElement, 1)
            self.addParameterWithAtomTypes(atypes, paramDict)
        else:
            self.addParameterWithSmirks(
                polarElement.get("smirks"),
                paramDict
            )
    
    def setPolarizationGroup(self, topology: Topology):
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(atom for atom in topology.atoms())
        for atom in topology.atoms():
            try:
                paramIdx = self.getParameterIdxWithAtomType((atom.atomType,))
            except:
                self.raise_exception(f"Atom {atom.idx} not match")
            
            param = self.getParameterWithIdx(paramIdx)
            for nei in atom.getNeighbors():
                if nei.atomType in param['grp']:
                    graph.add_edge(atom, nei)

        for group in nx.connected_components(graph):
            for atom in group:
                atom.setPolarizationGroup(group)
            
    def createTerms(self, topology: Topology, **kwargs):
        polTerms = TermList(IsotropicPolarization)

        useSmirks = kwargs.get("useSmirks", False)
        if useSmirks:
            raise NotImplementedError()
        else:
            self.setPolarizationGroup(topology)

            for atom in topology.atoms():
                paramIdx = self.getParameterIdxWithAtomType((atom.atomType,))
                param = self.getParameterWithIdx(paramIdx)
                group = [at.idx for at in atom.polarizationGroup]
                group.sort()
                term = IsotropicPolarization(
                    atom.idx, 
                    param['alpha'], 
                    param['thole'], 
                    group,
                    paramIdx=paramIdx
                )
                polTerms.append(term)
        return polTerms
    

Parsers['AmoebaMultipoleForce'] = [
    MultipoleGenerator,
    IsotropicPolarizationGenerator
]


class AmoebaVdwGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, ['r0', 'epsilon', 'reduction'], True)
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(AmoebaVdwGenerator)
        generator.setMetadata("type", element.get("type"))
        generator.setMetadata("radiusrule", element.get("radiusrule"))
        generator.setMetadata("radiustype", element.get("radiustype"))
        generator.setMetadata("radiussize", element.get("radiussize"))
        generator.setMetadata("epsilonrule", element.get('epsilonrule'))
        generator.setMetadata("vdw-13-scale", str2float(element.get("vdw-13-scale")))
        generator.setMetadata("vdw-14-scale", str2float(element.get("vdw-14-scale")))
        generator.setMetadata("vdw-15-scale", str2float(element.get("vdw-15-scale")))
        for vdw in element.findall("Vdw"):
            generator.addVdw(vdw)
    
    def addVdw(self, vdwElement: ET.Element):
        paramDict = {
            "r0": str2float(vdwElement.get("sigma")), # TODO: sigma -> r0
            "epsilon": str2float(vdwElement.get("epsilon")),
            "reduction": str2float(vdwElement.get('reduction'))
        }
        if "smirks" not in vdwElement.attrib:
            atypes = self.ff.findAtomTypes(vdwElement, 1)
            self.addParameterWithAtomTypes(atypes, paramDict)
        else:
            self.addParameterWithSmirks(
                vdwElement.get("smirks"),
                paramDict
            )
    
    def createTerms(self, topology: Topology, **kwargs):
        vdwTerms = TermList(AmoebaVdw147)
        for atom in topology.atoms():
            try:
                paramIdx = self.getParameterIdxWithAtomType((atom.atomType,))
            except:
                self.raise_exception(f"Atom {atom.idx} does not match")
            
            param = self.getParameterWithIdx(paramIdx)

            if param['reduction'] != 1.00:
                neis = list(atom.getNeighbors())
                assert len(neis) == 1
                parentIdx = neis[0].idx
            else:
                parentIdx = -1
            
            term = AmoebaVdw147(atom.idx, param['epsilon'], param['r0'], parentIdx, param['reduction'], paramIdx=paramIdx)
            vdwTerms.append(term)

        return vdwTerms
    
Parsers['AmoebaVdwForce'] = AmoebaVdwGenerator


class AmoebaStretchBendGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, ['th0', 'b01', 'b02', 'kb1', 'kb2'], False)
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(AmoebaStretchBendGenerator)
        bondGenerator = ff.getGeneratorWithClass(AmoebaBondGenerator)
        angleGenerator = ff.getGeneratorWithClass(AmoebaAngleGenerator)
        assert bondGenerator is not None, "AmoebaBondForce is not defined"
        assert angleGenerator is not None, "AmoebaAngleForce is not defined"
        for strbnd in element.findall("StretchBend"):
            if "smirks" in strbnd.attrib:
                raise NotImplementedError("Does not support assign AmoebaStretchBend with SMIRKS")
            atypes = ff.findAtomTypes(strbnd, 3)
            
            bondParam1 = bondGenerator.getParameterWithAtomType((atypes[0][0], atypes[0][1]))
            if bondParam1 is None:
                bondParam1 = bondGenerator.getParameterWithAtomType((atypes[0][1], atypes[0][0]))
            
            bondParam2 = bondGenerator.getParameterWithAtomType((atypes[0][1], atypes[0][2]))
            if bondParam2 is None:
                bondParam2 = bondGenerator.getParameterWithAtomType((atypes[0][2], atypes[0][1]))
            
            angleParam = angleGenerator.getParameterWithAtomType(atypes[0])
            if angleParam is None:
                angleParam = angleGenerator.getParameterWithAtomType(tuple(reversed(atypes[0])))
            
            # The parameter file contain some strbnd terms that will never exist
            if angleParam is None or bondParam1 is None or bondParam2 is None:
                th0 = [-1.0, -1.0, -1.0]
                b01 = -1.0
                b02 = -1.0
            else:
                th0 = angleParam['th0']
                b01 = bondParam1['b0']
                b02 = bondParam2['b0']

            param = {
                "th0": th0,
                "b01": b01,
                "b02": b02,
                "kb1": str2float(strbnd.get("k1")),
                "kb2": str2float(strbnd.get("k2"))
            }
            generator.addParameterWithAtomTypes(atypes, param)
    
    def createTerms(self, topology: Topology, **kwargs):
        strbndTerms = TermList(AmoebaStretchBend)
        
        if kwargs.get("useSmirks", False):
            raise NotImplementedError()
        
        for angle in topology.bondedAtoms[2]:
            atom1, atom2, atom3 = angle[0], angle[1], angle[2]
            paramIdx = None
            
            paramIdx = self.getParameterIdxWithAtomType((atom1.atomType, atom2.atomType, atom3.atomType))
            if paramIdx is None:
                paramIdx = self.getParameterIdxWithAtomType((atom3.atomType, atom2.atomType, atom1.atomType))
            
            if paramIdx is None:
                continue
                # self.raise_exception(f"Angle between {atom1.idx}, {atom2.idx} and {atom3.idx} not matched")
            
            param = self.getParameterWithIdx(paramIdx)
            if len(param['th0']) > 1:
                numHydrogens = 0
                for nei in atom2.getNeighbors():
                    if (nei is not atom1) and (nei is not atom3) and (nei.element.atomicNum == 1):
                        numHydrogens += 1
                th0 = param['th0'][numHydrogens]
            else:
                th0 = param['th0'][0]
            
            term = AmoebaStretchBend(
                atom1.idx, atom2.idx, atom3.idx,
                th0, param['b01'], param['b02'], param['kb1'], param['kb2'],
                paramIdx=paramIdx
            )
            strbndTerms.append(term)
        
        # strbndTerms.sort(key=lambda t: (t.p0, t.p1, t.p2))
        return strbndTerms

Parsers['AmoebaStretchBendForce'] = AmoebaStretchBendGenerator


class AmoebaOutOfPlaneBendGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, ['k'], False)
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(AmoebaOutOfPlaneBendGenerator)
        generator.setMetadata("opbendType", element.get("type"))
        generator.setMetadata("opbendCubic", str2float(element.get("opbend-cubic")))
        generator.setMetadata("opbendQuartic", str2float(element.get("opbend-quartic")))
        generator.setMetadata("opbendPentic", str2float(element.get("opbend-pentic")))
        generator.setMetadata("opbendSextic", str2float(element.get("opbend-sextic")))
        for opbend in element.findall("Angle"):
            generator.addOutofPlaneBend(opbend)
    
    def addOutofPlaneBend(self, ele: ET.Element):
        paramDict = {"k": str2float(ele.get("k"))}
        if "smirks" not in ele.attrib:
            atypes = self.ff.findAtomTypes(ele, 4)
            self.addParameterWithAtomTypes(atypes, paramDict)
        else:
            self.addParameterWithSmirks(
                ele.get("smirks"),
                paramDict
            )
    
    def createTerms(self, topology: Topology, **kwargs):
        # TODO: terms paramIdx not recorded properly
        opbendTerms = TermList(AmoebaOutOfPlaneBend)
        useSmirks = kwargs.get("useSmirks", False)
        if useSmirks:
            raise NotImplementedError()
        else:
            paramIdxs = []
            for atom in topology.atoms():
                neighbors = atom.getNeighbors()
                if len(neighbors) != 3:
                    continue

                trials = [
                    (neighbors[0], atom, neighbors[1], neighbors[2]),
                    (neighbors[1], atom, neighbors[0], neighbors[2]),
                    (neighbors[2], atom, neighbors[0], neighbors[1])
                ]
                paramIdxTmp = []
                termsTmp = []
                for trial in trials:
                    paramIdx = None
                    try:
                        order = [0, 1, 2, 3]
                        paramIdx = self.getParameterIdxWithAtomType(tuple(trial[i].atomType for i in order))
                    except:
                        order = [0, 1, 3, 2]
                        paramIdx = self.getParameterIdxWithAtomType(tuple(trial[i].atomType for i in order))
                    
                    paramIdxTmp.append(paramIdx)
                    if paramIdx is not None:
                        termsTmp.append(AmoebaOutOfPlaneBend(
                            trial[order[0]].idx,
                            trial[order[1]].idx,
                            trial[order[2]].idx,
                            trial[order[3]].idx,
                            self.getParameterWithIdx(paramIdx)['k']
                        ))
                
                if len(termsTmp) == 3:
                    paramIdxs += paramIdxTmp
                    for term in termsTmp:
                        opbendTerms.append(term)
        # opbendTerms.sort(key=lambda t: (t.p1, t.p0, t.p2, t.p3))
        return opbendTerms

Parsers['AmoebaOutOfPlaneBendForce'] = AmoebaOutOfPlaneBendGenerator


class AmoebaPiTorsionGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, ['k'], False)
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(AmoebaPiTorsionGenerator)
        for pitor in element.findall("PiTorsion"):
            generator.addPiTorsion(pitor)
    
    def addPiTorsion(self, ele: ET.Element):
        paramDict = {"k": str2float(ele.get("k"))}
        if "smirks" not in ele.attrib:
            atypes = self.ff.findAtomTypes(ele, 2)
            self.addParameterWithAtomTypes(atypes, paramDict)
        else:
            self.addParameterWithSmirks(
                ele.get("smirks"),
                paramDict
            )
        
    def createTerms(self, topology: Topology, **kwargs):
        pitorTerms = TermList(AmoebaPiTorsion)

        useSmirks = kwargs.get("useSmirks", False)
        if useSmirks:
            raise NotImplementedError()
        else:
            for bond in topology.bondedAtoms[1]:
                atom1, atom2 = bond.atoms[0], bond.atoms[1]
                paramIdx = self.getParameterIdxWithAtomType((atom1.atomType, atom2.atomType))
                if paramIdx is None:
                    paramIdx = self.getParameterIdxWithAtomType((atom2.atomType, atom1.atomType))
                
                if paramIdx is None:
                    continue

                atom1nei = [nei for nei in atom1.getNeighbors() if nei is not atom2]
                atom2nei = [nei for nei in atom2.getNeighbors() if nei is not atom1]
                assert len(atom1nei) == 2 and len(atom2nei) == 2, "Trying to asssign PiTorsion to a non-sp2 atom"

                param = self.getParameterWithIdx(paramIdx)
                term = AmoebaPiTorsion(
                    atom1nei[0].idx,
                    atom1nei[1].idx,
                    atom1.idx,
                    atom2.idx,
                    atom2nei[0].idx,
                    atom2nei[1].idx,
                    param['k'],
                    paramIdx=paramIdx
                )
                pitorTerms.append(term)
        
        # pitorTerms.sort(key=lambda t: (t.p2, t.p3))
        return pitorTerms

Parsers['AmoebaPiTorsionForce'] = AmoebaPiTorsionGenerator


class PeriodicTorsionGenerator(Generator):
    def __init__(self, ff):
        super().__init__(
            ff, 
            ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6'],
            False
        )
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(PeriodicTorsionGenerator)
        for proper in element.findall("Proper"):
            generator.addProperTorsion(proper)
    
    def addProperTorsion(self, ele: ET.Element):
        paramDict = {}
        for i in range(1, 7):
            paramDict[f"phase{i}"] = str2float(ele.get(f"phase{i}", 0.0))
            paramDict[f"k{i}"] = str2float(ele.get(f"k{i}", 0.0))

        if "smirks" not in ele.attrib:
            atypes = self.ff.findAtomTypes(ele, 4)
            self.addParameterWithAtomTypes(atypes, paramDict)
        else:
            self.addParameterWithSmirks(
                ele.get("smirks"),
                paramDict
            )
    
    def createTerms(self, topology: Topology, **kwargs):
        torsionTerms = TermList(PeriodicTorsion)
        
        useSmirks = kwargs.get("useSmirks", False)
        if useSmirks:
            raise NotImplementedError()
        else:
            for torsion in topology.bondedAtoms[3]:
                atom1, atom2, atom3, atom4 = torsion.atoms[0], torsion.atoms[1], torsion.atoms[2], torsion.atoms[3]
                paramIdx = self.getParameterIdxWithAtomType((atom1.atomType, atom2.atomType, atom3.atomType, atom4.atomType))
                if paramIdx is None:
                    paramIdx = self.getParameterIdxWithAtomType((atom4.atomType, atom3.atomType, atom2.atomType, atom1.atomType))
                
                if paramIdx is None:
                    self.raise_exception(f"Torsion {atom1.idx}-{atom2.idx}-{atom3.idx}-{atom4.idx} cannot be matched")

                param = self.getParameterWithIdx(paramIdx)
                param['paramIdx'] = paramIdx
                term = PeriodicTorsion(
                    atom1.idx, atom2.idx, atom3.idx, atom4.idx,
                    **param
                )
                torsionTerms.append(term)
            
        # torsionTerms.sort(key=lambda t: (t.p0, t.p1, t.p2, t.p3))
        return torsionTerms
    
Parsers['PeriodicTorsionForce'] = PeriodicTorsionGenerator


class AnisotropicPolarizationGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, ["thole", "alpha", "grp"], True)
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(AnisotropicPolarizationGenerator)
        for polar in element.findall("Polarize"):
            generator.addPolarize(polar)
    
    def addPolarize(self, polarElement: ET.Element):
        paramDict = {
            "alpha": [
                str2float(polarElement.get("alphaxx", 0.0)),
                str2float(polarElement.get("alphaxy", 0.0)),
                str2float(polarElement.get("alphaxz", 0.0)),
                str2float(polarElement.get("alphayy", 0.0)),
                str2float(polarElement.get("alphayz", 0.0)),
                str2float(polarElement.get("alphazz", 0.0)),
            ],
            "thole": str2float(polarElement.get("thole")),
            "grp": set(polarElement.get(attr) for attr in polarElement.attrib if attr.startswith("pgrp"))
        }
        if "smirks" not in polarElement.attrib:
            atypes = self.ff.findAtomTypes(polarElement, 1)
            self.addParameterWithAtomTypes(atypes, paramDict)
        else:
            self.addParameterWithSmirks(
                polarElement.get("smirks"),
                paramDict
            )
    
    def exportParameterToStr(self):
        strs = []
        astrs = ['alphaxx', 'alphaxy', 'alphaxz', 'alphayy', 'alphayz', 'alphazz']
        tholes = self._parameters['thole']
        alphas = self._parameters['alpha']
        for atype, index in self._with_atom_types.items():
            typestr = f'type="{atype[0]}"'
            alpha = alphas[index]
            tholestr = f'thole="{float2str(tholes[index])}"'
            alphastrs = [f'{astr}="{float2str(alpha[i])}"' for i, astr in enumerate(astrs)]
            elestr = '\t\t<Polarize {:<10} {:<20} {} />'.format(typestr, tholestr, " ".join([f"{astr:<27}" for astr in alphastrs]))
            strs.append(elestr)
        
        return '\n'.join(strs)
    
    def setPolarizationGroup(self, topology: Topology):
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(atom for atom in topology.atoms())
        for atom in topology.atoms():
            try:
                paramIdx = self.getParameterIdxWithAtomType((atom.atomType,))
            except:
                self.raise_exception(f"Atom {atom.idx} not match")
            
            param = self.getParameterWithIdx(paramIdx)
            for nei in atom.getNeighbors():
                if nei.atomType in param['grp']:
                    graph.add_edge(atom, nei)

        for group in nx.connected_components(graph):
            for atom in group:
                atom.setPolarizationGroup(group)
            
    def createTerms(self, topology: Topology, **kwargs):
        polTerms = TermList(AnisotropicPolarization)

        useSmirks = kwargs.get("useSmirks", False)
        if useSmirks:
            raise NotImplementedError()
        else:
            self.setPolarizationGroup(topology)

            for atom in topology.atoms():
                paramIdx = self.getParameterIdxWithAtomType((atom.atomType,))
                param = self.getParameterWithIdx(paramIdx)
                group = [at.idx for at in atom.polarizationGroup]
                group.sort()
                term = AnisotropicPolarization(
                    atom.idx, 
                    param['alpha'][0], param['alpha'][1], param['alpha'][2],
                    param['alpha'][3], param['alpha'][4], param['alpha'][5],  
                    param['thole'], 
                    group,
                    paramIdx=paramIdx
                )
                polTerms.append(term)
        return polTerms
    
    
class MBUCBChargePenetrationGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, ['alpha', 'beta'], False)
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(MBUCBChargePenetrationGenerator)
        for term in element.findall("ChargePenetration"):
            generator.addTerm(term)
    
    def addTerm(self, element: ET.Element):
        paramDict = {
            "alpha": str2float(element.get("alpha")),
            "beta": str2float(element.get("beta"))
        }
        if "smirks" not in element.attrib:
            atypes = self.ff.findAtomTypes(element, 1)
            self.addParameterWithAtomTypes(atypes, paramDict)
        else:
            self.addParameterWithSmirks(
                element.get("smirks"),
                paramDict
            )
    
    def exportParameterToStr(self):
        alphas = self._parameters['alpha']
        betas = self._parameters['beta']
        strs = []
        for atype, index in self._with_atom_types.items():
            typestr = f'type="{atype[0]}"'
            alphastr = f'alpha="{float2str(alphas[index])}"'
            betastr = f'beta="{float2str(betas[index])}"'
            elestr = '\t\t<ChargePenetration {:<10} {:<20} {:<19} />'.format(typestr, alphastr, betastr)
            strs.append(elestr)
        
        return '\n'.join(strs)

    def createTerms(self, topology: Topology, **kwargs):
        terms = TermList(MBUCBChargePenetration)

        useSmirks = kwargs.get("useSmirks", False)
        if useSmirks:
            raise NotImplementedError()
        else:
            for atom in topology.atoms():
                paramIdx = self.getParameterIdxWithAtomType((atom.atomType,))
                param = self.getParameterWithIdx(paramIdx)
                term = MBUCBChargePenetration(
                    atom.idx, 
                    param['alpha'], 
                    param['beta'], 
                    paramIdx=paramIdx
                )
                terms.append(term)

        return terms


Parsers['MBUCBMultipoleForce'] = [
    MultipoleGenerator,
    AnisotropicPolarizationGenerator,
    MBUCBChargePenetrationGenerator
]


class MBUCBChargeTransferGenerator(Generator):
    def __init__(self, ff):
        super().__init__(ff, ['b', 'd', 'alpha'], False)
    
    @staticmethod
    def parseElement(element: ET.Element, ff: ForceField):
        generator = ff.addGeneratorWithClass(MBUCBChargeTransferGenerator)
        for term in element.findall("ChargeTransfer"):
            generator.addTerm(term)
    
    def addTerm(self, element: ET.Element):
        paramDict = {
            "b": str2float(element.get("b")),
            "d": str2float(element.get("d")),
            "alpha": str2float(element.get("alpha"))
        }
        if "smirks" not in element.attrib:
            atypes = self.ff.findAtomTypes(element, 1)
            self.addParameterWithAtomTypes(atypes, paramDict)
        else:
            self.addParameterWithSmirks(
                element.get("smirks"),
                paramDict
            )
    
    def exportParameterToStr(self):
        ds = self._parameters['d']
        bs = self._parameters['b']
        alphas = self._parameters['alpha']
        strs = []
        for atype, index in self._with_atom_types.items():
            typestr = f'type="{atype[0]}"'
            bstr = f'b="{float2str(bs[index])}"'
            dstr = f'd="{float2str(ds[index])}"'
            alphastr = f'alpha="{float2str(alphas[index])}"'
            elestr = '\t\t<ChargeTransfer {:<10} {:<16} {:<16} {:<25} />'.format(typestr, dstr, bstr, alphastr)
            strs.append(elestr)
        
        return '\n'.join(strs)
    
    def createTerms(self, topology: Topology, **kwargs):
        terms = TermList(MBUCBChargeTransfer)

        useSmirks = kwargs.get("useSmirks", False)
        if useSmirks:
            raise NotImplementedError()
        else:
            for atom in topology.atoms():
                paramIdx = self.getParameterIdxWithAtomType((atom.atomType,))
                param = self.getParameterWithIdx(paramIdx)
                term = MBUCBChargeTransfer(
                    atom.idx, 
                    param['d'], 
                    param['b'],
                    param['alpha'],
                    paramIdx=paramIdx
                )
                terms.append(term)

        return terms


Parsers['MBUCBChargeTransferForce'] = MBUCBChargeTransferGenerator
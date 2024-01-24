from typing import List, Dict, Any
from dataclasses import dataclass
from itertools import product, combinations
import os
import xml
import xml.etree.ElementTree as ET
import warnings

from ..system import System
from ..topology import Topology
from ..template import TEMPLATES
from ..terms import TermList, Particle


def str2float(string):
    return float(string)


def str2int(string):
    return int(string)


def float2str(number):
    if number == 0.0 or abs(number) > 1e-5:
        return f"{number:.10f}"
    else:
        return f"{number:.10e}"


def str2bool(string):
    if string == "True" or string == "true":
        return True
    elif string == "False" or string == 'false':
        return False
    else:
        raise ValueError(f"Could not convert string to bool: {string}")


def xmlele2str(xmlele: ET.Element):
    uglystr = ET.tostring(xmlele, "unicode")
    return uglystr.strip()


@dataclass
class AtomType:
    name: str
    atomClass: str


# Parsers for force XML elements
Parsers = {}


class ForceField:
    def __init__(self, *files):
        self.files = self.processFileNames(files)
        self.trees = [ET.parse(f) for f in self.files]

        self.atomTypes: Dict[str, AtomType] = {}
        self.atomClasses: Dict[str, List[AtomType]] = {}
        self._generators: List[Generator] = []
        self._forces: List[str] = []
        
        self.loadAtomTypes()
        self.loadAtomTypeDefs()
        self.loadForces()
    
    def processFileNames(self, files):
        dirname = os.path.dirname(__file__)
        files = list(files) if isinstance(files, tuple) else [files]
        for i in range(len(files)):
            if not os.path.exists(files[i]):
                trial = os.path.join(dirname, files[i])
                if not os.path.isfile(trial):
                    raise FileNotFoundError()
                else:
                    files[i] = trial
        return files

    @property
    def generators(self):
        return self._generators
    
    def addGenerator(self, generator):
        self._generators.append(generator)
    
    def getGeneratorWithClass(self, generatorClass):
        idx = None
        for i in range(len(self.generators)):
            if isinstance(self.generators[i], generatorClass):
                idx = i
                break
        if idx is None:
            return None
        else:
            return self.generators[i]
    
    def addGeneratorWithClass(self, generatorClass):
        generator = self.getGeneratorWithClass(generatorClass)
        if generator is None:
            generator = generatorClass(self)
            self.addGenerator(generator)
        return generator
           
    def addAtomType(self, atomTypeElement: ET.Element):
        name = atomTypeElement.get("name")
        aclass = atomTypeElement.get("class", "")
        atype = AtomType(name, aclass)
        if name in self.atomTypes:
            raise Exception(f"Duplicated atom type: {name}")
        self.atomTypes[name] = atype
        cls2type = self.atomClasses.get(aclass, [])
        cls2type.append(atype)
        self.atomClasses[aclass] = cls2type
    
    def loadAtomTypes(self):
        for tree in self.trees:
            atomTypes = tree.getroot().find("AtomTypes")
            for atomTypeElement in atomTypes.findall("Type"):
                self.addAtomType(atomTypeElement)
    
    def loadAtomTypeDefs(self):
        for tree in self.trees:
            residues = tree.getroot().find("Residues")
            for res in residues.findall("Residue"):
                ## TODO: GET ALL RESIDUES COMPLETED
                if res.get("name") not in TEMPLATES:
                    continue
                template = TEMPLATES[res.get("name")]
                for atom in res.findall("Atom"):
                    name = atom.get("name")
                    atype = atom.get('type')
                    template.setAtomType(name, atype)
    
    def loadForces(self):
        for tree in self.trees:
            for child in tree.getroot():
                if child.tag in Parsers:
                    self._forces.append(child.tag)
                    if isinstance(Parsers[child.tag], list):
                        for parser in Parsers[child.tag]:
                            parser.parseElement(child, self)
                    else:
                        Parsers[child.tag].parseElement(child, self)
                elif child.tag in ["Info", "Residues", "AtomTypes"]:
                    pass
                else:
                    pass
                    # raise ValueError(f"{child.tag} is not supported")

    def assignAtomTypes(self, topology: Topology):
        for res in topology.residues:
            if res.stdName not in TEMPLATES:
                raise KeyError(f"ResidueTemplate {res.stdName} not defined")
            template = TEMPLATES[res.stdName]
            for atom in res.atoms:
                atype = template.getAtomType(atom.name)
                atom.setAtomType(atype)
                aclass = self.atomTypes[atype].atomClass
                atom.setAtomClass(aclass)
    
    def findAtomTypes(self, element: ET.Element, numAtoms: int):
        useType = any(key.startswith("type") for key in element.attrib.keys())
        useClass = any(key.startswith("class") for key in element.attrib.keys())
        
        if useType and useClass:
            raise ValueError(f"Specified both a type and a class for the same atom: {element.attrib}")
        elif (not useType) and (not useClass):
            raise ValueError(f"Either a type or class has to be specified for: {element.attrib}")
        
        atypes = []
        for i in range(numAtoms):
            suffix = "" if numAtoms == 1 else str(i+1)
            if useType:
                atype = element.get(f"type{suffix}")
                if atype == "":
                    # handle wild card
                    atypes.append(["*"])
                else:
                    atypes.append([self.atomTypes[atype].name])
            else:
                aclass = element.get(f"class{suffix}")
                if aclass == "":
                    # handle wild card
                    atypes.append(["*"])
                else:
                    atypes.append([atype.name for atype in self.atomClasses[aclass]])
        
        atypes = list(product(*atypes))
        return atypes
    
    def createSystem(self, topology: Topology, **kwargs):

        self.assignAtomTypes(topology)
        system = System()
        system.addMeta("name", topology.name)
        particles = TermList(Particle)
        for atom in topology.atoms():
            particles.append(Particle(
                atom.idx, atom.name, atom.symbol, atom.mass, 
                f"{atom.residue.number}{atom.residue.insertionCode}", atom.residue.name,
                atom.xx, atom.xy, atom.xz
            ))
        system.addTerms(particles)
        for generator in self.generators:
            for key, value in generator._meta.items():
                system.addMeta(key, value)
            terms = generator.createTerms(topology, **kwargs)
            if isinstance(terms, tuple):
                for t in terms:
                    system.addTerms(t)
            else:
                system.addTerms(terms)
        return system
    
    def getParameters(self, asJaxNumpy: bool = True):
        params = {}
        for generator in self.generators:
            name = generator.__class__.__name__
            params[name] = generator.getParameters(asJaxNumpy=asJaxNumpy)
        self._parameters = params
        return self._parameters
    
    def updateParameters(self, param: Dict[str, Any]):
        for generator in self.generators:
            name = generator.__class__.__name__
            generator.updateParameters(param[name])
    
    def exportAtomTypes(self):
        strs = []
        for tree in self.trees:
            atomTypes = tree.getroot().find("AtomTypes")
            for atomTypeElement in atomTypes.findall("Type"):
                strs.append(f'\t\t{xmlele2str(atomTypeElement)}')
        atypestr = '\t<AtomTypes>\n{}\n\t</AtomTypes>'.format('\n'.join(strs))
        return atypestr
    
    def exportAtomTypeDefs(self):
        strs = []
        for tree in self.trees:
            residues = tree.getroot().find("Residues")
            for res in residues.findall("Residue"):
                restr = '\t\t<Residue name="{}">\n{}\n\t\t</Residue>'.format(
                    res.get("name"),
                    '\n'.join(f'\t\t\t{xmlele2str(atomEle)}' for atomEle in res.findall("Atom"))
                )
                strs.append(restr)
        return '\t<Residues>\n{}\n\t</Residues>'.format('\n'.join(strs))

    def save(self, path: os.PathLike):
        forcestrs = []
        for force in self._forces:
            if isinstance(Parsers[force], list):
                forcestr = "\n".join([
                    self.getGeneratorWithClass(gencls).exportParameterToStr() for gencls in Parsers[force]
                ])
            else:
                forcestr = self.getGeneratorWithClass(Parsers[force]).exportParameterToStr()
            forcestrs.append(f"\t<{force}>\n{forcestr}\n\t</{force}>")
        ffstr = "<ForceField>\n{}\n{}\n{}\n</ForceField>".format(
            self.exportAtomTypes(),
            self.exportAtomTypeDefs(),
            '\n'.join(forcestrs)
        )
        with open(path, 'w') as f:
            f.write(ffstr)

    
class Generator:
    """
    Base class for a force generator
    """
    def __init__(self, ff: ForceField, paramFields: List[str] = [], raiseError: bool = True):
        self.ff = ff
        # parameters
        self._with_atom_types = {}
        self._with_smirks = {}
        self._parameters: Dict[str, List[Any]] = {}
        self._meta = {}
        self._raiseError = raiseError
        for fd in paramFields:
            self.addParameterField(fd)
    
    def getMetadata(self, key: str):
        return self._meta[key]
    
    def setMetadata(self, key: str, value: Any):
        self._meta[key] = value
    
    @property
    def paramFields(self) -> List[str]:
        return sorted(list(self._parameters.keys()))
    
    @property
    def numParameters(self) -> int:
        key = list(self._parameters.keys())[0]
        return len(self._parameters[key])
    
    def addParameterField(self, name: str):
        if name not in self._parameters:
            self._parameters[name] = []
    
    def checkParameter(self, paramDict: Dict[str, Any]):
        keys = sorted(list(paramDict.keys()))
        assert keys == self.paramFields, f"Parameter fields does not match {keys} != {self.paramFields}"

    def addParameterWithAtomTypes(self, typeOrtypes, paramDict: Dict[str, Any]):
        types = typeOrtypes if isinstance(typeOrtypes, list) else [typeOrtypes]
        for typ in types:
            self._with_atom_types[typ] = self.numParameters
        self.setParameterWithIdx(paramDict, self.numParameters)
    
    def addParameterWithSmirks(self, smirks: str, paramDict: Dict[str, Any]):
        self._with_smirks[smirks] = self.numParameters
        self.setParameterWithIdx(paramDict, self.numParameters)
        
    def setParameterWithIdx(self, paramDict: Dict[str, Any], paramIdx: int):
        self.checkParameter(paramDict)
        if paramIdx >= self.numParameters:
            for k, v in paramDict.items():
                self._parameters[k].append(v)
        else:
            for k, v in paramDict.items():
                self._parameters[k][paramIdx] = v

    def getParameterIdxWithAtomType(self, typeQuery):
        paramIdx = self._with_atom_types.get(typeQuery, None)
        if paramIdx is None:
            # try wildcard match
            for numWildCard in range(1, 1+len(typeQuery)):
                for wildCardPos in combinations(range(len(typeQuery)), numWildCard):
                    typeQueryWithWildCard = list(typeQuery)
                    for p in wildCardPos:
                        typeQueryWithWildCard[p] = "*"
                    paramIdx = self._with_atom_types.get(tuple(typeQueryWithWildCard), None)
                    if paramIdx is not None:
                        break
                if paramIdx is not None:
                    break
        
        if (paramIdx is None) and self._raiseError:
            raise KeyError(typeQuery)
        return paramIdx
    
    def getParameterWithAtomType(self, typeQuery):
        paramIdx = self.getParameterIdxWithAtomType(typeQuery)
        if paramIdx is None:
            return None
        else:
            return self.getParameterWithIdx(paramIdx)

    def getParameterIdxWithSmirks(self, smirksQuery: str):
        if self._raiseError:
            paramIdx = self._with_smirks[smirksQuery]
        else:
            paramIdx = self._with_smirks.get(smirksQuery, None)
        return paramIdx
    
    def getParameterWithSmirks(self, smirksQuery: str):
        paramIdx = self.getParameterIdxWithSmirks(smirksQuery)
        if paramIdx is None:
            return None
        else:
            return self.getParameterWithIdx(paramIdx)
    
    def getParameterWithIdx(self, idx: int):
        return {k: self._parameters[k][idx] for k in self._parameters.keys()}
    
    def getParameters(self, asJaxNumpy: bool = False):
        if asJaxNumpy:
            import jax.numpy as jnp    
            jaxParam = {}
            for key, value in self._parameters.items():
                try:
                    jaxParam[key] = jnp.array(value)
                except TypeError as e:
                    continue
            return jaxParam
        else:
            return self._parameters
    
    def updateParameters(self, param: Dict[str, Any]):
        self._parameters.update(param)

    def createTerms(self, topology: Topology, **kwargs):
        raise NotImplementedError()

    def raise_exception(self, msg: str, raiseError: bool = True):
        if raiseError:
            raise Exception(msg)
        else:
            warnings.warn(msg)
    
    def exportParameterToStr(self):
        return ""
        # raise NotImplementedError()

import os
import glob
import xml.etree.ElementTree as ET
from typing import List, Dict, Any


TEMPLATES = {}


class ResidueTemplate:
    def __init__(self, 
                 name: str, 
                 atoms: Dict[str, Dict[str, Any]], 
                 bonds: List[Dict[str, Any]], 
                 altNames: List[str] = []):
        self.name = name
        self.atoms = atoms
        self.bonds = bonds
        self.altNames = altNames
    
    @classmethod
    def fromXMLData(cls, data: ET.Element):
        name = data.get("name")
        altNames = [data.get(attrName) for attrName in data.attrib if attrName.startswith("altname")]
        atoms = {}
        
        for ele in data.findall("Atom"):
            atomName = ele.get("name")
            if atomName in atoms:
                raise RuntimeError(f"Atom name duplicated: {atomName}")
            atoms[atomName] = {
                "altNames": [ele.get(attrName) for attrName in ele.attrib if attrName.startswith("altname")]
            }
        
        bonds = []
        for ele in data.findall("Bond"):
            bonds.append({
                "atom1": ele.get("from"),
                "atom2": ele.get("to"),
                "order": float(ele.get('order'))
            })
        
        return cls(name=name, atoms=atoms, bonds=bonds, altNames=altNames)
    
    def setAtomType(self, atomName: str, atomType: str):
        self.atoms[atomName]['atomType'] = atomType
    
    def getAtomType(self, atomName: str):
        return self.atoms[atomName]['atomType']


def loadTemplateDefinitions(fname: os.PathLike):
    xmlobj = ET.parse(fname)
    root = xmlobj.getroot()
    for ele in root.findall("Residue"):
        name = ele.get("name")
        altnames = [ele.get(attrName) for attrName in ele.attrib if attrName.startswith("altname")]
        for key in [name] + altnames:
            if key in TEMPLATES:
                raise RuntimeError(f"Duplicated template: {key}")
            TEMPLATES[key] = ResidueTemplate.fromXMLData(ele)


for fxml in glob.glob(os.path.join(os.path.dirname(__file__), "templates/*.xml")):
    loadTemplateDefinitions(fxml)
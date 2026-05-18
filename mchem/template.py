"""Residue templates (atoms and bonds) loaded from XML; used for topology matching."""

import os
import glob
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

TEMPLATES = {}


class ResidueTemplate:
    """
    Template for a residue: atom names (with alternates) and bond list for matching.
    """

    def __init__(
        self,
        name: str,
        atoms: Dict[str, Dict[str, Any]],
        bonds: List[Dict[str, Any]],
        altNames: List[str] = [],
    ):
        """
        Parameters
        ----------
        name : str
            Residue template name.
        atoms : dict
            Map atom name -> dict with ``altNames`` (and optionally ``atomType``).
        bonds : list
            List of dicts with keys ``atom1``, ``atom2``, ``order``.
        altNames : list, optional
            Alternative names for this template.
        """
        self.name = name
        self.atoms = atoms
        self.bonds = bonds
        self.altNames = altNames

    @classmethod
    def fromXMLData(cls, data: ET.Element):
        """Build a :class:`ResidueTemplate` from an XML ``Residue`` element (name, Atom, Bond children)."""
        name = data.get("name")
        altNames = [
            data.get(attrName)
            for attrName in data.attrib
            if attrName.startswith("altname")
        ]
        atoms = {}

        for ele in data.findall("Atom"):
            atomName = ele.get("name")
            if atomName in atoms:
                raise RuntimeError(f"Atom name duplicated: {atomName}")
            atoms[atomName] = {
                "altNames": [
                    ele.get(attrName)
                    for attrName in ele.attrib
                    if attrName.startswith("altname")
                ],
                **ele.attrib,  # charges from residue template
            }

        bonds = []
        for ele in data.findall("Bond"):
            bond: dict[str, float | str | None] = {
                "atom1": ele.get("from") or ele.get("atomName1"),
                "atom2": ele.get("to") or ele.get("atomName2"),
                "order": float(ele.get("order") or 1.0),
            }
            bonds.append(bond)

        for ele in data.findall("ExternalBond"):
            if ele.get("atomName") == "N":
                bond = {"atom1": "-C", "atom2": "N", "order": 1.0}
                bonds.append(bond)
            elif ele.get("atomName") == "C":
                pass

        return cls(name=name, atoms=atoms, bonds=bonds, altNames=altNames)

    def setAtomType(self, atomName: str, atomType: str):
        """Set force-field atom type for the given atom name."""
        self.atoms[atomName]["atomType"] = atomType

    def getAtomType(self, atomName: str):
        """Return force-field atom type for the given atom name."""
        return self.atoms[atomName]["atomType"]

    def getAtom(self, atomName: str) -> dict:
        "Return force-field atom info"
        return self.atoms[atomName]


def loadTemplateDefinitions(fname: os.PathLike):
    """
    Load residue templates from an XML file and add them to :data:`TEMPLATES`.

    Parameters
    ----------
    fname : os.PathLike
        Path to XML file containing ``Residue`` elements.
    """
    xmlobj = ET.parse(fname)
    root = xmlobj.getroot()
    for ele in root.findall("Residue"):
        name = ele.get("name")
        altnames = [
            ele.get(attrName)
            for attrName in ele.attrib
            if attrName.startswith("altname")
        ]
        for key in [name] + altnames:
            if key in TEMPLATES:
                raise RuntimeError(f"Duplicated template: {key}")
            TEMPLATES[key] = ResidueTemplate.fromXMLData(ele)


def loadNamedTemplateDefinitions(ffname: str):
    for fxml in glob.glob(
        os.path.join(os.path.dirname(__file__), f"templates/{ffname}/*.xml")
    ):
        loadTemplateDefinitions(fxml)


for fxml in glob.glob(os.path.join(os.path.dirname(__file__), "templates/*.xml")):
    loadTemplateDefinitions(fxml)

import os
import pytest
import xml.etree.ElementTree as ET

from mchem.template import ResidueTemplate, loadTemplateDefinitions, TEMPLATES
from mchem.topology import Topology, Residue, Bond, Atom


@pytest.fixture(scope="session")
def water_template():
    xmlstr = """<Residues>
  <Residue name="HOH" altname1="SOL" altname2="WAT" >
    <Atom name="O" altname1="OW" />
    <Atom name="H1" altname1="HW1" />
    <Atom name="H2" altname1="HW2" />
    <Bond from="O" to="H1" order="1" />
    <Bond from="O" to="H2" order="1" />
  </Residue>
</Residues>
"""
    root = ET.fromstring(xmlstr)
    res = root.find("Residue")
    tmpl = ResidueTemplate.fromXMLData(res)
    return tmpl


def test_init_template(water_template):
    assert water_template.name == "HOH"
    assert "SOL" in water_template.altNames
    assert "WAT" in water_template.altNames
    assert water_template.atoms["O"]["altNames"] == ["OW"]
    assert water_template.atoms["H1"]["altNames"] == ["HW1"]
    assert water_template.atoms["H2"]["altNames"] == ["HW2"]
    assert water_template.bonds[0]['atom1'] == "O"
    assert water_template.bonds[0]['atom2'] == "H1"
    assert water_template.bonds[0]['order'] == 1.0
    assert water_template.bonds[1]['atom1'] == "O"
    assert water_template.bonds[1]['atom2'] == "H2"
    assert water_template.bonds[1]['order'] == 1.0


def test_match_template(water_template):
    top = Topology(name='water')
    with top.setEditable():
        atoms = {
            "OW": Atom("OW", "O"),
            "HW1": Atom("HW1", "H"),
            "HW2": Atom("HW2", "H")
        }
        residue = Residue("WAT", 1)
        for name, atom in atoms.items():
            residue.addAtom(atom)
        top.addResidue(residue)
        residue.matchTemplate(water_template)
    
    assert residue.name == "HOH"
    assert atoms['OW'].name == "O"
    assert atoms['HW1'].name == "H1"
    assert atoms['HW2'].name == "H2"
    assert len(top.bonds) == 2


def test_load_tempalte():
    fxml = os.path.join(os.path.dirname(__file__), 'nh3.xml')
    xmlstr = """<Residues>
  <Residue name="NH3" >
    <Atom name="N" />
    <Atom name="H1" />
    <Atom name="H2" />
    <Atom name="H3" />
    <Bond from="N" to="H1" order="1" />
    <Bond from="N" to="H2" order="1" />
    <Bond from="N" to="H3" order="1" />
  </Residue>
</Residues>
"""
    with open(fxml, 'w') as f:
        f.write(xmlstr)
    
    loadTemplateDefinitions(fxml)
    assert "NH3" in TEMPLATES
    os.remove(fxml)


def test_top_match_template():
    top = Topology("water")
    with top.setEditable():
        res = Residue("HOH", 1)
        res.addAtom(Atom("OW", "O"))
        res.addAtom(Atom("HW1", "H"))
        res.addAtom(Atom("HW2", "H"))
        top.addResidue(res)
        top.matchTemplates()
    assert len(top.bonds) == 2
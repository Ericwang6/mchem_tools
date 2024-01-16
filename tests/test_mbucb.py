import pytest
from mchem.system import System
from mchem.fileformats import load_pdb
from mchem.forcefield import ForceField

def test_mbucb():
    topology = load_pdb("tests/data/ace_nme_water.pdb")
    ff = ForceField("mbucb.xml")
    system = ff.createSystem(topology)
    paramDict = ff.getParameters(asJaxNumpy=True)
    ff.updateParameters(paramDict)
    # ff.save("check.xml")
    # system.save("tests/data/ace_nme_water.db", overwrite=True)

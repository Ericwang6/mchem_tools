import pytest
import jax.numpy as jnp
from mchem.system import System
from mchem.fileformats import load_pdb
from mchem.forcefield import ForceField

def test_mbucb():
    topology = load_pdb("tests/data/ace_nme_water.pdb")
    ff = ForceField("mbucb.xml")
    system = ff.createSystem(topology)
    paramDict = ff.getParameters(asJaxNumpy=True)
    paramDict["AmoebaVdwGenerator"]['sigma'] *= 10
    paramDict['AmoebaVdwGenerator']['epsilon'] /= 4.184
    ff.updateParameters(paramDict)
    ff.save("check.xml")

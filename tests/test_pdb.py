import pytest
import os
from mchem.fileformats import load_pdb


def test_load_pdb():
    path = os.path.dirname(__file__)
    top = load_pdb(os.path.join(path, "data/water.pdb"))
    assert top.numBonds == 4
    assert top.numAtoms == 6
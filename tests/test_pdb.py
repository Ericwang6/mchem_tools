import os

import pytest

from mchem.fileformats import load_pdb, read_pdb_box


def test_load_pdb():
    path = os.path.dirname(__file__)
    top = load_pdb(os.path.join(path, "data/water.pdb"))
    assert top.numBonds == 4
    assert top.numAtoms == 6


def test_read_pdb_box_present():
    path = os.path.dirname(__file__)
    box = read_pdb_box(os.path.join(path, "data/water.pdb"))
    assert box is not None
    a, b, c, alpha, beta, gamma = box
    assert (a, b, c) == (100.0, 100.0, 100.0)
    assert (alpha, beta, gamma) == (90.0, 90.0, 90.0)


def test_read_pdb_box_absent():
    path = os.path.dirname(__file__)
    box = read_pdb_box(os.path.join(path, "data/water216.pdb"))
    assert box is None
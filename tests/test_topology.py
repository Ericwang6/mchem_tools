import pytest
import numpy as np
from mchem.topology import Residue, Topology, Atom, Bond, BondedAtoms
from pprint import pprint


def test_topological_info():
    top = Topology("ethane")
    with top.setEditable():
        res = Residue("UNK", 1)
        cs = [Atom("C1", "C"), Atom("C2", "C")]
        hs = [Atom(f"H{i+1}", "H") for i in range(6)]
        bonds = [Bond(cs[0], cs[1], 1)]
        for i in range(2):
            c = cs[i]
            for j in range(3):
                h = hs[i*3+j]
                bond = Bond(c, h, 1)
                bonds.append(bond)
        
        for atom in cs:
            res.addAtom(atom)
        for atom in hs:
            res.addAtom(atom)
        top.addResidue(res)
        for bond in bonds:
            top.addBond(bond)
    
    # Atom-level Top Info
    refBondedAtoms = {cs[0]: 1, cs[1]: 2, hs[1]: 2, hs[2]: 2, hs[3]: 3, hs[4]: 3, hs[5]: 3}
    assert hs[0].bondedAtoms == refBondedAtoms

    refPathsToBondedAtoms = {
        1: set([BondedAtoms([hs[0], cs[0]])]),
        2: set([
            BondedAtoms([hs[0], cs[0], cs[1]]), 
            BondedAtoms([hs[0], cs[0], hs[1]]), 
            BondedAtoms([hs[0], cs[0], hs[2]])
        ]),
        3: set([
            BondedAtoms([hs[0], cs[0], cs[1], hs[3]]),
            BondedAtoms([hs[0], cs[0], cs[1], hs[4]]),
            BondedAtoms([hs[0], cs[0], cs[1], hs[5]]),
        ]),
        4: set(),
        5: set(),
    }
    
    assert refPathsToBondedAtoms == hs[0].pathsToBondedAtoms
    
    # Topology-level Top Info
    refBonedAtomsTopLevel = {
        1: set([
            BondedAtoms([hs[0], cs[0]]),
            BondedAtoms([hs[1], cs[0]]),
            BondedAtoms([hs[2], cs[0]]),
            BondedAtoms([hs[3], cs[1]]),
            BondedAtoms([hs[4], cs[1]]),
            BondedAtoms([hs[5], cs[1]]),
            BondedAtoms([cs[0], cs[1]])    
        ]),
        2: set([
            BondedAtoms([hs[0], cs[0], cs[1]]),
            BondedAtoms([hs[0], cs[0], hs[1]]),
            BondedAtoms([hs[0], cs[0], hs[2]]),
            BondedAtoms([hs[1], cs[0], cs[1]]),
            BondedAtoms([hs[1], cs[0], hs[2]]),
            BondedAtoms([hs[2], cs[0], cs[1]]),
            BondedAtoms([hs[3], cs[1], cs[0]]),
            BondedAtoms([hs[3], cs[1], hs[4]]),
            BondedAtoms([hs[3], cs[1], hs[5]]),
            BondedAtoms([hs[4], cs[1], cs[0]]),
            BondedAtoms([hs[4], cs[1], hs[5]]),
            BondedAtoms([hs[5], cs[1], cs[0]]),         
        ]),
        3: set([
            BondedAtoms([hs[0], cs[0], cs[1], hs[3]]),
            BondedAtoms([hs[0], cs[0], cs[1], hs[4]]),
            BondedAtoms([hs[0], cs[0], cs[1], hs[5]]),
            BondedAtoms([hs[1], cs[0], cs[1], hs[3]]),
            BondedAtoms([hs[1], cs[0], cs[1], hs[4]]),
            BondedAtoms([hs[1], cs[0], cs[1], hs[5]]),
            BondedAtoms([hs[2], cs[0], cs[1], hs[3]]),
            BondedAtoms([hs[2], cs[0], cs[1], hs[4]]),
            BondedAtoms([hs[2], cs[0], cs[1], hs[5]]),
        ]),
        4: set(),
        5: set()
    }
    assert top.bondedAtoms == refBonedAtomsTopLevel

    refConnTable = []
    for numConnect in refBonedAtomsTopLevel:
        for p in refBonedAtomsTopLevel[numConnect]:
            if p.atoms[0].idx < p.atoms[-1].idx:
                refConnTable.append([p.atoms[0].idx, p.atoms[-1].idx, numConnect])
            else:
                refConnTable.append([p.atoms[-1].idx, p.atoms[0].idx, numConnect])
    refConnTable.sort(key=lambda x: (x[-1], x[0], x[1]))
    np.testing.assert_equal(refConnTable, top.connTable)
    


def test_editable():
    top = Topology("edit")

    with pytest.raises(RuntimeError):
        res = Residue("UNK", 1)
        top.addResidue(res)

    assert not top.editable
    with top.setEditable():
        assert top.editable
    assert not top.editable


def test_remove_atoms():
    top = Topology("test_remove_atoms")
    with top.setEditable():
        res = Residue("UNK", 1)
        at1 = Atom("H1", "H")
        at2 = Atom("H2", "H")
        res.addAtom(at1)
        res.addAtom(at2)
        top.addResidue(res)
        top.addBond(Bond(at1, at2, 1))
    assert len([at for at in top.atoms()]) == 2
    assert len(top.bonds) == 1
    assert len(top.getAtomWithIdx(0).getNeighbors()) == 1

    with top.setEditable():
        res = top.getResidueWithIdx(0)
        res.removeAtom(res.atoms[1])
    assert len([at for at in top.atoms()]) == 1
    assert len(top.bonds) == 0
    assert len(top.getAtomWithIdx(0).getNeighbors()) == 0


def test_bonded_atoms():
    atoms = [
        Atom("H1", "H"), Atom("O", "O"), Atom("H2", "H")
    ]
    angleAtoms = BondedAtoms(atoms)
    angleAtoms2 = BondedAtoms([atoms[-1], atoms[-2], atoms[-3]])
    
    # test set eq
    testSet = set()
    testSet.add(angleAtoms)
    testSet.add(angleAtoms2)
    assert len(testSet) == 1

    # test dict
    testDict = dict()
    testDict[angleAtoms] = 1
    assert testDict[angleAtoms2] == 1
    diheAtoms = angleAtoms.extend([Atom("C", "C")])
    assert angleAtoms == angleAtoms2
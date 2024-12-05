import argparse
from mchem.forcefield import ForceField
from mchem.fileformats import load_pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser("M-Chem Front-end: Convert PDB to SQLite-DB formatted parameters")
    parser.add_argument('-i', '--input', dest='input', help='Input PDB')
    parser.add_argument('-f', '--forcefield', dest='forcefield', help='ForceField XML', default='amoebabio18.xml')
    parser.add_argument('-o', '--output', dest='output', help='Output DB file')
    args = parser.parse_args()
    top = load_pdb(args.input)
    ff = ForceField(args.forcefield)
    system = ff.createSystem(top)
    system.save(args.output)
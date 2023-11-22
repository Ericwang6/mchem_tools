import os
from pathlib import Path
from ..topology import Bond, Atom, Topology, Residue



def load_pdb(fname: os.PathLike):
    top = Topology(name=Path(fname).stem)
    residues = []
    with open(fname, 'r') as f:
        prevSig = ""
        for line in f:
            record = line[:6].strip()
            if record == "ATOM" or record == "HETATM":
                atIdx = int(line[6:11])
                atName = line[12:16].strip()
                altLoc = line[16].strip()
                resName = line[17:20].strip()
                chain = line[21]
                resNum = int(line[22:26])
                iCode = line[26]
                xx = float(line[30:38])
                xy = float(line[38:46])
                xz = float(line[46:54])
                occ = float(line[54:60])
                tempFactor = float(line[60:66])
                element = line[76:78].strip()
                
                try:
                    charge = float(line[78:80])
                except ValueError:
                    charge = 0.0
                
                sig = f"{resName}{resNum}{iCode}/{chain}"
                if sig != prevSig:
                    residues.append(Residue(resName, resNum, chain, iCode))
                if (not altLoc) or (altLoc == "A"):
                    res = residues[-1]
                    atom = Atom(atName, element)
                    atom.setPosition([xx, xy, xz])
                    res.addAtom(atom)
                prevSig = sig
    
    with top.setEditable():
        for res in residues:
            top.addResidue(res)
        top.matchTemplates()
    
    return top



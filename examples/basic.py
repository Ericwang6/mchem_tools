from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mchem.forcefield import ForceField
from mchem.fileformats import load_pdb


if __name__ == "__main__":
    pdb = Path(__file__).resolve().parent.parent / "tests/data/ace_ala_nme_water.pdb"
    top = load_pdb(pdb)
    ff = ForceField("amoebabio18.xml")
    system = ff.createSystem(top)
    system.save("ace_ala_nme_water.db")
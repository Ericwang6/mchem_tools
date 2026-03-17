"""AMOEBA force field parser for ``.prm`` parameter files."""

import os


class AmoebaForceField:
    """
    Parser and container for AMOEBA force field parameters from a ``.prm`` file.

    Stores atom types, VdW, bonds, angles, stretch-bend, Urey-Bradley,
    out-of-plane bend, torsions, pi-torsions, stretch-torsion, angle-torsion,
    torsion-torsion, multipoles, and polarizability parameters.
    """

    def __init__(self, fname: os.PathLike):
        """
        Load AMOEBA parameters from a ``.prm`` file.

        Parameters
        ----------
        fname : os.PathLike
            Path to the AMOEBA parameter file.
        """
        self.meta = {}
        self.atomType = {}
        self.vdw = {}
        self.vdwpair = {}
        self.bond = {}
        self.angle = {}
        self.strbnd = {}
        self.ureybrad = {}
        self.opbend = {}
        self.torsion = {}
        self.pitors = {}
        self.strtors = {}
        self.angtors = {}
        self.tortors = {}
        self.multipole = {}
        self.polarize = {}
        self.read_prm(fname)

    def read_prm(self, fname: os.PathLike):
        """
        Parse the given ``.prm`` file and populate all parameter dictionaries.

        Parameters
        ----------
        fname : os.PathLike
            Path to the AMOEBA parameter file.
        """
        mode = 0

        def _parse_meta_line(line):
            if line:
                key, value = tuple(line.split())
                if "." in value:
                    value = float(value)
                self.meta[key] = value
        
        def _parse_atom_type_line(line):
            content = line.split()
            atype = int(content[1])
            aclass = int(content[2])
            name = content[3]
            valence = int(content[-1])
            mass = float(content[-2])
            atnum = int(content[-3])
            desc = " ".join(content[4:-3])
            desc = desc[1:-1]
            self.atomType[atype] = {
                "atomClass": aclass,
                "name": name,
                "valence": valence,
                "mass": mass,
                "atnum": atnum,
                "desc": desc
            }
        
        def _parse_vdw_line(line):
            content = line.split()
            aclass = int(content[1])
            r0 = float(content[2])
            eps = float(content[3])
            reduction = float(content[-1]) if len(content) >= 5 else 0.0
            self.vdw[aclass] = {"r0": r0, "eps": eps, "reduction": reduction}

        def _parse_vdwpair_line(line):
            content = line.split()
            aclass1 = int(content[1])
            aclass2 = int(content[2])
            r0 = float(content[3])
            eps = float(content[4])
            self.vdwpair[(aclass1, aclass2)] = {"r0": r0, "eps": eps}
        
        def _parse_bond_line(line):
            content = line.split()
            aclass1 = int(content[1])
            aclass2 = int(content[2])
            kb = float(content[3])
            b0 = float(content[4])
            self.bond[(aclass1, aclass2)] = {"kb": kb, "b0": b0}
        
        def _parse_angle_line(line):
            content = line.split()
            inPlane = content[0] == "anglep"
            aclass1 = int(content[1])
            aclass2 = int(content[2])
            aclass3 = int(content[3])
            kth = float(content[4])
            th01 = float(content[5])
            if len(content) > 6:
                th02 = float(content[6])
                th03 = float(content[7])
            else:
                th02, th03 = 0.0, 0.0
            self.angle[(aclass1, aclass2, aclass3)] = {
                "kth": kth,
                "th01": th01, "th02": th02, "th03": th03,
                "inPlane": inPlane
            }
        
        def _parse_strbnd_line(line):
            content = line.split()
            aclass1, aclass2, aclass3 = map(int, content[1:4])
            kb1, kb2 = map(float, content[4:])
            self.strbnd[(aclass1, aclass2, aclass3)] = {"kb1": kb1, "kb2": kb2}
        
        def _parse_ureybrad_line(line):
            content = line.split()
            aclass1, aclass2, aclass3 = map(int, content[1:4])
            r0, fc = map(float, content[4:])
            self.ureybrad[(aclass1, aclass2, aclass3)] = {
                "r0": r0, "fc": fc
            }
        
        def _parse_opbend_line(line):
            content = line.split()
            atoms = tuple(map(int, content[1:5]))
            fc = float(content[5])
            self.opbend[atoms] = {'fc': fc}
        
        def _parse_torsion_line(line):
            content = line.split()
            atoms = tuple(map(int, content[1:5]))
            fc1, fc2, fc3 = map(float, content[5::3])
            phi01, phi02, phi03 = map(float, content[6::3])
            mult1, mult2, mult3 = map(int, content[7::3])
            self.torsion[atoms] = {
                "fc1": fc1, "fc2": fc2, "fc3": fc3,
                "phi01": phi01, "phi02": phi02, "phi03": phi03
            }
        
        def _parse_pitors_line(line):
            content = line.split()
            atoms = tuple(map(int, content[1:3]))
            fc = float(content[3])
            self.pitors[atoms] = {"fc": fc}
        
        def _parse_strtors_line(line):
            content = line.split()
            atoms = tuple(map(int, content[1:5]))
            k11, k12, k13, k21, k22, k23, k31, k32, k33 = map(float, content[5:])
            self.strtors[atoms] = {
                "k11": k11, "k12": k12, "k13": k13,
                "k21": k21, "k22": k22, "k23": k23,
                "k31": k31, "k32": k32, "k33": k33
            }
        
        def _parse_angtors_line(line):
            content = line.split()
            atoms = tuple(map(int, content[1:5]))
            k11, k12, k13, k21, k22, k23 = map(float, content[5:])
            self.angtors[atoms] = {
                "k11": k11, "k12": k12, "k13": k13,
                "k21": k21, "k22": k22, "k23": k23
            }
        
        def _parse_tortors_line(line):
            content = line.split()
            atoms = tuple(map(int, content[1:6]))
            ngrid1, ngrid2 = map(int, content[6:])
            data = [] 
            for _ in range(ngrid1 * ngrid2):
                phi, psi, ene = map(float, f.readline().split())
                data.append((phi, psi, ene))
            self.tortors[atoms] = data
        
        def _parse_multipole_line(line):
            content = line.split()
            atype = int(content[1])
            kz = int(content[2])
            kx = int(content[3])
            ky = int(content[4]) if len(content) > 5 else 0
            c0 = float(content[-1])
            dx, dy, dz = map(float, f.readline().strip().split())
            qxx = float(f.readline().strip())
            qxy, qyy = map(float, f.readline().strip().split())
            qxz, qyz, qzz = map(float, f.readline().strip().split())
            self.multipole[atype] = {
                "kz": kz, "kx": kx, "ky": ky,
                "c0": c0, 
                "dx": dx, "dy": dy, "dz": dz,
                "qxx": qxx, "qxy": qxy, "qyy": qyy, "qxz": qxz, "qyz": qyz, "qzz": qzz
            }
        
        def _parse_polarize_line(line):
            content = line.split()
            atype = int(content[1])
            alpha, thole = map(float, content[2:4])
            if len(content) > 4:
                group = list(map(int, content[4:]))
            else:
                group = []
            self.polarize[atype] = {"alpha": alpha, "thole": thole, "group": group}


        with open(fname) as f:
            for line in f:
                line = line.strip()
                if line.startswith("forcefield"):
                    mode = 1
                elif (mode == 1) and ("#" in line):
                    mode = 0
                elif line.startswith("##  Atom Type Definitions  ##"):
                    mode = 2
                elif (mode == 2) and (line.startswith("atom")):
                    mode = 3
                elif (mode == 3) and (not line):
                    mode = 0
                elif line.startswith("##  Van der Waals Parameters  ##"):
                    mode = 4
                elif (mode == 4) and (line.startswith("vdw")):
                    mode = 5
                elif (mode == 5) and (not line):
                    mode = 0
                elif line.startswith("##  Van der Waals Pair Parameters  ##"):
                    mode = 6
                elif (mode == 6) and (line.startswith("vdwpair")):
                    mode = 7
                elif (mode == 7) and (not line):
                    mode = 0
                elif line.startswith("##  Bond Stretching Parameters  ##"):
                    mode = 8
                elif (mode == 8) and (line.startswith("bond")):
                    mode = 9
                elif (mode == 9) and (not line):
                    mode = 0
                elif line.startswith("##  Angle Bending Parameters  ##"):
                    mode = 10
                elif (mode == 10) and (line.startswith("angle")):
                    mode = 11
                elif (mode == 11) and (not line):
                    mode = 0
                elif line.startswith("##  Stretch-Bend Parameters  ##"):
                    mode = 12
                elif (mode == 12) and (line.startswith("strbnd")):
                    mode = 13
                elif (mode == 13) and (not line):
                    mode = 0
                elif line.startswith("##  Urey-Bradley Parameters  ##"):
                    mode = 14
                elif (mode == 14) and (line.startswith("ureybrad")):
                    mode = 15
                elif (mode == 15) and (not line):
                    mode = 0
                elif line.startswith("##  Out-of-Plane Bend Parameters  ##"):
                    mode = 16
                elif (mode == 16) and (line.startswith("opbend")):
                    mode = 17
                elif (mode == 17) and (not line):
                    mode = 0
                elif line.startswith("##  Torsional Parameters  ##"):
                    mode = 18
                elif (mode == 18) and (line.startswith("torsion")):
                    mode = 19
                elif (mode == 19) and (not line):
                    mode = 0
                elif line.startswith("##  Pi-Torsion Parameters  ##"):
                    mode = 20
                elif (mode == 20) and (line.startswith("pitors")):
                    mode = 21
                elif (mode == 21) and (not line):
                    mode = 0
                elif line.startswith("##  Stretch-Torsion Parameters  ##"):
                    mode = 22
                elif (mode == 22) and line.startswith("strtors"):
                    mode = 23
                elif (mode == 23) and (not line):
                    mode = 0
                elif line.startswith("##  Angle-Torsion Parameters  ##"):
                    mode = 24
                elif (mode == 24) and (line.startswith("angtors")):
                    mode = 25
                elif (mode == 25) and (not line):
                    mode = 0
                elif line.startswith("##  Torsion-Torsion Parameters  ##"):
                    mode = 26
                elif (mode == 26) and (line.startswith("tortors")):
                    mode = 27
                elif (mode == 27) and (not line):
                    mode = 26
                elif (mode == 27) and (line.startswith("#")):
                    mode = 0
                elif line.startswith("##  Atomic Multipole Parameters  ##"):
                    mode = 28
                elif (mode == 28) and (line.startswith("multipole")):
                    mode = 29
                elif (mode == 29) and (not line):
                    mode = 0
                elif line.startswith("##  Dipole Polarizability Parameters  ##"):
                    mode = 30
                elif (mode == 30) and (line.startswith("polarize")):
                    mode = 31
                elif (mode == 31) and (not line):
                    mode = 0
                
                if mode == 1:
                    _parse_meta_line(line)
                elif mode == 3:
                    _parse_atom_type_line(line)
                elif mode == 5:
                    _parse_vdw_line(line)
                elif mode == 7:
                    _parse_vdwpair_line(line)
                elif mode == 9:
                    _parse_bond_line(line)
                elif mode == 11:
                    _parse_angle_line(line)
                elif mode == 13:
                    _parse_strbnd_line(line)
                elif mode == 15:
                    _parse_ureybrad_line(line)
                elif mode == 17:
                    _parse_opbend_line(line)
                elif mode == 19:
                    _parse_torsion_line(line)
                elif mode == 21:
                    _parse_pitors_line(line)
                elif mode == 23:
                    _parse_strtors_line(line)
                elif mode == 25:
                    _parse_angtors_line(line)
                elif mode == 27:
                    _parse_tortors_line(line)
                elif mode == 29:
                    _parse_multipole_line(line)
                elif mode == 31:
                    _parse_polarize_line(line)


                

                


if __name__ == "__main__":
    ff = AmoebaForceField("tests/amoebabio18.prm")
    # print(ff.angle)
    print(ff.polarize)
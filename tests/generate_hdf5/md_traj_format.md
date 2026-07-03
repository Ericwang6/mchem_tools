# M-Chem trajectory HDF5 format (QArchive)

This folder contains a **small but representative** example of the HDF5 files M-Chem writes during MD. The layout follows the **QArchive** schema (`libarchive` on HDF5 via `libstore`). Authoritative spec: `libarchive/docs/qarchive.schema.json`.

## Example files in this directory

| File | Size | Description |
|------|------|-------------|
| `md_traj_1.h5` | ~1.9 MB | Trajectory from a 10-step NVT run |
| `topology.h5` | ~116 KB | Static topology written once at startup |
| `aladi_water_box_md.in` | — | Input that produced these files |

**System:** alanine dipeptide in a water box (`aladi_water_box.db`), AMOEBA09, 30 Å cubic box, 2602 atoms.

**Run settings:** `nsteps = 10`, `dt = 0.001` ps, `trajectory_write_frq = 1` (one frame per step).

---

## Container layout

Both files share the same QArchive conventions:

- Paths look like `job/{job_id}/molecular_dynamics/...`
- Job and frame indices start at **1**, not 0
- Hidden group `.counters` stores append/resume counters
- Units are defined in the schema JSON; they are **not** stored as HDF5 attributes on each dataset

### `md_traj_1.h5` tree (actual `h5ls -r` output)

```
/
├── .counters/
│   ├── job                                    → 1
│   └── job\1\molecular_dynamics\time_step     → 10   (last frame index)
└── job/
    └── 1/
        └── molecular_dynamics/
            ├── natoms                         → 2602
            └── time_step/
                ├── 1/ … 10/                 (one group per saved frame)
                │   ├── time                   scalar, float64
                │   ├── coordinates            [3, 2602], float64
                │   ├── velocity               [3, 2602], float64
                │   ├── acceleration           [3, 2602], float64
                │   └── box                    [3], float64
```

### `topology.h5` tree

```
/
├── .counters/job                              → 1
└── job/1/molecular_dynamics/topology/
    ├── natoms                                 → 2602
    ├── atype                                  [2602], uint64
    ├── atsym                                  [2602], int8 (one char per atom)
    └── conn                                   [5, 2602], uint64
```

Row 0 of `conn` = number of bonds; rows 1–4 = bonded atom indices (0-based).

---

## Per-frame datasets (`time_step/{frame}/`)

Written in `libmchem/libmchem/mchem.C` when `istep % trajectory_write_frq == 0`.

| Key | HDF5 dtype | Shape | Unit | Notes |
|-----|------------|-------|------|-------|
| `time` | `float64` | scalar | **ps** | Simulation time at start of step |
| `coordinates` | `float64` | **[3, natoms]** | **Å** | Row = X/Y/Z, col = atom index |
| `velocity` | `float64` | **[3, natoms]** | **Å/ps** | Same layout as coordinates |
| `acceleration` | `float64` | **[3, natoms]** | **Å/ps²** | Same layout as coordinates |
| `box` | `float64` | **[3]** | **Å** | Orthorhombic lengths `[Lx, Ly, Lz]` |

**Not written by M-Chem (but defined in schema):** `temperature` (K), `pressure` (atm).

**Shape note:** The schema JSON lists coordinates as `[natoms, 3]`, but Armadillo stores **3 × natoms** and HDF5 mirrors that as `[3, natoms]`. Index as `coord(component, atom)` with component 0=X, 1=Y, 2=Z.

---

## Measured values from this example

### Frame times (`md_traj_1.h5`)

| Frame | `time` (ps) |
|-------|-------------|
| 1 | 0.000 |
| 2 | 0.001 |
| 3 | 0.002 |
| 4 | 0.003 |
| 5 | 0.004 |
| 6 | 0.005 |
| 7 | 0.006 |
| 8 | 0.007 |
| 9 | 0.008 |
| 10 | 0.009 |

### Frame 1 samples

**`box`** (Å):

```
[30.0, 30.0, 30.0]
```

**`coordinates`** — first three atoms, columns 0–2 (Å):

```
        atom0      atom1      atom2
X:     36.4881    35.1652    34.7205
Y:     13.2460    12.5645    11.7687
Z:    -22.4775   -22.2151   -23.0470
```

**`topology/atsym`** — first 8 atoms as ASCII:

```
C C O H H H N C   (67, 67, 79, 72, 72, 72, 78, 67)
```

---

## Physical units (MD engine)

| Quantity | Unit | Source |
|----------|------|--------|
| Coordinates, box | Å | Input box / `mchem_params::boxs` |
| Time | ps | `dt` in `.in` file |
| Velocity | Å/ps | Maxwell–Boltzmann init |
| Acceleration | Å/ps² | `a = F/m × 418.4` (`kcal_mole_to_g_A2_ps2` in `libclmd/util/units.h`) |

---

## Restart / naming

- First trajectory file: `md_traj_1.h5`
- On restart (`restart = true`, `last_frame = md_traj_1.h5`): next file is `md_traj_2.h5`, etc.
- Frames are read back via `read_coord_from_hdf5()` (coordinates, box, optional velocity/acceleration)

---

## Inspecting these files

```bash
module load gcc-native cray-hdf5
h5ls -r generate_hdf5/md_traj_1.h5
h5dump -d /job/1/molecular_dynamics/natoms generate_hdf5/md_traj_1.h5
h5dump -d /job/1/molecular_dynamics/time_step/1/box generate_hdf5/md_traj_1.h5
```

Python (requires `h5py`):

```python
import h5py
f = h5py.File("generate_hdf5/md_traj_1.h5", "r")
natoms = f["job/1/molecular_dynamics/natoms"][()]
coords = f["job/1/molecular_dynamics/time_step/1/coordinates"][:]  # shape (3, natoms)
time_ps = f["job/1/molecular_dynamics/time_step/1/time"][()]
```

---

## Related code and docs

- Writer: `libmchem/libmchem/mchem.C`
- Topology writer: `libmchem/libmchem/save_topology.C`
- Reader: `libmchem/libmchem/read_coord_from_hdf5.C`
- Schema: `libarchive/docs/qarchive.schema.json` (`molecular_dynamics` / `time_step` sections)
- Test writer: `libclmd/tests/archive_test.C`

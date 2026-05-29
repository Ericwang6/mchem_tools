M-Chem database (``.db``) format
========================================

An **M-Chem** ``.db`` file is a standard **SQLite 3** database that stores a
parameterized molecular system: global force-field metadata, per-particle
state, and bonded / non-bonded force terms. **M-Chem** reads these files as the
primary input for MD simulations.

Conceptually, a ``.db`` is the **output of force-field parameterization** for one
specific structure, rather than the type-based **force field definition**. It holds
system-assigned atom types, enumerated bonded and non-bonded terms, coordinates,
and simulation metadata for that instance. This is analogous to a GROMACS
``.tpr`` file, an OpenMM ``System`` object, or the AMBER ``prmtop`` + ``inpcrd``
pair combined: each encodes a ready-to-run parameterized system.

Overview
--------

Every mchem database contains:

1. A **`meta`** table — one row of system-wide force-field settings (AMOEBA
   polynomial coefficients, VdW combining rules, etc.).
2. A **`class`** table — maps each **term table name** to the Python dataclass
   used to deserialize rows.
3. One or more **term tables** — each row is one force term; column names match
   dataclass field names in :mod:`mchem.terms`.
4. Optional **`box`** table (class ``Box``) holds periodic cell lengths and angles
   from PDB ``CRYST1`` when a box was present at conversion time.

.. .. note::
..    For developers, table and column layouts are derived from dataclasses at save time. If you
..    add a new term type in code, register it with :func:`mchem.system.register_term_class`
..    before loading a DB that contains that class name.

Example file
------------

``tests/data/DHFR.db`` is a full AMOEBA-parameterized dihydrofolate reductase
system produced from ``tests/data/DHFR.pdb``. It illustrates a production-sized
database:

.. list-table:: Selected tables in ``DHFR.db``
   :header-rows: 1
   :widths: 50 50

   * - Table
     - Rows
   * - ``Particle``
     - 23,558
   * - ``AmoebaBond``
     - 16,569
   * - ``Multipole``
     - 23,558
   * - ``AmoebaTorsionTorsionGrid``
     - 625
   * - ``box``
     - 1

Inspect it with any SQLite client:

.. code-block:: bash

   sqlite3 tests/data/DHFR.db ".schema"
   sqlite3 tests/data/DHFR.db "SELECT * FROM meta;"
   sqlite3 tests/data/DHFR.db "SELECT tablename, clsname FROM class;"

Programmatic access:

.. code-block:: python

   from mchem.system import System

   system = System("tests/data/DHFR.db")
   print(system.meta["name"])           # DHFR
   print(len(system["Particle"]))       # 23558
   print(system["box"][0].a)          # 61.645 (Angstroms)

Required system tables
----------------------

``meta``
~~~~~~~~

Exactly **one row**. Columns are created dynamically from keys passed to
:meth:`mchem.system.System.addMeta` when the system is built (typically from
force-field XML via :meth:`mchem.forcefield.base.ForceField.createSystem`).

For ``DHFR.db``, columns include:

.. list-table:: ``meta`` columns in ``DHFR.db``
   :header-rows: 1
   :widths: 22 78

   * - Column
     - Meaning (typical)
   * - ``name``
     - System / topology name (e.g. ``DHFR``)
   * - ``bondCubic``
     - AMOEBA bond cubic coefficient
   * - ``bondQuartic``
     - AMOEBA bond quartic coefficient
   * - ``angleCubic``
     - AMOEBA angle cubic coefficient
   * - ``angleQuartic``
     - AMOEBA angle quartic coefficient
   * - ``anglePentic``
     - AMOEBA angle quintic coefficient
   * - ``angleSextic``
     - AMOEBA angle sextic coefficient
   * - ``opbendType``
     - Out-of-plane bend type (e.g. ``ALLINGER``)
   * - ``opbendCubic``
     - Out-of-plane bend cubic coefficient
   * - ``opbendQuartic``
     - Out-of-plane bend quartic coefficient
   * - ``opbendPentic``
     - Out-of-plane bend quintic coefficient
   * - ``opbendSextic``
     - Out-of-plane bend sextic coefficient
   * - ``type``
     - Non-bonded type label (e.g. ``BUFFERED-14-7``)
   * - ``radiusrule``
     - VdW radius combining rule (e.g. ``CUBIC-MEAN``)
   * - ``radiustype``
     - Radius definition (e.g. ``R-MIN``)
   * - ``radiussize``
     - Radius size convention (e.g. ``DIAMETER``)
   * - ``epsilonrule``
     - Epsilon combining rule (e.g. ``HHG``)
   * - ``vdw_13_scale``
     - 1-3 VdW scale factor
   * - ``vdw_14_scale``
     - 1-4 VdW scale factor
   * - ``vdw_15_scale``
     - 1-5 VdW scale factor
   * - ``ubCubic``
     - Urey-Bradley cubic coefficient
   * - ``ubQuartic``
     - Urey-Bradley quartic coefficient

Exact columns depend on which force generators ran for that system.

``class``
~~~~~~~~~

Maps **SQLite table name** → **Python class name** (must exist in
:mod:`mchem.system`’s internal registry or be registered explicitly).

Example from ``DHFR.db``:

.. code-block:: text

   tablename              clsname
   ---------------------  ------------------------
   Particle               Particle
   AmoebaBond               AmoebaBond
   box                      Box
   ...

The **table name** used in SQL may differ from ``clsname`` when terms are stored
under a custom name (e.g. ``CMAPTable1`` mapping to ``CMAPTable``). Always use
the name in ``class.tablename`` when querying.

Term tables
-----------

Each term table corresponds to one :class:`mchem.terms.base.TermList` in memory.
**Column names** equal dataclass field names; **SQLite types** are inferred from
Python types:

+---------------+------------------+
| Python type   | SQLite type      |
+===============+==================+
| ``int``       | ``INTEGER``      |
| ``float``     | ``FLOAT``        |
| ``str``       | ``TEXT``         |
| ``bool``      | ``INTEGER``      |
| ``list``      | ``TEXT`` (space- |
|               | separated ints)  |
+---------------+------------------+

On load, ``list`` fields (e.g. ``IsotropicPolarization.grp``) are split from
space-separated text back into integers.

Particle indices
~~~~~~~~~~~~~~~~

Bonded terms reference particles by integer indices ``p0``, ``p1``, … that
match ``Particle.idx``. Per-particle terms use ``idx``.

Common term types (AMOEBA / ``DHFR.db``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Table / class
     - Role
   * - ``Particle``
     - Atom index, name, element, mass, residue, coordinates, velocities
   * - ``AmoebaBond``
     - Bond stretch (``p0``, ``p1``, ``b0``, ``kb``, cubic, quartic)
   * - ``AmoebaAngle``
     - Valence angle
   * - ``AmoebaAngleInPlane``
     - In-plane valence angle (4 atoms)
   * - ``AmoebaOutOfPlaneBend``
     - Out-of-plane bend
   * - ``PeriodicTorsion``
     - Fourier dihedral (up to 6 terms)
   * - ``AmoebaPiTorsion``
     - Pi-system torsion (6 atoms)
   * - ``AmoebaStretchBend``
     - Stretch–bend coupling
   * - ``AmoebaUreyBradley``
     - Urey–Bradley 1–3 distance term
   * - ``AmoebaTorsionTorsion``
     - Torsion–torsion coupling (links to grid)
   * - ``AmoebaTorsionTorsionGrid``
     - 2D grid points (``angle1``, ``angle2``, ``f``, ``gridIdx``)
   * - ``AmoebaVdw147``
     - Buffered 14-7 VdW per particle
   * - ``Multipole``
     - Charge, dipole, quadrupole, axis frame
   * - ``IsotropicPolarization``
     - Polarizability, Thole, group list
   * - ``Box`` (table ``box``)
     - Cell lengths and angles (Å, degrees)

Field-level definitions live in :mod:`mchem.terms.bonded` and
:mod:`mchem.terms.nonbonded` (also exposed in the :doc:`api/index`).

Other term classes (``HarmonicBond``, ``CMAP``, ``PairList``, MBUCB terms, etc.)
produce additional tables when present in a system; their schemas follow the
same dataclass-column rule.

Sample rows (``DHFR.db``)
-------------------------

**Periodic box** (cubic 61.645 Å, 90° angles):

.. code-block:: text

   a=61.645  b=61.645  c=61.645  alpha=90  beta=90  gamma=90

**First particle** (MET N):

.. code-block:: text

   idx=0  name=N  element=N  mass=14.0067  resnum=1  resname=MET
   xx=35.334  xy=19.608  xz=36.364  vx=vy=vz=0

**First bond** (particles 0–1):

.. code-block:: text

   p0=0  p1=1  b0=0.1015  kb=193258.96  cubic=-25.5  quartic=379.3125  paramIdx=69

Writing and reading
-------------------

**Create** via CLI:

.. code-block:: bash

   mchem-tools convert -i structure.pdb -o system.db -f amoebabio18.xml

**Create / update** in Python:

.. code-block:: python

   from mchem.forcefield import ForceField
   from mchem.fileformats import load_pdb, read_pdb_box
   from mchem.system import System, Box

   top = load_pdb("structure.pdb")
   system = ForceField("amoebabio18.xml").createSystem(top)
   box = read_pdb_box("structure.pdb")
   if box is not None:
       system.addTerm(Box(*box), "box")
   system.save("system.db", overwrite=True)

**Load**:

.. code-block:: python

   system = System("system.db")
   bonds = system["AmoebaBond"]

**Custom term types**: define a dataclass, call
:func:`mchem.system.register_term_class` before ``System(path)``, and ensure
the ``class`` table’s ``clsname`` matches the registered class name.

Implementation notes
--------------------

- On save, mchem uses a transaction, sets ``PRAGMA journal_mode=WAL`` and
  ``PRAGMA synchronous=OFF``, then commits.
- Term rows are sorted before insert by particle columns ``p0``, ``p1``, … (or
  ``idx`` when no ``p*`` fields exist) for stable, diff-friendly ordering.
- Saving with ``overwrite=False`` (default) raises if the output file already
  exists.
- The loader requires ``meta`` and ``class`` tables; unknown ``clsname`` values
  raise unless registered.

See also
--------

- :doc:`example_usage` — solvate and ``convert`` workflow
- :class:`mchem.system.System` — load/save implementation
- :mod:`mchem.terms` — term dataclass definitions

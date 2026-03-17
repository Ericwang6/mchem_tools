Example: Solvate and Parameterize
=================================

This page walks through command-line usage: solvating a solute PDB, then
parameterizing the solvated system and saving it to a SQLite database. The
example uses the provided peptide structure ``examples/ace_ala_nme.pdb``.

Prerequisites
-------------

- The CLI is installed as **mchem-tools** (see the project's installation
  instructions). Run ``mchem-tools --help`` to see available commands.
- For the **solvate** command, **Packmol** must be installed and on your
  ``PATH``. The **convert** command does not require external binaries.

Input PDB requirements
----------------------

.. warning::
   The input PDB must be **prepared/fixed** before use with mchem. mchem does
   **not** add hydrogens, missing atoms, missing residues, or terminal (capping)
   oxygens.

PDB files are read as-is: :func:`mchem.fileformats.pdb.load_pdb` parses
ATOM/HETATM records and the topology is matched to residue templates via
:meth:`mchem.topology.Topology.matchTemplates`. Incomplete structures will fail
template matching or yield incorrect parameters. Use an external tool (e.g.
OpenMM Modeller, PDB2PQR, or a molecular builder) to add hydrogens, cap
termini, and fill missing atoms or residues before running ``solvate`` or
``convert``.

Example: Solvate then parameterize
----------------------------------

Two-step workflow using the command line.

Step 1 — Solvate
~~~~~~~~~~~~~~~~~

- **Input**: A solute-only PDB (e.g. ``examples/ace_ala_nme.pdb``).
- **Output**: A solvated PDB with water, optional ions, and a CRYST1 periodic
  box record.

.. code-block:: bash

   mchem-tools solvate -i examples/ace_ala_nme.pdb -o solvated.pdb

Optional flags: ``--box-shape`` (cube, dodecahedron, octahedron), ``--buffer``
(minimum padding in Å), ``--no-neutralize``, ``--ionic-strength``,
``--positive-ion``, ``--negative-ion``. Run ``mchem-tools solvate --help`` for
details.

Step 2 — Parameterize (convert) to DB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Input**: The solvated PDB from step 1.
- **Output**: A SQLite database (e.g. ``system.db``) containing force-field
  terms.

.. code-block:: bash

   mchem-tools convert -i solvated.pdb -o system.db -f amoebabio18.xml


What the solvate command does
------------------------------

The ``solvate`` command (implemented in :mod:`mchem.solvate` and invoked from
:mod:`mchem.main`) performs the following:

- **Load solute**: The input PDB is loaded into a :class:`mchem.topology.Topology`;
  atom positions are taken from the topology.

- **Charge**: The net charge is computed by temporarily building a system with
  ``ForceField("amoebabio18_solvate.xml")`` and summing multipole monopoles.
  This value determines how many counterions to add when neutralization is
  requested.

- **Box**: A periodic box shape is chosen (cube, dodecahedron, or octahedron).
  The box size is determined from the solute’s bounding sphere and the
  ``--buffer`` (minimum padding between solute and box edge). Box vectors are
  computed in reduced form (e.g. orthorhombic or triclinic).

- **Solute placement**: The solute is shifted so its center lies at the box
  center in the orthorhombic frame used for packing.

- **Water and ions**: The number of water molecules is estimated from
  (box volume − approximate solute volume) × bulk water number density. Ion
  counts come from neutralization and, optionally, the requested ionic strength.
  Supported ion types are defined in :mod:`mchem.solvate` (e.g. Na+, K+, Cl-).

- **Packmol**: The external `Packmol <https://github.com/leandromartinez98/packmol>`_
  program is run to place one fixed solute, N waters, and ions inside the box
  (default tolerance 2.0 Å).

- **Output**: The combined solvated structure is read from Packmol’s output
  PDB; the CLI writes it to your output path with
  :func:`mchem.fileformats.pdb.write_pdb`, including box vectors as a CRYST1
  record.

Packmol must be installed and on your ``PATH`` for ``solvate`` to succeed.

What the convert command does
-----------------------------

The ``convert`` command (implemented in :mod:`mchem.main`) does the following:

- **Load topology**: :func:`mchem.fileformats.pdb.load_pdb` reads the PDB,
  builds a :class:`mchem.topology.Topology` with residues and atoms, and runs
  :meth:`mchem.topology.Topology.matchTemplates` to assign residue templates
  (bonds and standard atom names).

- **Box (optional)**: :func:`mchem.fileformats.pdb.read_pdb_box` reads the
  CRYST1 record from the PDB if present. If present, a :class:`mchem.system.Box`
  term is added to the system so the DB stores periodic boundary conditions.

- **Parameterize**: :meth:`mchem.forcefield.base.ForceField.createSystem` (i)
  assigns atom types from the force field XML, (ii) builds a
  :class:`mchem.system.System` with a :class:`mchem.terms.nonbonded.Particle`
  list (one per atom: index, name, symbol, mass, residue id, residue name,
  x, y, z), (iii) runs each force generator to create terms (bonds, angles,
  torsions, multipoles, vdW, polarization, etc.).

- **Save**: :meth:`mchem.system.System.save` writes the system to a SQLite
  file: a metadata table, a class map, and one table per term type (e.g.
  Particle, AmoebaBond, Multipole).

No external binaries are required for ``convert``; only the chosen force field
XML (and its referenced templates) must be available.

See also
--------

- :doc:`api/index` — API reference for :mod:`mchem.main`, :mod:`mchem.solvate`,
  :mod:`mchem.system`, and related modules.

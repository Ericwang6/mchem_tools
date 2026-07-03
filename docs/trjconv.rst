Trajectory conversion (``trjconv``)
===================================

The ``trjconv`` command extracts a frame from an **M-Chem** MD trajectory and
writes it as a structure file. It is a small analogue of ``gmx trjconv``: give
it a topology and a trajectory, and it produces coordinates in a common format.

The current version supports:

- **Topology input:** SQLite ``.db`` (see :doc:`db_format`)
- **Trajectory input:** M-Chem HDF5 (QArchive) ``.h5`` / ``.hdf5``
- **Output:** single-frame PDB ``.pdb``

The implementation (:mod:`mchem.trjconv`) is organized around three extensible
abstractions -- :class:`mchem.trjconv.TopologyProvider`,
:class:`mchem.trjconv.TrajectoryReader`, and
:class:`mchem.trjconv.TrajectoryWriter` -- so additional topology formats
(e.g. PDB), trajectory formats, and output formats (e.g. XTC) and multi-frame
output can be added without changing the :func:`mchem.trjconv.trjconv`
orchestrator.

Command-line usage
------------------

.. code-block:: bash

   mchem-tools trjconv -p system.db -i md_traj_1.h5 -o frame.pdb -f 10

.. list-table:: ``trjconv`` options
   :header-rows: 1
   :widths: 22 78

   * - Option
     - Meaning
   * - ``-p`` / ``--topology``
     - Topology file (currently ``.db``). Provides atom names, elements, and
       residue information.
   * - ``-i`` / ``--input``
     - Input M-Chem HDF5 trajectory (``.h5``).
   * - ``-o`` / ``--output``
     - Output structure file (currently ``.pdb``).
   * - ``-f`` / ``--frame``
     - Frame to extract. Uses the trajectory's native **1-based** index; ``-1``
       (the default) selects the **last** frame.

.. note::

   Frames use the trajectory's native **1-based** numbering, matching the HDF5
   ``time_step`` groups. Passing ``-1`` (or omitting ``-f``) selects the last
   frame.

Python API
----------

.. code-block:: python

   from mchem.trjconv import trjconv

   # Write the last frame (default)
   frame = trjconv("system.db", "md_traj_1.h5", "last.pdb")
   print(frame.index, frame.time, frame.box)

   # Write a specific frame
   trjconv("system.db", "md_traj_1.h5", "frame1.pdb", frame=1)

Lower-level building blocks can be used directly, for example to iterate frames:

.. code-block:: python

   from mchem.trjconv import DBTopologyProvider, HDF5TrajectoryReader

   provider = DBTopologyProvider("system.db")
   with HDF5TrajectoryReader("md_traj_1.h5") as reader:
       print(reader.n_frames, "frames")
       for frame in reader:
           coords = frame.coordinates      # (natoms, 3), Angstrom
           box = frame.box                 # (a, b, c), Angstrom

M-Chem HDF5 trajectory format
----------------------------

M-Chem writes trajectories as HDF5 files following the **QArchive** schema.
The reader (:class:`mchem.trjconv.HDF5TrajectoryReader`) expects the following
layout:

.. code-block:: text

   /
   |- .counters/                              (append/resume bookkeeping)
   \- job/
      \- {job_id}/
         \- molecular_dynamics/
            |- natoms                         scalar, number of atoms
            \- time_step/
               |- 1/ ... N/                   one group per saved frame
               |  |- time                     scalar, float64   (ps)
               |  |- coordinates              [3, natoms], float64 (Angstrom)
               |  |- velocity                 [3, natoms], float64 (Angstrom/ps)
               |  |- acceleration             [3, natoms], float64 (Angstrom/ps^2)
               |  \- box                       [3], float64        (Angstrom)

Key conventions:

- **1-based indexing.** Both job and frame (``time_step``) groups start at 1.
- **Column-major coordinates.** ``coordinates`` is stored as ``[3, natoms]``
  (row = X/Y/Z component, column = atom). The reader transposes this to
  ``(natoms, 3)``.
- **Orthorhombic box.** ``box`` holds cell lengths ``[Lx, Ly, Lz]`` in
  Angstroms; PDB output writes these as a ``CRYST1`` record with 90 degree
  angles.
- **Units** are defined by the schema rather than stored per dataset:
  coordinates and box in Angstroms, time in picoseconds.

.. list-table:: Per-frame datasets (``time_step/{frame}/``)
   :header-rows: 1
   :widths: 20 20 20 40

   * - Key
     - dtype
     - Shape
     - Notes
   * - ``time``
     - float64
     - scalar
     - Simulation time (ps)
   * - ``coordinates``
     - float64
     - ``[3, natoms]``
     - X/Y/Z by atom (Angstrom)
   * - ``velocity``
     - float64
     - ``[3, natoms]``
     - Angstrom/ps
   * - ``acceleration``
     - float64
     - ``[3, natoms]``
     - Angstrom/ps^2
   * - ``box``
     - float64
     - ``[3]``
     - Orthorhombic lengths (Angstrom)

See also
--------

- :doc:`db_format` -- the ``.db`` topology used as ``-p`` input
- :mod:`mchem.trjconv` -- API reference for the conversion classes

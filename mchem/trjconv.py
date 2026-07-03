"""
Trajectory conversion utilities (an M-Chem analogue of ``gmx trjconv``).

This module reads a topology (currently the SQLite ``.db`` format) together with
an M-Chem HDF5 (QArchive) trajectory and writes selected frames to a molecular
structure/trajectory file (currently PDB).

The design is intentionally modular so that additional topology formats
(e.g. PDB), trajectory formats, and output formats (e.g. XTC) can be added by
implementing the relevant abstract base class and registering it, without
changing the :func:`trjconv` orchestrator.

Frame indexing
--------------
Frames use the trajectory's native **1-based** numbering (matching the HDF5
``time_step`` groups). A frame index of ``-1`` (or ``None``) selects the last
frame.
"""

import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type

import numpy as np

from .fileformats.pdb import format_atom_record, format_cryst1_record


@dataclass
class AtomInfo:
    """
    Minimal per-atom metadata required to write a structure record.

    Parameters
    ----------
    idx : int
        Zero-based atom index (matches the trajectory column order).
    name : str
        Atom name (e.g. ``"CA"``).
    element : str
        Element symbol.
    resnum : int
        Residue sequence number.
    resname : str
        Residue name.
    chain : str
        Chain identifier (single character).
    """

    idx: int
    name: str
    element: str
    resnum: int
    resname: str
    chain: str = "A"


@dataclass
class Frame:
    """
    A single trajectory frame.

    Parameters
    ----------
    index : int
        Native (1-based) frame index within the trajectory.
    coordinates : numpy.ndarray
        Cartesian coordinates in Angstroms, shape ``(natoms, 3)``.
    box : tuple of float, optional
        Orthorhombic cell lengths ``(a, b, c)`` in Angstroms, or None.
    time : float, optional
        Simulation time in picoseconds.
    """

    index: int
    coordinates: np.ndarray
    box: Optional[Tuple[float, float, float]] = None
    time: Optional[float] = None


# --------------------------------------------------------------------------- #
# Topology providers
# --------------------------------------------------------------------------- #


class TopologyProvider(ABC):
    """
    Abstract source of per-atom metadata for writing structure records.

    Subclasses expose an ordered list of :class:`AtomInfo` whose order matches
    the atom/column order of the trajectory being converted.
    """

    @property
    @abstractmethod
    def atoms(self) -> List[AtomInfo]:
        """Return the ordered list of :class:`AtomInfo`."""

    @property
    def natoms(self) -> int:
        """Number of atoms in the topology."""
        return len(self.atoms)


class DBTopologyProvider(TopologyProvider):
    """
    Topology provider backed by an M-Chem SQLite ``.db`` file.

    Reads the ``Particle`` term table directly (see :doc:`db_format`), which
    keeps the provider lightweight and robust to auxiliary tables that are not
    needed to write structure records.

    Parameters
    ----------
    path : os.PathLike
        Path to the ``.db`` topology file.
    """

    def __init__(self, path: os.PathLike):
        self._path = str(path)
        self._atoms = self._build_atoms()

    def _build_atoms(self) -> List[AtomInfo]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            names = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            if "Particle" not in names:
                raise ValueError(
                    f"Topology '{self._path}' has no 'Particle' table; "
                    "cannot build atoms."
                )
            rows = conn.execute(
                "SELECT idx, name, element, resnum, resname FROM Particle"
            ).fetchall()
        finally:
            conn.close()

        atoms = [
            AtomInfo(
                idx=r["idx"],
                name=r["name"],
                element=r["element"],
                resnum=r["resnum"],
                resname=r["resname"],
            )
            for r in rows
        ]
        atoms.sort(key=lambda a: a.idx)
        return atoms

    @property
    def atoms(self) -> List[AtomInfo]:
        return self._atoms


# --------------------------------------------------------------------------- #
# Trajectory readers
# --------------------------------------------------------------------------- #


class TrajectoryReader(ABC):
    """
    Abstract random-access reader over trajectory frames.

    Frames are addressed by native 1-based index. Iterating a reader yields
    :class:`Frame` objects in ascending frame order.
    """

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """Total number of frames available."""

    @property
    @abstractmethod
    def frame_indices(self) -> List[int]:
        """Sorted list of available (1-based) frame indices."""

    @abstractmethod
    def read_frame(self, index: int) -> Frame:
        """
        Read one frame.

        Parameters
        ----------
        index : int
            Native 1-based frame index. ``-1`` selects the last frame.

        Returns
        -------
        Frame
        """

    def resolve_index(self, index: Optional[int]) -> int:
        """Resolve ``None``/``-1`` to the last frame; validate other values."""
        indices = self.frame_indices
        if not indices:
            raise ValueError("Trajectory contains no frames.")
        if index is None or index == -1:
            return indices[-1]
        if index not in indices:
            raise IndexError(
                f"Frame {index} not found. Available frames: "
                f"{indices[0]}..{indices[-1]} ({self.n_frames} total)."
            )
        return index

    def __iter__(self) -> Iterator[Frame]:
        for i in self.frame_indices:
            yield self.read_frame(i)

    def close(self) -> None:
        """Release any underlying resources (no-op by default)."""

    def __enter__(self) -> "TrajectoryReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class HDF5TrajectoryReader(TrajectoryReader):
    """
    Reader for M-Chem HDF5 (QArchive) trajectories.

    The expected layout is::

        job/{job_id}/molecular_dynamics/
            natoms
            time_step/{frame}/
                coordinates   [3, natoms]  (Angstrom)
                box            [3]          (Angstrom)
                time           scalar       (ps)

    Coordinates are stored column-major as ``(3, natoms)`` and are transposed
    to ``(natoms, 3)`` on read.

    Parameters
    ----------
    path : os.PathLike
        Path to the ``.h5`` trajectory file.
    job_id : int, optional
        Job group to read. If None, the single available job is auto-detected.
    """

    def __init__(self, path: os.PathLike, job_id: Optional[int] = None):
        try:
            import h5py  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "Reading M-Chem HDF5 trajectories requires 'h5py'. "
                "Install it with 'pip install h5py'."
            ) from exc

        self._h5py = h5py
        self._path = str(path)
        self._file = h5py.File(self._path, "r")
        self._md_group = self._locate_md_group(job_id)
        self._ts_group = self._md_group["time_step"]
        self._frame_indices = sorted(int(k) for k in self._ts_group.keys())
        self._natoms = int(self._md_group["natoms"][()])

    def _locate_md_group(self, job_id: Optional[int]):
        jobs = self._file["job"]
        if job_id is None:
            keys = sorted(int(k) for k in jobs.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"Multiple jobs found ({keys}); specify job_id explicitly."
                )
            job_id = keys[0]
        return jobs[str(job_id)]["molecular_dynamics"]

    @property
    def natoms(self) -> int:
        """Number of atoms per frame as recorded in the trajectory."""
        return self._natoms

    @property
    def n_frames(self) -> int:
        return len(self._frame_indices)

    @property
    def frame_indices(self) -> List[int]:
        return list(self._frame_indices)

    def read_frame(self, index: int) -> Frame:
        index = self.resolve_index(index)
        grp = self._ts_group[str(index)]
        coords = np.asarray(grp["coordinates"][:], dtype=float)
        if coords.shape[0] == 3 and coords.shape[1] != 3:
            coords = coords.T
        elif coords.shape[1] != 3 and coords.shape[0] != 3:
            raise ValueError(
                f"Unexpected coordinate shape {coords.shape} for frame {index}."
            )
        box = None
        if "box" in grp:
            box_arr = np.asarray(grp["box"][:], dtype=float).ravel()
            if box_arr.size >= 3:
                box = (float(box_arr[0]), float(box_arr[1]), float(box_arr[2]))
        time = float(grp["time"][()]) if "time" in grp else None
        return Frame(index=index, coordinates=coords, box=box, time=time)

    def close(self) -> None:
        self._file.close()


# --------------------------------------------------------------------------- #
# Trajectory writers
# --------------------------------------------------------------------------- #


class TrajectoryWriter(ABC):
    """
    Abstract writer for one or more frames to an output file.

    Use as a context manager so resources are flushed and closed. Multiple
    calls to :meth:`write_frame` append successive models/frames.
    """

    @abstractmethod
    def write_frame(self, frame: Frame, atoms: Sequence[AtomInfo]) -> None:
        """Write a single :class:`Frame` given the topology ``atoms``."""

    def close(self) -> None:
        """Finalize and close the output (no-op by default)."""

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class PDBTrajectoryWriter(TrajectoryWriter):
    """
    Write frames as PDB records.

    A single frame is written as a plain ``ATOM``/``HETATM`` block preceded by a
    ``CRYST1`` record (when box information is available). When more than one
    frame is written, each is wrapped in ``MODEL``/``ENDMDL`` records, laying the
    groundwork for future multi-frame output.

    Parameters
    ----------
    path : os.PathLike
        Output PDB path.
    """

    def __init__(self, path: os.PathLike):
        self._path = str(path)
        self._file = open(self._path, "w")
        self._n_written = 0

    def write_frame(self, frame: Frame, atoms: Sequence[AtomInfo]) -> None:
        coords = frame.coordinates
        if coords.shape[0] != len(atoms):
            raise ValueError(
                f"Frame has {coords.shape[0]} atoms but topology has "
                f"{len(atoms)}; cannot write PDB."
            )

        model_number = self._n_written + 1
        if model_number == 2:
            # Retroactively wrap the first (already-written) block as MODEL 1.
            self._rewrite_first_as_model()
        multi_model = model_number >= 2

        if frame.box is not None:
            a, b, c = frame.box
            self._file.write(format_cryst1_record(a, b, c, 90.0, 90.0, 90.0))
        if multi_model:
            self._file.write(f"MODEL     {model_number:4d}\n")

        for atom in atoms:
            x, y, z = coords[atom.idx]
            self._file.write(
                format_atom_record(
                    atom.idx + 1, atom.name, atom.resname, atom.chain,
                    atom.resnum, x, y, z, atom.element,
                )
            )
        if multi_model:
            self._file.write("ENDMDL\n")
        self._n_written = model_number

    def _rewrite_first_as_model(self) -> None:
        """Wrap the already-written single-frame block in MODEL 1/ENDMDL."""
        self._file.close()
        with open(self._path, "r") as f:
            lines = f.readlines()
        out: List[str] = []
        inserted = False
        for line in lines:
            if not inserted and line.startswith(("ATOM", "HETATM")):
                out.append("MODEL        1\n")
                inserted = True
            out.append(line)
        out.append("ENDMDL\n")
        with open(self._path, "w") as f:
            f.writelines(out)
        self._file = open(self._path, "a")

    def close(self) -> None:
        if not self._file.closed:
            self._file.write("END\n")
            self._file.close()


# --------------------------------------------------------------------------- #
# Format registries
# --------------------------------------------------------------------------- #

TOPOLOGY_PROVIDERS: Dict[str, Type[TopologyProvider]] = {
    ".db": DBTopologyProvider,
}

TRAJECTORY_READERS: Dict[str, Type[TrajectoryReader]] = {
    ".h5": HDF5TrajectoryReader,
    ".hdf5": HDF5TrajectoryReader,
}

TRAJECTORY_WRITERS: Dict[str, Type[TrajectoryWriter]] = {
    ".pdb": PDBTrajectoryWriter,
}


def _select(registry: Dict[str, type], path: os.PathLike, kind: str) -> type:
    ext = Path(str(path)).suffix.lower()
    if ext not in registry:
        raise ValueError(
            f"Unsupported {kind} format '{ext}' for '{path}'. "
            f"Supported: {', '.join(sorted(registry))}."
        )
    return registry[ext]


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def trjconv(
    topology_path: os.PathLike,
    trajectory_path: os.PathLike,
    output_path: os.PathLike,
    frame: Optional[int] = None,
) -> Frame:
    """
    Convert an M-Chem trajectory frame to a structure file.

    Picks a topology provider, trajectory reader, and output writer based on
    file extensions, extracts the requested frame, and writes it out.

    Parameters
    ----------
    topology_path : os.PathLike
        Topology file (currently ``.db``).
    trajectory_path : os.PathLike
        M-Chem HDF5 trajectory (``.h5``/``.hdf5``).
    output_path : os.PathLike
        Output structure file (currently ``.pdb``).
    frame : int, optional
        Native 1-based frame index. ``None`` or ``-1`` selects the last frame.

    Returns
    -------
    Frame
        The frame that was written (with resolved index).

    Raises
    ------
    ValueError
        If a format is unsupported or the topology and trajectory atom counts
        do not match.
    IndexError
        If the requested frame does not exist.
    """
    provider_cls = _select(TOPOLOGY_PROVIDERS, topology_path, "topology")
    reader_cls = _select(TRAJECTORY_READERS, trajectory_path, "trajectory")
    writer_cls = _select(TRAJECTORY_WRITERS, output_path, "output")

    provider = provider_cls(topology_path)
    with reader_cls(trajectory_path) as reader:
        traj_natoms = getattr(reader, "natoms", None)
        if traj_natoms is not None and traj_natoms != provider.natoms:
            raise ValueError(
                f"Atom count mismatch: topology has {provider.natoms} atoms "
                f"but trajectory has {traj_natoms}."
            )
        frame_obj = reader.read_frame(reader.resolve_index(frame))
        if frame_obj.coordinates.shape[0] != provider.natoms:
            raise ValueError(
                f"Atom count mismatch: topology has {provider.natoms} atoms "
                f"but frame has {frame_obj.coordinates.shape[0]}."
            )
        with writer_cls(output_path) as writer:
            writer.write_frame(frame_obj, provider.atoms)

    return frame_obj

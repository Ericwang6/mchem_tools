"""System of force terms and particles; load/save from SQLite DB."""

import re
import sqlite3
from typing import Any, Dict, List, Optional
import os
from dataclasses import dataclass, fields, is_dataclass

from .terms import TermList
from .terms.bonded import (
    HarmonicBond, AmoebaBond, HarmonicAngle, AmoebaAngle,
    AmoebaAngleInPlane, AmoebaStretchBend, AmoebaUreyBradley,
    AmoebaOutOfPlaneBend, PeriodicTorsion, AmoebaStretchTorsion,
    AmoebaAngleTorsion, AmoebaPiTorsion, CMAPTable, CMAP,
    AmoebaTorsionTorsion, AmoebaTorsionTorsionGrid,
)
from .terms.nonbonded import (
    Particle, AmoebaVdw147, Multipole,
    IsotropicPolarization, AnisotropicPolarization,
    MBUCBChargePenetration, MBUCBChargeTransfer, PairList,
)


@dataclass
class Box:
    """
    Periodic box dimensions from PDB CRYST1: cell lengths a, b, c (Angstroms)
    and angles alpha, beta, gamma (degrees).
    """
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


_CLASS_REGISTRY: Dict[str, type] = {cls.__name__: cls for cls in [
    HarmonicBond, AmoebaBond, HarmonicAngle, AmoebaAngle,
    AmoebaAngleInPlane, AmoebaStretchBend, AmoebaUreyBradley,
    AmoebaOutOfPlaneBend, PeriodicTorsion, AmoebaStretchTorsion,
    AmoebaAngleTorsion, AmoebaPiTorsion, CMAPTable, CMAP,
    AmoebaTorsionTorsion, AmoebaTorsionTorsionGrid,
    Particle, AmoebaVdw147, Multipole,
    IsotropicPolarization, AnisotropicPolarization,
    MBUCBChargePenetration, MBUCBChargeTransfer, PairList,
    Box,
]}

_SQL_TYPE_MAP = {
    int: "INTEGER", str: "TEXT", float: "FLOAT",
    bool: "INTEGER", list: "TEXT",
}


def register_term_class(cls):
    """Register a custom term dataclass so it can be deserialized from a
    database file.  Call this before loading a ``.db`` that contains the
    class.
    """
    _CLASS_REGISTRY[cls.__name__] = cls


def _term_to_row(term) -> tuple:
    row = []
    for f in fields(term):
        val = getattr(term, f.name)
        if isinstance(val, list):
            val = " ".join(str(x) for x in val)
        row.append(val)
    return tuple(row)


def _sort_key_fields(datacls) -> List[str]:
    """Return the field names to sort a term table by.

    Prefers particle-index columns (p0, p1, p2, ...) in numeric order,
    falls back to ``idx`` if present, otherwise returns an empty list
    (no sorting).
    """
    names = [f.name for f in fields(datacls)]
    p_cols = sorted(
        [n for n in names if re.fullmatch(r"p\d+", n)],
        key=lambda n: int(n[1:]),
    )
    if p_cols:
        return p_cols
    if "idx" in names:
        return ["idx"]
    return []


class System:
    """
    Container for force-field terms and metadata, backed by SQLite.

    Terms are stored by table name (e.g. ``Particle``, ``AmoebaBond``).
    Metadata is stored in a separate ``meta`` table. Use :func:`register_term_class`
    to register custom term dataclasses before loading a DB that contains them.
    """

    def __init__(self, path: Optional[os.PathLike] = None):
        """
        Create an empty system or load from an existing SQLite DB.

        Parameters
        ----------
        path : os.PathLike, optional
            If given, path to SQLite file to load; otherwise an empty system.
        """
        self._data = {}
        self._meta = {}
        if path:
            self.conn = sqlite3.connect(str(path))
            self.fromDatabase()
        else:
            self.conn = None

    def __getitem__(self, key: str):
        """Return the term list for the given table name."""
        return self.data[key]

    @property
    def data(self):
        """Mapping of table name to :class:`TermList` of terms."""
        return self._data

    @property
    def meta(self):
        """Mapping of metadata keys to values (e.g. name, units)."""
        return self._meta

    # ---- read from database ------------------------------------------------

    def fromDatabase(self):
        """Load all tables from the connected SQLite database into :attr:`data` and :attr:`meta`."""
        names = self.getTableNames()
        assert "meta" in names, "Metadata missing"

        prev_row_factory = self.conn.row_factory
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()

        cur.execute("SELECT * FROM meta")
        self._meta = dict(cur.fetchone())
        names.remove("meta")

        cur.execute("SELECT * FROM class")
        clsmap = {}
        for row in cur.fetchall():
            d = dict(row)
            clsmap[d["tablename"]] = d["clsname"]
        names.remove("class")

        for name in names:
            cur.execute(f"SELECT * FROM {name}")
            clsname = clsmap[name]
            if clsname not in _CLASS_REGISTRY:
                raise ValueError(
                    f"Unknown term class '{clsname}'. "
                    "Register it via register_term_class() before loading."
                )
            cls = _CLASS_REGISTRY[clsname]
            terms = TermList(cls)
            for row in cur.fetchall():
                terms.append(cls(**dict(row)))
            self._data[name] = terms

        self.conn.row_factory = prev_row_factory

    def getTableNames(self):
        """Return list of table names in the connected database."""
        cur = self.conn.execute("SELECT name FROM sqlite_master")
        return [res[0] for res in cur.fetchall()]

    # ---- in-memory mutation -------------------------------------------------

    def getTermsByName(self, name: str) -> TermList:
        """Return the :class:`TermList` for the given term table name."""
        return self.data[name]

    def addTerm(self, term, name: Optional[str] = None):
        """Append a single term; use its class name as table name if `name` is not given."""
        name = term.__class__.__name__ if name is None else name
        if name not in self.data:
            self._data[name] = TermList(term.__class__)
        self._data[name].append(term)

    def addTerms(self, terms: TermList, name: Optional[str] = None):
        """Append all terms from a :class:`TermList`."""
        for term in terms:
            self.addTerm(term, name)

    def addMeta(self, key: str, value: Any):
        """Set a metadata key (hyphens in `key` are replaced with underscores)."""
        key = key.replace("-", "_")
        self._meta[key] = value

    # ---- write to database --------------------------------------------------

    def save(self, path: os.PathLike, overwrite: bool = False):
        """
        Write :attr:`meta` and all term tables to a new SQLite file.

        Parameters
        ----------
        path : os.PathLike
            Output DB path.
        overwrite : bool, optional
            If True, replace existing file; otherwise raise if file exists.
        """
        if (not overwrite) and os.path.isfile(path):
            raise FileExistsError(f"{path} already exists")
        elif overwrite and os.path.isfile(path):
            os.remove(path)

        self.conn = sqlite3.connect(str(path), isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=OFF")

        try:
            self.conn.execute("BEGIN")
            self._write_meta()
            clsmap = self._write_terms()
            self._write_class_map(clsmap)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        finally:
            self.close()

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def _write_meta(self):
        cols = {key: type(value) for key, value in self.meta.items()}
        self._create_table("meta", cols)

        keys = list(self.meta.keys())
        placeholders = ", ".join("?" for _ in keys)
        self.conn.execute(
            f"INSERT INTO meta({', '.join(keys)}) VALUES ({placeholders})",
            tuple(self.meta.values()),
        )

    def _write_terms(self) -> Dict[str, str]:
        clsmap = {}
        for name, terms in self.data.items():
            self._create_table_from_class(terms.cls, name)
            self._insert_terms(terms, name)
            clsmap[name] = terms.cls.__name__
        return clsmap

    def _write_class_map(self, clsmap: Dict[str, str]):
        self._create_table("class", {"tablename": str, "clsname": str})
        self.conn.executemany(
            "INSERT INTO class(tablename, clsname) VALUES (?, ?)",
            list(clsmap.items()),
        )

    def _create_table(self, name: str, columns: Dict[str, Any]):
        col_defs = ", ".join(
            f"{col} {_SQL_TYPE_MAP[typ]}" for col, typ in columns.items()
        )
        self.conn.execute(f"CREATE TABLE {name} ({col_defs})")

    def _create_table_from_class(self, datacls, table_name: Optional[str] = None):
        table_name = datacls.__name__ if table_name is None else table_name
        columns = {attr.name: attr.type for attr in fields(datacls)}
        self._create_table(table_name, columns)

    def _insert_terms(self, terms: TermList, table_name: Optional[str] = None):
        if not terms:
            return
        table_name = table_name or terms.cls.__name__
        attrnames = [attr.name for attr in fields(terms.cls)]
        placeholders = ", ".join("?" for _ in attrnames)
        query = f"INSERT INTO {table_name}({', '.join(attrnames)}) VALUES ({placeholders})"

        sort_keys = _sort_key_fields(terms.cls)
        if sort_keys:
            sorted_terms = sorted(
                terms, key=lambda t: tuple(getattr(t, k) for k in sort_keys)
            )
        else:
            sorted_terms = terms

        self.conn.executemany(query, [_term_to_row(t) for t in sorted_terms])

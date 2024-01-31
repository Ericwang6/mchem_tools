import sqlite3
from typing import List, Any, Dict, Tuple, Union, Optional
import os
from dataclasses import fields, is_dataclass

from .terms import TermList


def tuple2str(tup):
    if len(tup) >= 2:
        return str(tup)
    else:
        strtup = str(tup)
        strtup = strtup[:-2] + ")"
        return strtup


class System:
    def __init__(self, path: Optional[os.PathLike] = None):
        self._data = {}
        self._meta = {}
        if path:
            self.conn = sqlite3.connect(str(path))
            self.fromDatabase()
        else:
            self.conn = None
    
    def __getitem__(self, key: str):
        return self.data[key]
    
    @property
    def data(self):
        return self._data
    
    @property
    def meta(self):
        return self._meta
    
    def fromDatabase(self):
        names = self.getTableNames()
        assert "meta" in names, "Metadata missing"
        
        prevRowFactory = self.conn.row_factory
        self.conn.row_factory = sqlite3.Row

        cur = self.conn.cursor()

        # retrieve metadata
        cur.execute("SELECT * FROM meta")
        self._meta = dict(cur.fetchone())
        names.remove('meta')

        # retrive class names
        cur.execute("SELECT * FROM class")
        clsmap = {}
        for row in cur.fetchall():
            d = dict(row)
            clsmap[d['tablename']] = d['clsname']
        names.remove("class")
        
        for name in names:
            cur.execute(f"SELECT * FROM {name}")
            cls = eval(clsmap[name])
            terms = TermList(cls)
            for row in cur.fetchall():
                term = cls(**dict(row))
                terms.append(term)
            self._data[name] = terms
        
        self.conn.row_factory = prevRowFactory
    
    def getTermsByName(self, name: str) -> TermList:
        return self.data[name]
    
    def addTerm(self, term, name: Optional[str] = None):
        name = term.__class__.__name__ if name is None else name
        if name not in self.data:
            self._data[name] = TermList(term.__class__)
            self._data[name].append(term)
        else:
            self._data[name].append(term)
    
    def addTerms(self, terms: TermList, name: Optional[str] = None):
        for term in terms:
            self.addTerm(term, name)
    
    def addMeta(self, key: str, value: Any):
        key = key.replace("-", "_")
        self._meta[key] = value

    def save(self, path: os.PathLike, overwrite: bool = False):
        if (not overwrite) and os.path.isfile(path):
            raise FileExistsError(f"{path} already exists")
        elif overwrite and os.path.isfile(path):
            os.remove(path)

        self.conn = sqlite3.connect(str(path))
        
        # save meta data
        cols = {}
        for key, value in self.meta.items():
            cols[key] = type(value)
        
        self.createTable("meta", cols)
        self.addTermToDatabase(self.meta, "meta")
        
        clsmap = {}

        # save data
        for name, terms in self.data.items():
            self.createTableFromClass(terms.cls, name)
            self.addTermsToDatabase(terms, name)
            clsmap[name] = terms.cls.__name__
        
        # save class name
        self.createTable("class", {"tablename": str, "clsname": str})
        with self.conn:
            cur = self.conn.cursor()
            cur.execute("INSERT INTO class(tablename, clsname) VALUES " + ', '.join(str(item) for item in clsmap.items()))

        self.close()

    def createTable(self, name: str, columns: Dict[str, Any]):
        typeMap = {int: "INTEGER", str: "TEXT", float: "FLOAT", bool: "INTEGER", list: "TEXT"}
        tmp = ", ".join(f"{col} {typeMap[typ]}" for col, typ in columns.items())
        with self.conn:
            cur = self.conn.cursor()
            query = f"CREATE TABLE {name} ({tmp})"
            cur.execute(query)
    
    def createTableFromClass(self, datacls, tableName: Optional[str] = None):
        tableName = datacls.__name__ if tableName is None else tableName
        columns = {attr.name: attr.type for attr in fields(datacls)}
        self.createTable(tableName, columns)
    
    def getTableNames(self):
        with self.conn:
            cur = self.conn.cursor()
            cur.execute("SELECT name FROM sqlite_master")
        names = [res[0] for res in cur.fetchall()]
        return names
    
    def addTermsToDatabase(self, terms: TermList, tableName: Optional[str] = None):
        if tableName is None:
            tableName = terms.cls.__name__
        assert tableName in self.getTableNames(), f"{tableName} class not registered. Run System.createTableFromClass() first."

        attrnames = []
        for attr in fields(terms.cls):
            attrnames.append(attr.name)
        
        data = []
        for term in terms:
            tmp = []
            for name in attrnames:
                attr = getattr(term, name)
                if isinstance(attr, list):
                    attr = " ".join(str(x) for x in attr)
                tmp.append(attr)
            data.append(tuple(tmp))

        query = f"INSERT INTO {tableName}(" + ', '.join(attrnames) + ') VALUES ' + ', '.join(tuple2str(d) for d in data)
        with self.conn:
            cur = self.conn.cursor()
            cur.execute(query)
    
    def addTermToDatabase(self, term, tableName: Optional[str] = None):
        if is_dataclass(term):
            terms = TermList(term.__class__)
            terms.append(term)
            self.addTermsToDatabase(terms, tableName)
        elif isinstance(term, dict):
            assert tableName is not None, 'tableName is None'
            keys, values = [], []
            for k, v in term.items():
                keys.append(k)
                values.append(v)
            query = f"INSERT INTO {tableName}(" + ', '.join(keys) + ') VALUES ' + tuple2str(tuple(values))

            with self.conn:
                cur = self.conn.cursor()
                cur.execute(query)

    def close(self):
        self.conn.close()


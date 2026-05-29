# AGENTS.md — mchem

Guidance for AI agents and contributors working in this repository.

## Project overview

This repo contains the front-end of **M-Chem**, which is a proprietary molecular dynamics package developed by Q-Chem. It features high and scalable performance support for fixed-charge models, polarziable force field (AMOEBA-like) and QM/MM simulations. The main goal of this front end is to help users to setup the simulation system. It takes PDB structures (protein), adds solvents/ions, parametrize the system with XML-format force field definitions, and convert into SQLite databases for downstream M-Chem workflows.

- **Package name:** `mchem` (see `pyproject.toml`)
- **CLI:** `mchem-tools` (`mchem.main:main`) — `convert` (PDB → `.db`), `solvate`, etc.
- **Python:** `>=3.9` (CI runs on 3.11)
- **Published docs:** [GitHub Pages](https://ericwang6.github.io/mchem_tools/) (built from `docs/` on `master`)

Install for development:

```bash
pip install -e ".[dev]"
```

Optional docs extra:

```bash
pip install -e ".[docs]"
```

## Code structure (`mchem/`)

```
mchem/
├── main.py              # CLI (rich-click): convert, solvate, ...
├── system.py            # System: aggregate force terms; load/save SQLite .db
├── topology.py          # Topology, Residue, Atom, Bond — molecular graph from PDB
├── template.py          # Residue templates from XML
├── element.py           # Element data
├── units.py             # Unit helpers
├── solvate.py           # Solvation (box, water placement)
├── amoeba.py            # Amoeba-specific helpers
├── fileformats/
│   └── pdb.py           # read/write PDB, CRYST1 box
├── forcefield/
│   ├── base.py          # ForceField, Generator, XML parsers
│   ├── generators.py    # Per-force generators (bonds, angles, multipoles, ...)
│   ├── amoebabio18.xml  # Default AMOEBA force field definitions
│   └── ...              # Additional XML parameter sets
├── terms/
│   ├── base.py          # TermList
│   ├── bonded.py        # Bonded dataclass terms (AmoebaBond, CMAP, ...)
│   └── nonbonded.py     # Particles, multipoles, VdW, polarization, ...
└── templates/           # Residue/ion/water XML templates (protein, water, ions, ...)
```

**Data flow (typical `convert` path):**

1. `fileformats.load_pdb` → `Topology`
2. `forcefield.ForceField` parses XML and builds `Generator`s
3. `ForceField.createSystem(topology)` → `System` with `TermList`s of dataclass terms
4. Optional `Box` from PDB CRYST1 → `system.addTerm`
5. `System.save` → SQLite `.db`

**Tests and fixtures:** `tests/` mirrors package concerns (`test_system.py`, `test_forcefield.py`, `test_pdb.py`, …). PDB/DB/JSON fixtures live under `tests/data/`. Examples under `examples/`.

## Keeping this file current

**Update `AGENTS.md` in the same change** whenever you alter any of the following. Do not leave the doc stale for a follow-up PR.

| Change type | What to update in `AGENTS.md` |
|-------------|-------------------------------|
| **Code structure** | `mchem/` tree (new/removed modules or packages), data-flow steps, CLI commands or entry points |
| **Tests** | `tests/` layout, fixture locations, how to run tests, coverage expectations, CI workflow behavior |
| **Environment** | Python version, `pyproject.toml` extras/deps, install commands, optional tools (e.g. OpenMM), CI matrix |

Also refresh the **Quick reference** table and **Project overview** bullets when install or run commands change.

Agents and contributors: if you move, rename, or add a top-level package area, change pytest/CI setup, or bump dependencies — edit the relevant sections here before finishing the task.

## Development SOP

### 1. Test-driven development (TDD)

- **Write or update tests before or alongside implementation** in `tests/`.
- Run the full suite from the repo root:

  ```bash
  pytest -q
  ```

- Use `pytest` fixtures (e.g. `tmp_path`) for filesystem tests; reuse `tests/data/` for golden PDB/DB inputs.
- Dev dependencies include `pytest`, `pytest-cov`, and `openmm` (for some integration-style checks). Install with `pip install -e ".[dev]"`.
- CI (`.github/workflows/tests.yml`) runs `pytest -q` on pushes/PRs to `master`.

Prefer small, focused tests that assert behavior (term counts, round-trip save/load, CLI paths) rather than implementation details.

### 2. Docstrings

- **Language:** English only.
- **Style:** Match existing code. Where there is no clear precedent, use **NumPy-style** docstrings with **reST** syntax (Sphinx Napoleon is enabled in `docs/conf.py`).
- Document public modules, classes, and methods that appear in the API or CLI. Skip private helpers (`_prefix`) and trivial one-liners unless behavior is non-obvious.
- Use reST for cross-references (`:class:`, `:func:`, `:meth:`), code literals, math, and notes/warnings.
- For cross-file references in docstrings, use absolute paths (e.g. ``:class:`mchem.system.System` ``).
- Do not duplicate identical docstrings in subclasses; reference the parent briefly.
- Keep docstrings accurate and concise; update them when behavior changes.

### 3. Documentation build (`docs/`)

Sphinx autodoc pulls docstrings from `mchem/`. After API or narrative doc changes:

```bash
cd docs && make html
```

Open `docs/_build/html/index.html` locally.

Useful targets:

- `make clean` — remove old build artifacts before a full rebuild
- `make html` — HTML output in `docs/_build/html/`
- `make help` — list all Sphinx make targets

Layout:

- `docs/conf.py` — Sphinx config (autodoc, napoleon, autosummary)
- `docs/index.rst` — landing toctree
- `docs/example_usage.rst` — usage examples
- `docs/db_format.rst` — SQLite `.db` on-disk format (update when `System.save` / term schema changes)
- `docs/api/` — API reference (`mchem.rst`, etc.)

Docs are deployed via `.github/workflows/docs-gh-pages.yml` to GitHub Pages. **Rebuild and spot-check HTML** when changing public APIs or user-facing examples.

## Quick reference

| Task              | Command                          |
|-------------------|----------------------------------|
| Install (dev)     | `pip install -e ".[dev]"`        |
| Run tests         | `pytest -q`                      |
| CLI help          | `mchem-tools --help`             |
| Build docs        | `cd docs && make html`           |
| View local docs   | open `docs/_build/html/index.html` |

## Conventions for agents

- Minimize diff scope; follow existing naming, dataclass patterns, and import style.
- **Keep `AGENTS.md` in sync** with code structure, test, and environment changes in the same edit (see [Keeping this file current](#keeping-this-file-current)).
- Do not commit secrets (`.env`, credentials).
- Do not create git commits or push unless the user explicitly asks.
- Only add tests that meaningfully cover real behavior; avoid trivial assertions.

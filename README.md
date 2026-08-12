# mchem

**M-Chem front-end:** convert PDB structures to SQLite-DB formatted force-field parameters for downstream workflows.

<p align="center">
  <a href="https://ericwang6.github.io/mchem_tools/">
    <img src="https://img.shields.io/badge/Documentation-GitHub%20Pages-2ea44f?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Browse the documentation on GitHub Pages">
  </a>
</p>

The HTML reference (API, examples, and usage) is published automatically from the `master` branch. After the first successful deploy, open the badge above or go to **https://ericwang6.github.io/mchem_tools/**.

## Installation

`mchem` is not published on PyPI, but it can be installed directly from GitHub with `pip`:

```bash
pip install git+https://github.com/Ericwang6/mchem_tools.git
```

To install a specific branch, tag, or commit, append `@<ref>`:

```bash
pip install git+https://github.com/Ericwang6/mchem_tools.git@master
```

If you have SSH access to the repository, use the SSH form instead:

```bash
pip install git+ssh://git@github.com/Ericwang6/mchem_tools.git
```

Optional extras are available as usual:

```bash
pip install "mchem[dev] @ git+https://github.com/Ericwang6/mchem_tools.git"
```

### Versioning and upgrades

The version is derived from git by [setuptools-scm](https://setuptools-scm.readthedocs.io/),
so every build identifies the commit it came from: a tagged commit yields a
plain release version such as `0.1.0`, and later commits yield
`0.1.1.dev3+g027b942`. This means `pip` can tell two git installs apart:

```bash
pip install --upgrade git+https://github.com/Ericwang6/mchem_tools.git
python -c "import mchem; print(mchem.__version__)"
```

Because the version is computed from git metadata, installing from a source
copy with no `.git` directory falls back to `0.0.0+unknown`.

The install requires Python 3.9+ and a working `git` executable. It provides the
`mchem-tools` command-line entry point and bundles the force-field and residue
template XML files, so no extra data download is needed:

```bash
mchem-tools --help
```

## Local development

Install with test dependencies:

```bash
pip install -e ".[dev]"
```

Run the CLI (see `--help` for options):

```bash
mchem-tools
```

## Building the docs locally

```bash
pip install -e ".[docs]"
cd docs && make html
```

Then open `docs/_build/html/index.html` in your browser.

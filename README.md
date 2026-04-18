# mchem

**M-Chem front-end:** convert PDB structures to SQLite-DB formatted force-field parameters for downstream workflows.

<p align="center">
  <a href="https://ericwang6.github.io/mchem_tools/">
    <img src="https://img.shields.io/badge/Documentation-GitHub%20Pages-2ea44f?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Browse the documentation on GitHub Pages">
  </a>
</p>

The HTML reference (API, examples, and usage) is published automatically from the `master` branch. After the first successful deploy, open the badge above or go to **https://ericwang6.github.io/mchem_tools/**.

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

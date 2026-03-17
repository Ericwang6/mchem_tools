# mchem Documentation

This directory contains the Sphinx documentation for the mchem project.

## Building the Documentation

### Prerequisites

Install the package with the docs extra:

```bash
pip install -e ".[docs]"
```

### Build Commands

To build the HTML documentation:

```bash
cd docs
make html
```

The generated documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser to view it.

### Other Build Options

- `make clean` - Remove all build files
- `make html` - Build HTML documentation
- `make latexpdf` - Build PDF documentation (requires LaTeX)
- `make help` - Show all available build options

### Auto-generating API Documentation

The documentation is automatically generated from docstrings in the source code using Sphinx's autodoc extension. To update the documentation after making changes to the code:

1. Ensure your docstrings are up to date
2. Run `make clean` to remove old build files
3. Run `make html` to rebuild the documentation

## Documentation Structure

- `conf.py` - Sphinx configuration file
- `index.rst` - Main documentation index
- `api/` - API reference documentation
  - `index.rst` - API documentation index
  - `mchem.rst` - mchem package documentation

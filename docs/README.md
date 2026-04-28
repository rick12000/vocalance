# Documentation

This directory contains the Sphinx documentation for the Vocalance voice command assistant.

## Building the Documentation

### Prerequisites

Install the project with documentation dependencies (run from the repository root):

```bash
pip install -e ".[docs]"
```

### Building HTML Documentation

On Windows:

```bash
make.bat html
```

On Linux/macOS:

```bash
make html
```

### Live Rebuild Server

For automatic rebuilding during development:

On Windows:

```bash
make.bat livehtml
```

On Linux/macOS:

```bash
make livehtml
```

The documentation will be available at `http://localhost:8000`.

### Cleaning Build Files

```bash
make.bat clean  # Windows
make clean      # Linux/macOS
```

## Documentation Structure

- `index.rst` - Main documentation index with the toctree
- `setup/installation.rst` - Installation instructions
- `developer/` - Developer guide (one chapter per RST file, read top to bottom)
- `contact.rst` - Project contact links
- `_static/` - Static assets (CSS, JavaScript, images)
- `_templates/` - Custom Sphinx templates
- `conf.py` - Sphinx configuration

## Styling

The documentation uses a custom blue theme on top of `sphinx_rtd_theme`. The
styling is defined in:

- `_static/custom.css` - Main styling
- `_static/layout-manager.js` - Enhanced UX features

## ReadTheDocs Integration

The documentation is configured for ReadTheDocs via `.readthedocs.yaml` in the
project root. RTD installs the project with `pip install --ignore-requires-python -e ".[docs]"`
because the application's own `pyproject.toml` pins Python 3.13.9 exactly while
RTD ships only the 3.13 minor series.

## Contributing

When adding new documentation:

1. Follow the existing RST formatting style.
2. Ensure all title underlines match the title length.
3. Add blank lines after directive blocks (especially mermaid diagrams).
4. Test the build locally before committing.
5. Check that all cross-references resolve correctly. The project sets
   `fail_on_warning: true` on RTD, so any unresolved reference will break the build.

## Notes

- Markdown files use `myst_parser` (this README is one of them, but it is
  excluded from the build via `exclude_patterns` in `conf.py`).
- Mermaid diagrams are supported via `sphinxcontrib.mermaid`.
- Code blocks have automatic copy buttons via `sphinx_copybutton`.
- Sphinx's `autodoc` and `autosummary` are enabled, but the developer guide
  is hand-written rather than auto-generated; docstrings are picked up only
  where pages explicitly use the directives.

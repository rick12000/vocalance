# Documentation

This directory contains the Sphinx documentation for the Vocalance voice command assistant.

## Building the Documentation

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
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

- `index.rst` - Main documentation index
- `api_reference.rst` - API reference documentation
- `user_guide/` - User guides and tutorials
- `developer/` - Developer documentation
- `_static/` - Static assets (CSS, JavaScript, images)
- `_templates/` - Custom Sphinx templates
- `conf.py` - Sphinx configuration

## Styling

The documentation uses a custom blue theme built on top of `sphinx_rtd_theme`. The styling is defined in:

- `_static/custom.css` - Main styling
- `_static/layout-manager.js` - Enhanced UX features

## ReadTheDocs Integration

The documentation is configured for ReadTheDocs via `.readthedocs.yaml` in the project root.

## Contributing

When adding new documentation:

1. Follow the existing RST formatting style
2. Ensure all title underlines match the title length
3. Add blank lines after directive blocks (especially mermaid diagrams)
4. Test the build locally before committing
5. Check that all cross-references resolve correctly

## Notes

- The documentation uses `myst_parser` for Markdown support
- Mermaid diagrams are supported via `sphinxcontrib.mermaid`
- Code blocks have automatic copy buttons via `sphinx_copybutton`
- API documentation is auto-generated from Python docstrings

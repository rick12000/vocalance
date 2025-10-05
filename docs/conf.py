# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# RTD environment detection (optional, for any future customizations)
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
rtd_version = os.environ.get("READTHEDOCS_VERSION", "latest")

# -- Project information -----------------------------------------------------

project = "Iris"
copyright = "2025, Riccardo Doyle"
author = "Riccardo Doyle"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

# MyST parser configuration
myst_enable_extensions = ["colon_fence", "deflist"]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "https://iris.readthedocs.io/",
    "logo_only": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2563eb",
    # Toc options - optimized for better navigation
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]

# Custom logo
html_logo = "../iris/app/assets/logo/iris_logo_full_size.png"

# The root toctree document (updated from deprecated master_doc)
root_doc = "index"

# The name of the Pygments (syntax highlighting) style to use
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing
todo_include_todos = False

# Suppress common warnings
suppress_warnings = ["ref.doc"]

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
}

latex_documents = [
    (root_doc, "iris.tex", "Iris Documentation", author, "manual"),
]

# -- Options for manual page output ------------------------------------------

man_pages = [(root_doc, "iris", "Iris Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        root_doc,
        "iris",
        "Iris Documentation",
        author,
        "iris",
        "Voice Command Assistant for Accessibility.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------

epub_title = project
epub_exclude_files = ["search.html"]
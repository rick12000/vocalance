import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Iris"
copyright = "2025, Riccardo Doyle"
author = "Riccardo Doyle"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "linkify",
    "replacements",
    "smartquotes",
    "tasklist",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"

autosummary_generate = True
autosummary_generate_overwrite = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

mermaid_version = "10.6.1"
mermaid_init_js = """
mermaid.initialize({
    startOnLoad: true,
    theme: 'default',
    securityLevel: 'loose',
    fontFamily: 'arial',
    flowchart: {
        useMaxWidth: false,
        htmlLabels: true,
        curve: 'basis',
        padding: 15,
        rankSpacing: 50,
        nodeSpacing: 50
    },
    themeVariables: {
        fontSize: '12px'
    }
});
"""

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "style_nav_header_background": "#2563eb",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "navigation_with_keys": True,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["layout-manager.js"]

html_context = {
    "display_github": True,
    "github_user": "rick12000",
    "github_repo": "iris",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_logo = "../iris/app/assets/logo/logo_full_text_full_size.png"
html_favicon = None

root_doc = "index"
pygments_style = "sphinx"

suppress_warnings = [
    "ref.doc",
    "epub.unknown_project_files",
]

nitpicky = False

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
}

latex_documents = [
    (root_doc, "iris.tex", "Iris Documentation", author, "manual"),
]

man_pages = [(root_doc, "iris", "Iris Documentation", [author], 1)]

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

epub_title = project
epub_exclude_files = ["search.html"]

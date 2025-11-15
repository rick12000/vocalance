import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Vocalance"
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
    theme: 'dark',
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
        fontSize: '14px',
        fontFamily: 'arial',
        primaryColor: '#60a5fa',
        primaryTextColor: '#ffffff',
        primaryBorderColor: '#3b82f6',
        lineColor: '#ffffff',
        secondaryColor: '#3b82f6',
        secondaryTextColor: '#ffffff',
        secondaryBorderColor: '#60a5fa',
        tertiaryColor: '#2563eb',
        tertiaryTextColor: '#ffffff',
        tertiaryBorderColor: '#60a5fa',
        background: '#374151',
        mainBkg: '#60a5fa',
        secondBkg: '#3b82f6',
        tertiaryBkg: '#2563eb',
        mainContrastColor: '#ffffff',
        darkMode: true,
        nodeBorder: '#3b82f6',
        clusterBkg: '#475569',
        clusterBorder: '#3b82f6',
        defaultLinkColor: '#ffffff',
        titleColor: '#ffffff',
        edgeLabelBackground: '#2563eb',
        nodeTextColor: '#ffffff',
        actorBorder: '#3b82f6',
        actorBkg: '#60a5fa',
        actorTextColor: '#ffffff',
        actorLineColor: '#ffffff',
        signalColor: '#ffffff',
        signalTextColor: '#ffffff',
        labelBoxBkgColor: '#2563eb',
        labelBoxBorderColor: '#60a5fa',
        labelTextColor: '#ffffff',
        loopTextColor: '#ffffff',
        noteBorderColor: '#3b82f6',
        noteBkgColor: '#60a5fa',
        noteTextColor: '#ffffff',
        activationBorderColor: '#3b82f6',
        activationBkgColor: '#60a5fa',
        sequenceNumberColor: '#ffffff',
        sectionBkgColor: '#60a5fa',
        altSectionBkgColor: '#3b82f6',
        sectionBkgColor2: '#2563eb',
        taskBorderColor: '#3b82f6',
        taskBkgColor: '#60a5fa',
        taskTextColor: '#ffffff',
        taskTextOutsideColor: '#ffffff',
        activeTaskBorderColor: '#3b82f6',
        activeTaskBkgColor: '#60a5fa',
        gridColor: '#ffffff',
        doneTaskBkgColor: '#10b981',
        doneTaskBorderColor: '#059669',
        critBkgColor: '#ef4444',
        critBorderColor: '#dc2626',
        todayLineColor: '#ffffff',
        labelColor: '#ffffff',
        errorBkgColor: '#ef4444',
        errorTextColor: '#ffffff',
        classText: '#ffffff',
        fillType0: '#60a5fa',
        fillType1: '#3b82f6',
        fillType2: '#2563eb',
        fillType3: '#2563eb',
        fillType4: '#1e40af',
        fillType5: '#1d4ed8',
        fillType6: '#93c5fd',
        fillType7: '#7dd3fc'
    }
});
"""

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "style_nav_header_background": "#1f2937",
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
    "github_repo": "vocalance",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_logo = "../vocalance/app/assets/logo/logo_full_text_full_size.png"
html_favicon = None

root_doc = "index"
pygments_style = "monokai"

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
    (root_doc, "vocalance.tex", "Documentation", author, "manual"),
]

man_pages = [(root_doc, "vocalance", "Documentation", [author], 1)]

texinfo_documents = [
    (
        root_doc,
        "vocalance",
        "Documentation",
        author,
        "vocalance",
        "Voice Command Assistant for Accessibility.",
        "Miscellaneous",
    ),
]

epub_title = project
epub_exclude_files = ["search.html"]

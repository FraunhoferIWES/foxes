import pathlib
import sys

repository_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root))

# -- Version extraction from pyproject.toml
pyproject_path = repository_root / "pyproject.toml"

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def get_version_from_pyproject(pyproject_path):
    if tomllib is not None and pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            return data.get("project", {}).get("version", "unknown")
    return "unknown"


__version__ = get_version_from_pyproject(pyproject_path)

# -- Project information -----------------------------------------------------

project = "foxes"
copyright = "2026, Fraunhofer IWES"
author = "Fraunhofer IWES"

version = __version__
release = __version__


# -- General configuration ---------------------------------------------------

extensions = [
    "numpydoc",
    "sphinx_immaterial",
    "autoapi.extension",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "myst_nb",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

# Source file types handled by Sphinx and MyST-NB.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

master_doc = "index"

language = "en"
# Exclude generated files and notebooks deferred from documentation execution.
exclude_patterns = [
    "build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "notebooks/dyn_wakes.ipynb",
    "notebooks/timelines.ipynb",
    "notebooks/sequential.ipynb",
]

pygments_style = None

# NumPy-style docstrings and stable cross-reference labels.
# Keep class members visible so constructors and inherited API members appear in
# the generated reference pages for the concrete model classes.
numpydoc_use_rtype = False
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False
autosectionlabel_prefix_document = True


# -- Options for autodoc ----------------------------------------------------
autodoc_typehints = "signature"
autodoc_class_signature = "separated"
autoapi_python_class_content = "both"

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_immaterial"
html_theme_options = {
    "site_url": "https://fraunhoferiwes.github.io/foxes.docs/index.html",
    "repo_url": "https://github.com/FraunhoferIWES/foxes",
    "icon": {"repo": "fontawesome/brands/github", "edit": "material/file-edit-outline"},
    "palette": {"primary": "teal"},
    "toc_title_is_page_title": True,
}
htmlhelp_basename = "foxesdoc"

latex_documents = [
    (master_doc, "foxes.tex", "foxes Documentation", "Fraunhofer IWES", "manual"),
]
man_pages = [(master_doc, "foxes", "foxes Documentation", [author], 1)]
texinfo_documents = [
    (
        master_doc,
        "foxes",
        "foxes Documentation",
        author,
        "foxes",
        "Farm Optimization and eXtended yield Evaluation Software",
        "Miscellaneous",
    ),
]
epub_title = project
epub_exclude_files = ["search.html"]

# -- AutoAPI configuration ------------------------------------------------------

autoapi_dirs = [
    str(repository_root / "foxes"),
]
autoapi_root = "_autoapi"
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "inherited-members",
]
autoapi_member_order = "groupwise"
autoapi_python_use_implicit_namespaces = False
autoapi_keep_files = False
autoapi_ignore = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/_version.py",
]

# -- Notebook configuration --------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

nb_execution_mode = "auto"
nb_execution_timeout = 300
nb_ipywidgets_js = {
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js": {
        "integrity": "sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=",
        "crossorigin": "anonymous",
    },
    "https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@*/dist/embed-amd.js": {
        "data-jupyter-widgets-cdn": "https://cdn.jsdelivr.net/npm/",
        "crossorigin": "anonymous",
    },
}


def remove_autoapi_submodule_heading(app, docname, source):
    if docname.startswith("autoapi/") and docname.endswith("/index"):
        source[0] = source[0].replace("Submodules\n----------\n\n", "")


def setup(app):
    app.connect("source-read", remove_autoapi_submodule_heading)


suppress_warnings = ["mystnb.unknown_mime_type"]

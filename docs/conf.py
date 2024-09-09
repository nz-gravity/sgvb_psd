author = "Jianan, Avi and friends"
copyright = "2024"
exclude_patterns = ["**.ipynb_checkpoints", ".DS_Store", "Thumbs.db", "_build"]
extensions = [
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "myst_nb",
    "jupyter_book",
    "sphinx_thebe",
    "sphinx_comments",
    "sphinx_external_toc",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_book_theme",
    "sphinx.ext.autodoc",
    "sphinx_jupyterbook_latex",
'IPython.sphinxext.ipython_console_highlighting',
]
external_toc_exclude_missing = False
external_toc_path = "_toc.yml"
html_baseurl = ""
# html_favicon = "https://icons8.com/icon/GgALV7LjQj0u/treble-clef"
# html_logo = "_static/logo.png"
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "search_bar_text": "Search this book...",
    "announcement": "",
    "analytics": {"google_analytics_id": ""},
    "use_edit_page_button": False,
    "external_links": [
        {
            "name": "Paper",
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        },
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/nz-gravity/sgvb_psd",  # required
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "show_prev_next": False,
    "footer_start": "copyright",
    "footer_end": "author",
    "navbar_start": ["navbar-logo", "navbar-nav"],
    "navbar_center": [],
    "navbar_end": ["navbar-icon-links"],
}
html_title = "SGVB-psd"
html_context = {
    "github_repo": "https://github.com/nz-gravity/sgvb_psdo",
    "github_version": "main",
    "doc_path": "docs",
}
latex_engine = "pdflatex"
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "linkify",
    "substitution",
    "tasklist",
]
myst_url_schemes = ["mailto", "http", "https"]
nb_execution_allow_errors = False
nb_execution_cache_path = ""
nb_execution_excludepatterns = []
nb_execution_in_temp = False
nb_execution_mode = "off"
nb_execution_timeout = 30
nb_output_stderr = "show"
numfig = True
pygments_style = "sphinx"
suppress_warnings = ["myst.domains"]
use_jupyterbook_latex = True
use_multitoc_numbering = True

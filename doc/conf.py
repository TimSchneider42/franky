import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

project = "franky"
author = "Tim Schneider"
copyright = f"%Y, {author}"
version = release = (ROOT_DIR / "VERSION").read_text().strip()

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

# Generating the Python API reference requires the franky module (including the compiled
# _franky extension) to be importable, e.g. by installing a franky-control wheel.
autosummary_generate = True
autodoc_member_order = "groupwise"
templates_path = ["_templates"]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "html_image",
]
myst_heading_anchors = 4

# tutorial.md is a redirect stub kept for old GitHub links; html/ is the Doxygen output
exclude_patterns = ["_build", "html", "tutorial.md", "requirements.txt"]

html_theme = "furo"
html_title = f"franky {release}"
html_logo = "logo.svg"
html_theme_options = {
    "source_repository": "https://github.com/TimSchneider42/franky",
    "source_branch": "master",
    "source_directory": "doc/",
}

# Set by the documentation workflow for dev builds to display a banner on every page
if os.environ.get("FRANKY_DOCS_ANNOUNCEMENT"):
    html_theme_options["announcement"] = os.environ["FRANKY_DOCS_ANNOUNCEMENT"]

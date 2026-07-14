import os

html_theme_options = {}

# Set by the documentation workflow for dev builds to display a banner on every page
if os.environ.get("FRANKY_DOCS_ANNOUNCEMENT"):
    html_theme_options["announcement"] = os.environ["FRANKY_DOCS_ANNOUNCEMENT"]

#!/usr/bin/env python3
"""Pre-commit wrapper around linkcheckmd that is aware of the branch deployments.

The documentation of master is deployed to the root of the GitHub Pages site, while the
documentation of other branches is deployed to branches/<name>/ (see documentation.yml).
Thus, on branches other than master, links to documentation pages are rewritten to point
to the deployment of the current branch (or dev if the current branch has none) before
checking, as they might refer to pages that do not exist on master yet. Links to the
package index (whl/) are exempt, as it exists only at the root.
"""

import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from linkcheckmd import check_links

PAGES_URL = "https://timschneider42.github.io/franky/"


def current_branch() -> str:
    # In GitHub Actions, the checkout is a detached HEAD, so the branch name has to be
    # taken from the environment (the PR head branch for pull_request events)
    branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")
    if branch:
        return branch
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def deployed_branch(branch: str) -> str:
    # Not every branch is deployed to branches/<name>/ (see documentation.yml), so check
    # whether a deployment exists and fall back to dev, which all branches merge into
    try:
        with urllib.request.urlopen(f"{PAGES_URL}branches/{branch}/", timeout=10):
            return branch
    except OSError:
        return "dev"


def main() -> int:
    paths = [Path(f) for f in sys.argv[1:]]
    if not paths:
        return 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        branch = current_branch()
        if branch != "master":
            docs_url = f"{PAGES_URL}branches/{deployed_branch(branch)}/"
            pattern = re.compile(re.escape(PAGES_URL) + r"(?!whl/|branches/)")
            print(f"Not on master; checking links against {docs_url}")
            checked_paths = []
            for path in paths:
                target = Path(tmp_dir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(pattern.sub(docs_url, path.read_text()))
                checked_paths.append(target.relative_to(tmp_dir))
            # Run from the temporary directory so that reported file names match the
            # original repository paths
            os.chdir(tmp_dir)
            paths = checked_paths
        bad = check_links(paths, ext=".md")
    # HTTP 429 means the server rate-limited the check (github.com in particular), not
    # that the link is broken
    bad = [(fn, url, code) for fn, url, code in bad if code != 429]
    return 22 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

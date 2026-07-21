#!/usr/bin/env python3
"""Pre-commit hook that checks HTTP links in Markdown files.

Links are deduplicated and checked concurrently, but requests to the same host are
serialized and spaced out to avoid rate limiting; if a host responds with HTTP 429
anyway, the request is retried with backoff (honoring Retry-After).

The documentation of master is deployed to the root of the GitHub Pages site, while the
documentation of other branches is deployed to branches/<name>/ (see documentation.yml).
Thus, on branches other than master, links to documentation pages are rewritten to point
to the deployment of the current branch (or dev if the current branch has none) before
checking, as they might refer to pages that do not exist on master yet. Links to the
package index (whl/) are exempt, as it exists only at the root.
"""

import concurrent.futures
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PAGES_URL = "https://timschneider42.github.io/franky/"

# URL regex from linkcheckmd (https://github.com/scivision/linkchecker-markdown),
# MIT License, Copyright (c) 2018 Michael Hirsch, Ph.D.
URL_PATTERN = re.compile(
    r"\((https?://[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[=a-zA-Z0-9\_\/\?\&\%\+\#\.\-]+)\)"
)
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
TIMEOUT = 10.0
MAX_ATTEMPTS = 4
HOST_REQUEST_GAP = 0.5

_host_locks: dict[str, threading.Lock] = {}
_host_locks_lock = threading.Lock()
_last_request_time: dict[str, float] = {}


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
        with urllib.request.urlopen(f"{PAGES_URL}branches/{branch}/", timeout=TIMEOUT):
            return branch
    except OSError:
        return "dev"


def _request(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.headers
    except urllib.error.HTTPError as e:
        return e.code, e.headers
    except OSError as e:
        return e, None


def _is_timeout(result) -> bool:
    return isinstance(result, TimeoutError) or (
        isinstance(result, urllib.error.URLError)
        and isinstance(result.reason, TimeoutError)
    )


def _retry_wait(headers, attempt: int) -> float:
    retry_after = headers.get("Retry-After", "") if headers else ""
    if retry_after.isdigit():
        return min(int(retry_after), 60)
    return 5.0 * 2**attempt


def check_url(url: str):
    """Return None if the URL is reachable, else the HTTP status code or exception."""
    host = urllib.parse.urlsplit(url).netloc
    with _host_locks_lock:
        lock = _host_locks.setdefault(host, threading.Lock())
    result = None
    for attempt in range(MAX_ATTEMPTS):
        with lock:
            wait = (
                _last_request_time.get(host, 0.0) + HOST_REQUEST_GAP - time.monotonic()
            )
            if wait > 0:
                time.sleep(wait)
            result, headers = _request(url)
            _last_request_time[host] = time.monotonic()
            if result != 429:
                break
            if attempt < MAX_ATTEMPTS - 1:
                # Sleep while still holding the lock to pause all requests to this host
                time.sleep(_retry_wait(headers, attempt))
    if isinstance(result, int) and result < 400:
        return None
    # A timeout means the server is slow, not that the link is broken
    if _is_timeout(result):
        return None
    return result


def check_links(links: dict[str, list[Path]]) -> list[tuple[Path, str, object]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = dict(zip(links, executor.map(check_url, links)))
    bad = []
    for url, result in results.items():
        if result is None:
            continue
        # HTTP 415 means the server rejected the checker (bot protection of doxygen.nl
        # on CI runners in particular), not that the link is broken
        if result == 415:
            continue
        bad.extend((path, url, result) for path in links[url])
    return bad


def main() -> int:
    paths = [Path(f) for f in sys.argv[1:]]
    contents = {path: path.read_text(errors="ignore") for path in paths}

    branch = current_branch()
    if branch != "master":
        docs_url = f"{PAGES_URL}branches/{deployed_branch(branch)}/"
        pattern = re.compile(re.escape(PAGES_URL) + r"(?!whl/|branches/)")
        print(f"Not on master; checking links against {docs_url}")
        contents = {
            path: pattern.sub(docs_url, text) for path, text in contents.items()
        }

    links: dict[str, list[Path]] = {}
    for path, text in contents.items():
        for match in URL_PATTERN.finditer(text):
            links.setdefault(match.group(1), []).append(path)

    bad = check_links(links)
    for path, url, result in bad:
        print(f"{path}: {url} ({result})")
    return 22 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

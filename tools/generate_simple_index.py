#!/usr/bin/env python3
"""Generate a PEP 503 "simple" package index from the wheels attached to GitHub
releases.

All wheels served through the package index carry a PEP 440 local version label
ending in `libfranka.<version>` (e.g. 1.1.4+libfranka.0.9.2 or
1.1.5.dev123+g8cb09e5.libfranka.0.9.2). This script groups all such wheels by their
libfranka version and writes one sub-index per libfranka version:

    <output>/libfranka-<version>/index.html                 (project list)
    <output>/libfranka-<version>/franky-control/index.html  (file list)
    <output>/index.html                                     (human-readable landing page)

Deployed to gh-pages under whl/, this allows installing franky for a specific
libfranka version with

    pip install franky-control --extra-index-url https://<pages-url>/whl/libfranka-<version>/

Wheels without a libfranka label are separate builds for PyPI (against the latest
libfranka version) and are deliberately not indexed here, as PyPI already serves
them. Labelled wheels outrank the PyPI wheel of the same version, so a sub-index
always takes precedence over PyPI where it applies.

Releases can be filtered by tag with --only-tag/--exclude-tag, which is used to build
per-branch indexes (branches/<branch>/whl/) from that branch's rolling pre-release
while keeping its wheels out of the main index.

Only uses the Python standard library. Authenticates with GITHUB_TOKEN or GH_TOKEN if
set (recommended, to avoid API rate limits).
"""

import argparse
import html
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path


def get_server_mappings(compatibility_file, libfranka_versions):
    comp = json.loads(compatibility_file.read_text())

    def version_tuple(v):
        return tuple(
            int(x) for x in v.replace(">=", "").replace("<", "").strip().split(".")
        )

    avail_versions_sorted = sorted(libfranka_versions, key=version_tuple)

    def get_best_libfranka(min_v_str, max_v_str):
        min_v = version_tuple(min_v_str) if min_v_str else (0,)
        max_v = version_tuple(max_v_str) if max_v_str else (999,)

        best = None
        for v in avail_versions_sorted:
            vt = version_tuple(v)
            if vt >= min_v and vt < max_v:
                best = v
        return best

    def map_servers(reqs):
        items = sorted(reqs.items(), key=lambda x: version_tuple(x[1]))
        mapping = {}
        for i in range(len(items)):
            server = items[i][0]
            min_v = items[i][1]
            max_v = items[i + 1][1] if i + 1 < len(items) else None

            best = get_best_libfranka(min_v, max_v)
            if best:
                mapping[server] = best
        return mapping

    return map_servers(comp.get("robot_server", {})), map_servers(
        comp.get("gripper_server", {})
    )


LIBFRANKA_LABEL_RE = re.compile(r"(?:^|\.)libfranka\.([0-9]+(?:\.[0-9]+)*)$")


def github_api_get(url, token=None):
    request = urllib.request.Request(url)
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def fetch_releases(repository, token=None):
    releases = []
    page = 1
    while True:
        batch = github_api_get(
            f"https://api.github.com/repos/{repository}/releases"
            f"?per_page=100&page={page}",
            token=token,
        )
        releases.extend(batch)
        if len(batch) < 100:
            return releases
        page += 1


def parse_wheel_filename(filename):
    """Return (normalized project name, version) or None if not a valid wheel name."""
    if not filename.endswith(".whl"):
        return None
    parts = filename[: -len(".whl")].split("-")
    # name-version-[build tag-]python tag-abi tag-platform tag
    if len(parts) < 5:
        return None
    name = re.sub(r"[-_.]+", "-", parts[0]).lower()
    return name, parts[1]


def collect_wheels(releases):
    """Group wheel assets by libfranka version as {libfranka: {project: [asset...]}}."""
    wheels = defaultdict(lambda: defaultdict(dict))
    for release in releases:
        if release.get("draft"):
            continue
        for asset in release.get("assets", []):
            filename = asset["name"]
            parsed = parse_wheel_filename(filename)
            if parsed is None:
                continue
            project, version = parsed
            local = version.partition("+")[2]
            match = LIBFRANKA_LABEL_RE.search(local)
            if match is None:
                # Wheels without a libfranka label are published to PyPI; no need to
                # serve them here
                continue
            digest = asset.get("digest") or ""
            wheels[match.group(1)][project].setdefault(
                filename,
                {
                    "url": asset["browser_download_url"],
                    "sha256": digest.removeprefix("sha256:") if digest else None,
                },
            )
    return wheels


def version_sort_key(libfranka_version):
    return tuple(int(part) for part in libfranka_version.split("."))


def write_page(path, title, body_lines):
    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "  <head>",
        '    <meta name="pypi:repository-version" content="1.0">',
        f"    <title>{html.escape(title)}</title>",
        "  </head>",
        "  <body>",
        *(f"    {line}" for line in body_lines),
        "  </body>",
        "</html>",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def generate_index(wheels, libfranka_versions, output_dir):
    # Ensure every supported libfranka version gets a sub-index, even if (so far) only
    # dev wheels or no wheels exist for it, so that the documented URLs always resolve
    all_versions = sorted(
        set(libfranka_versions) | set(wheels), key=version_sort_key, reverse=True
    )

    landing_lines = ["<h1>franky package index</h1>"]
    for libfranka_version in all_versions:
        sub_index = f"libfranka-{libfranka_version}"
        landing_lines.append(f'<a href="{sub_index}/">{sub_index}</a><br>')
        projects = wheels.get(libfranka_version, {"franky-control": {}})

        write_page(
            output_dir / sub_index / "index.html",
            f"Package index for libfranka {libfranka_version}",
            [f'<a href="{project}/">{project}</a><br>' for project in sorted(projects)],
        )

        for project, files in projects.items():
            file_lines = []
            for filename in sorted(files):
                info = files[filename]
                href = info["url"]
                if info["sha256"]:
                    href += f"#sha256={info['sha256']}"
                file_lines.append(
                    f'<a href="{html.escape(href)}">{html.escape(filename)}</a><br>'
                )
            write_page(
                output_dir / sub_index / project / "index.html",
                f"{project} wheels for libfranka {libfranka_version}",
                file_lines,
            )
    write_page(output_dir / "index.html", "franky package index", landing_lines)


def generate_server_index(wheels, mapping, prefix, output_dir):
    prefix_dir = output_dir / prefix
    prefix_dir.mkdir(parents=True, exist_ok=True)

    server_lines = []
    for server_version in sorted(mapping.keys(), key=lambda x: int(x), reverse=True):
        libfranka_version = mapping[server_version]
        link_path = prefix_dir / server_version
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(f"../libfranka-{libfranka_version}", link_path)
        server_lines.append(f'<a href="{server_version}/">{server_version}</a><br>')

    write_page(
        prefix_dir / "index.html", f"Available {prefix.replace('-', ' ')}", server_lines
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub repository as owner/name (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--libfranka-versions-file",
        type=Path,
        default=Path(__file__).parent.parent / ".github" / "libfranka-versions.json",
        help="JSON file listing all supported libfranka versions",
    )
    parser.add_argument(
        "--releases-json",
        type=Path,
        default=None,
        help="Read releases from this JSON file instead of the GitHub API (for testing)",
    )
    parser.add_argument(
        "--only-tag",
        action="append",
        metavar="TAG",
        help="Only index wheels from releases with these tags (repeatable)",
    )
    parser.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        metavar="TAG",
        help="Skip releases with these tags (repeatable)",
    )
    args = parser.parse_args()

    if args.releases_json is not None:
        releases = json.loads(args.releases_json.read_text())
    else:
        if not args.repository:
            parser.error("--repository is required if $GITHUB_REPOSITORY is not set")
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        releases = fetch_releases(args.repository, token=token)

    if args.only_tag is not None:
        releases = [r for r in releases if r.get("tag_name") in args.only_tag]
    releases = [r for r in releases if r.get("tag_name") not in args.exclude_tag]

    libfranka_versions = json.loads(args.libfranka_versions_file.read_text())[
        "versions"
    ]

    comp_file = Path(__file__).parent.parent / "doc" / "compatibility.json"
    robot_map, gripper_map = get_server_mappings(comp_file, libfranka_versions)

    wheels = collect_wheels(releases)
    generate_index(wheels, libfranka_versions, args.output)
    generate_server_index(wheels, robot_map, "by-robot-server-version", args.output)
    generate_server_index(wheels, gripper_map, "by-gripper-server-version", args.output)

    total = sum(
        len(files) for projects in wheels.values() for files in projects.values()
    )
    print(f"Indexed {total} wheels for {len(wheels)} libfranka versions.")
    for libfranka_version in sorted(wheels, key=version_sort_key, reverse=True):
        count = sum(len(files) for files in wheels[libfranka_version].values())
        print(f"  libfranka {libfranka_version}: {count} wheels")


if __name__ == "__main__":
    sys.exit(main())

"""scikit-build-core dynamic metadata provider for the franky-control package version.

VERSION holds the last released version. Dev builds (FRANKY_DEV_BUILD_NUMBER set) are
versioned as dev releases of the next patch version, as PEP 440 orders X.Y.Z.devN before X.Y.Z.
FRANKY_DEV_BUILD_HASH and FRANKY_LIBFRANKA_VERSION_LABEL are appended as local version
segments; the latter distinguishes builds against different libfranka versions and must be
unset for PyPI wheels, as PyPI rejects local version labels.
"""

import os
from pathlib import Path


def get_version():
    version = (Path(__file__).resolve().parent.parent / "VERSION").read_text().strip()

    dev_build_number = os.environ.get("FRANKY_DEV_BUILD_NUMBER")
    dev_build_hash = os.environ.get("FRANKY_DEV_BUILD_HASH")
    libfranka_version = os.environ.get("FRANKY_LIBFRANKA_VERSION_LABEL")
    if dev_build_number:
        major, minor, patch = version.split(".")
        version = "{}.{}.{}.dev{}".format(
            major, minor, int(patch) + 1, dev_build_number
        )

    local_segments = []
    if dev_build_number and dev_build_hash:
        local_segments.append("g{}".format(dev_build_hash))
    if libfranka_version:
        local_segments.append("libfranka.{}".format(libfranka_version))
    if local_segments:
        version += "+" + ".".join(local_segments)
    return version


def dynamic_metadata(field, settings=None):
    if field != "version":
        raise ValueError("Only the 'version' field is supported")
    if settings:
        raise ValueError("No settings are supported")
    return get_version()


if __name__ == "__main__":
    print(get_version())

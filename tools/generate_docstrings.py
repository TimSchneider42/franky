#!/usr/bin/env python3
"""Regenerate python/docstrings.hpp from the doxygen comments in the C++ headers.

This script wraps pybind11_mkdoc (https://github.com/pybind/pybind11_mkdoc). It is invoked by the
franky-docstrings CMake target, which passes the include directories and preprocessor defines of
the actual build, so the generated docstrings always match the libfranka version franky is built
against. All franky headers and all public libfranka headers are discovered automatically.

Requires pybind11_mkdoc and libclang (e.g. apt install libclang-dev && pip install pybind11_mkdoc).
"""

import argparse
import re
import sys
from pathlib import Path


def find_headers(include_dirs: "list[Path]") -> "list[Path]":
    franky_include_dir = next(
        (d for d in include_dirs if (d / "franky").is_dir()), None
    )
    if franky_include_dir is None:
        sys.exit(
            "Error: none of the given include directories contains the franky headers."
        )
    franka_include_dir = next(
        (d for d in include_dirs if (d / "franka" / "robot.h").is_file()), None
    )
    if franka_include_dir is None:
        sys.exit(
            "Error: none of the given include directories contains the libfranka headers."
        )
    franky_headers = sorted((franky_include_dir / "franky").rglob("*.hpp"))
    franka_headers = sorted((franka_include_dir / "franka").glob("*.h"))
    return franky_headers + franka_headers


LIBFRANKA_NOTICE = """\

  The docstrings of the franka::* entities are extracted from the headers of libfranka
  (https://github.com/frankaemika/libfranka), Copyright (c) Franka Robotics GmbH, which is
  licensed under the Apache License, Version 2.0
  (http://www.apache.org/licenses/LICENSE-2.0).
"""


def postprocess(output_file: Path) -> None:
    # Clean up doxygen artifacts that pybind11_mkdoc does not handle (mostly from the libfranka
    # headers): "param [in]:" direction annotations and "\ " escapes.
    content = output_file.read_text()
    content = re.sub(r" \[(?:in|out|in,out)\]:", ":", content)
    content = content.replace("\\ ", " ")
    # Append the libfranka license notice to the header comment written by pybind11_mkdoc.
    mkdoc_header_end = "extracted by pybind11_mkdoc.\n"
    if mkdoc_header_end not in content:
        sys.exit(
            "Error: did not find the pybind11_mkdoc header comment to attach the notice to."
        )
    content = content.replace(mkdoc_header_end, mkdoc_header_end + LIBFRANKA_NOTICE, 1)
    output_file.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Path of the generated header."
    )
    parser.add_argument(
        "--include-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Include directories of franky and all of its dependencies.",
    )
    parser.add_argument(
        "--defines",
        nargs="*",
        default=[],
        help="Preprocessor defines (without leading -D), e.g. the FRANKA_X_Y version defines.",
    )
    parser.add_argument(
        "--std", default="c++17", help="C++ standard to parse the headers with."
    )
    parser.add_argument(
        "--width", type=int, default=100, help="Width to wrap the docstrings at."
    )
    args = parser.parse_args()

    try:
        from pybind11_mkdoc import mkdoc
    except ImportError:
        sys.exit(
            "Error: pybind11_mkdoc is not installed. Install it with libclang, e.g.:\n"
            "  apt install libclang-dev && pip install pybind11_mkdoc 'clang~=14.0'"
        )

    include_dirs = [d for d in args.include_dirs if d.is_dir()]
    headers = find_headers(include_dirs)

    mkdoc_args = [f"-std={args.std}"]
    mkdoc_args += [f"-I{d}" for d in include_dirs]
    mkdoc_args += [f"-D{d}" for d in args.defines]
    mkdoc_args += [str(h) for h in headers]

    mkdoc(mkdoc_args, args.width, str(args.output))
    postprocess(args.output)
    print(f"Generated {args.output} from {len(headers)} headers.")


if __name__ == "__main__":
    main()

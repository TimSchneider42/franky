import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

with (Path(__file__).resolve().parent / "README.md").open() as readme_file:
    long_description = readme_file.read()


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()

        franky_lib_cache_dir = os.environ.get("FRANKY_LIB_CACHE_DIR")

        prebuilt_lib = None
        if franky_lib_cache_dir is not None:
            candidate = Path(franky_lib_cache_dir) / "libfranky.so"
            if candidate.exists():
                prebuilt_lib = candidate

        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = tuple(
            int(component)
            for component in re.search(r"version\s*([\d.]+)", out.decode())
            .group(1)
            .split(".")
        )
        cmake_version += (0,) * (3 - len(cmake_version))
        if cmake_version < (3, 11, 0):
            raise RuntimeError("CMake >= 3.11.0 is required")

        build_type = os.environ.get("BUILD_TYPE", "Release")
        build_args = ["--config", build_type]

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}/".format(ext_dir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DCMAKE_BUILD_TYPE={}".format(build_type),
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_PYTHON_STUBS=ON",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        ]

        if prebuilt_lib is not None:
            ext_dir.mkdir(exist_ok=True, parents=True)
            dest = ext_dir / prebuilt_lib.name
            shutil.copy2(prebuilt_lib, dest)
            cmake_args.append("-DFRANKY_PREBUILT_LIB={}".format(dest))
        else:
            # This is a CMake cache variable, so explicitly clear a prebuilt
            # path left by an earlier build in the same build directory.
            cmake_args.append("-DFRANKY_PREBUILT_LIB=")

        Path(self.build_temp).mkdir(exist_ok=True, parents=True)

        subprocess.check_call(
            ["cmake", str(Path(".").resolve())] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "_franky"]
            + build_args
            + ["--", "-j", str(multiprocessing.cpu_count())],
            cwd=self.build_temp,
        )

        if franky_lib_cache_dir is not None and prebuilt_lib is None:
            Path(franky_lib_cache_dir).mkdir(exist_ok=True, parents=True)
            shutil.copy2(
                ext_dir / "libfranky.so", Path(franky_lib_cache_dir) / "libfranky.so"
            )


with (Path(__file__).resolve().parent / "VERSION").open() as f:
    version = f.read().strip()

dev_build_number = os.environ.get("FRANKY_DEV_BUILD_NUMBER")
dev_build_hash = os.environ.get("FRANKY_DEV_BUILD_HASH")
# Distinguishes builds against different libfranka versions; unset for PyPI wheels
libfranka_version = os.environ.get("FRANKY_LIBFRANKA_VERSION_LABEL")
if dev_build_number:
    # VERSION holds the last released version. Dev builds are versioned as dev
    # releases of the next patch version, as PEP 440 orders X.Y.Z.devN before X.Y.Z.
    major, minor, patch = version.split(".")
    version = "{}.{}.{}.dev{}".format(major, minor, int(patch) + 1, dev_build_number)

local_segments = []
if dev_build_number and dev_build_hash:
    local_segments.append("g{}".format(dev_build_hash))
if libfranka_version:
    local_segments.append("libfranka.{}".format(libfranka_version))
if local_segments:
    version += "+" + ".".join(local_segments)

setup(
    name="franky-control",
    version=version,
    description="High-level control library for Franka robots.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tim Schneider",
    author_email="tim@robot-learning.de",
    url="https://github.com/TimSchneider42/franky",
    packages=find_packages(),
    license="MIT",
    ext_modules=[Extension("franky/_franky", [])],
    cmdclass=dict(build_ext=CMakeBuild),
    keywords=["control", "robotics", "franka", "franka-panda", "franka-research-3"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: C++",
    ],
    install_requires=[
        "numpy",
        "websockets>=11",
        "typing_extensions; python_version < '3.8'",
    ],
    python_requires=">=3.7",
    zip_safe=False,
)

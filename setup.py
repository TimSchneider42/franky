import multiprocessing
import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

with (Path(__file__).resolve().parent / "README.md").open() as readme_file:
    long_description = readme_file.read()


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = LooseVersion(
            re.search(r"version\s*([\d.]+)", out.decode()).group(1)
        )
        if cmake_version < LooseVersion("3.10.0"):
            raise RuntimeError("CMake >= 3.10.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()

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


with (Path(__file__).resolve().parent / "VERSION").open() as f:
    version = f.read()

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
    license="LGPL-3.0-or-later",
    ext_modules=[Extension("franky/_franky", [])],
    cmdclass=dict(build_ext=CMakeBuild),
    keywords=["robot", "robotics", "trajectory-generation", "motion-control"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: C++",
    ],
    install_requires=["numpy"],
    python_requires=">=3.7",
    zip_safe=False,
)

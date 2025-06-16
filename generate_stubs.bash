#!/bin/bash

set -e

PYTHON=$1
BUILD_DIR=$2
LIB_FILE=$3

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

STUBS_GEN_DIR="${BUILD_DIR}/franky-stubs-gen"
LIB_DIR="$(dirname "${LIB_FILE}")"

cd "${BUILD_DIR}"
rm -rf "${STUBS_GEN_DIR}"
mkdir -p "${STUBS_GEN_DIR}"

${PYTHON} -m venv "${STUBS_GEN_DIR}/venv"
source "${STUBS_GEN_DIR}/venv/bin/activate"
pip install --upgrade pip --no-cache-dir > /dev/null
pip install setuptools --no-cache-dir > /dev/null
(cd ${SCRIPT_DIR} && python "setup.py" egg_info -e "${STUBS_GEN_DIR}" > /dev/null)
pip install -r "${STUBS_GEN_DIR}/franky_control.egg-info/requires.txt" --no-cache-dir > /dev/null
pip install pybind11-stubgen==2.2.2  --no-cache-dir > /dev/null

PYTHONPATH="${LIB_DIR}" "${SCRIPT_DIR}/custom_stubgen.py" -o "${LIB_DIR}" _franky

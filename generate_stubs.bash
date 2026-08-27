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
# Install the runtime dependencies of franky-control (from pyproject.toml) into the venv
pip install tomli --no-cache-dir > /dev/null
REQUIREMENTS_FILE="${STUBS_GEN_DIR}/requirements.txt"
python - "${SCRIPT_DIR}/pyproject.toml" > "${REQUIREMENTS_FILE}" <<'PYEOF'
import sys
import tomli

with open(sys.argv[1], "rb") as f:
    print("\n".join(tomli.load(f)["project"]["dependencies"]))
PYEOF
pip install -r "${REQUIREMENTS_FILE}" --no-cache-dir > /dev/null
pip install pybind11-stubgen==2.5.5 --no-cache-dir > /dev/null

PYTHONPATH="${LIB_DIR}" "${SCRIPT_DIR}/custom_stubgen.py" -o "${LIB_DIR}" _franky

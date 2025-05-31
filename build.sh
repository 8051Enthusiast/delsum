#!/usr/bin/env bash
# buildscript for the ci in order to build the
# static libraries
set -eu -o pipefail

GF2X_VERSION=1c4974a44bc69a5a5111b44871d96c0cc16c0144
GF2X_REPO="https://gitlab.inria.fr/gf2x/gf2x"

TMP_DIR="$(mktemp -d)"
PREFIX="$PWD/output"
echo Using PREFIX="$PREFIX"
cd "$TMP_DIR"
mkdir -p "$PREFIX"
git clone "$GF2X_REPO"
cd gf2x
git checkout "$GF2X_VERSION"
autoreconf --install
if [ -n "${HOST-}" ]; then
  HOST_ARG="--host=$HOST"
fi
./configure --host=wasm32 --disable-shared --enable-static --prefix="$PREFIX"
make -j "$(nproc)"
make install

#!/usr/bin/env bash
# buildscript for the ci in order to build the
# static libraries
# (only for x64 linux)
set -eu -o pipefail

GF2X_VERSION=1.3.0
GF2X_LINK="https://gitlab.inria.fr/gf2x/gf2x/uploads/c46b1047ba841c20d1225ae73ad6e4cd/gf2x-$GF2X_VERSION.tar.gz"
GMP_VERSION=6.3.0
GMP_LINK="https://gmplib.org/download/gmp/gmp-$GMP_VERSION.tar.lz"
NTL_VERSION=11.5.1
NTL_LINK="https://shoup.net/ntl/ntl-$NTL_VERSION.tar.gz"

TMP_DIR="$(mktemp -d)"
PREFIX="$PWD/output"
echo Using PREFIX="$PREFIX"
pushd "$TMP_DIR"
mkdir -p "$PREFIX"
wget "$GF2X_LINK"
tar xf gf2x-$GF2X_VERSION.tar.gz
cd gf2x-$GF2X_VERSION
# westmere is the first one with clmul
./configure CFLAGS=-march=westmere --disable-shared --enable-static --prefix="$PREFIX"
make -j "$(nproc)"
make install
cd ..
wget "$GMP_LINK"
tar xf gmp-$GMP_VERSION.tar.lz
cd gmp-$GMP_VERSION
./configure --disable-shared --enable-static --prefix="$PREFIX"
make -j "$(nproc)"
make install
cd ..
wget "$NTL_LINK"
tar xf ntl-$NTL_VERSION.tar.gz
cd ntl-$NTL_VERSION/src
./configure NTL_GF2X_LIB=on DEF_PREFIX="$PREFIX" NATIVE=off
make -j "$(nproc)"
make install
popd

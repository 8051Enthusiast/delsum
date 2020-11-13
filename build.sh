#!/usr/bin/env bash
set -eu -o pipefail

GF2X_VERSION=1.3.0
GF2X_LINK="https://gitlab.inria.fr/gf2x/gf2x/uploads/c46b1047ba841c20d1225ae73ad6e4cd/gf2x-$GF2X_VERSION.tar.gz"
GMP_VERSION=6.2.0
NTL_VERSION=11.4.3

TMP_DIR="$(mktemp -d)"
PREFIX="$1"
echo Using PREFIX="$PREFIX"
pushd "$TMP_DIR"
mkdir -p "$PREFIX"
wget "$GF2X_LINK"
tar xf gf2x-$GF2X_VERSION.tar.gz
cd gf2x-$GF2X_VERSION
./configure --disable-shared --enable-static --disable-hardware-specific-code --prefix="$PREFIX"
make -j "$(nproc)"
make install
cd ..
wget https://gmplib.org/download/gmp/gmp-$GMP_VERSION.tar.lz
tar xf gmp-$GMP_VERSION.tar.lz
cd gmp-$GMP_VERSION
./configure --disable-shared --enable-static --prefix="$PREFIX"
make -j "$(nproc)"
make install
cd ..
wget https://shoup.net/ntl/ntl-$NTL_VERSION.tar.gz
tar xf ntl-$NTL_VERSION.tar.gz
cd ntl-$NTL_VERSION/src
./configure NTL_GF2X_LIB=on DEF_PREFIX="$PREFIX" NATIVE=off
make -j "$(nproc)"
make install
popd

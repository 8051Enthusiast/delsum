#!/usr/bin/env bash
# buildscript for the ci in order to build the
# static libraries
set -eu -o pipefail
[ $(id -u) -eq 0 ] && (useradd -u $(stat -c %u README.md) -m build && su -c "$0" build)
[ $(id -u) -eq 0 ] && exit 0
export GF2POLY_STATIC_LIB=1
export GF2POLY_LIBRARY_PATH=/opt/mxe/usr/lib
export PATH=$PATH:$HOME/.cargo/bin:/opt/mxe/usr/bin
rustup install stable
rustup target add x86_64-pc-windows-gnu
cargo fetch
mkdir -p .cargo
echo -e "[target.x86_64-pc-windows-gnu]\nrustflags = \"-Cdlltool=x86_64-w64-mingw32.shared-dlltool -Cdefault-linker-libraries=yes -Ctarget-cpu=westmere -Clto=false\"\nlinker = \"x86_64-w64-mingw32.shared-gcc\"" >.cargo/config.toml
sed -i -e '/lto = true/d' Cargo.toml
cargo build --target x86_64-pc-windows-gnu --profile=dist

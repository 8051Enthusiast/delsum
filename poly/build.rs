fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("src/poly.cc")
        .flag_if_supported("-std=c++14")
        .compile("poly");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=include/poly.hh");
    println!("cargo:rerun-if-changed=src/poly.cc");
    println!("cargo:rustc-link-lib=ntl");
    println!("cargo:rustc-link-lib=gmp");
    println!("cargo:rustc-link-lib=gf2x");
}

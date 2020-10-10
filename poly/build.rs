fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("poly_ntl/poly.cc")
        .flag_if_supported("-std=c++14")
        .compile("poly");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=poly_ntl/poly.hh");
    println!("cargo:rerun-if-changed=poly_ntl/poly.cc");
    println!("cargo:rustc-link-lib=ntl");
    println!("cargo:rustc-link-lib=gmp");
    println!("cargo:rustc-link-lib=gf2x");
}

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=include/poly.hh");
    println!("cargo:rerun-if-changed=src/poly.cc");
    let mut link_type = "dylib";
    println!("cargo:rerun-if-env-changed=DELSUM_STATIC_LIBS");
    println!("cargo:rerun-if-env-changed=DELSUM_NTL_LIB_PATH");
    println!("cargo:rerun-if-env-changed=DELSUM_GF2X_LIB_PATH");
    println!("cargo:rerun-if-env-changed=DELSUM_GMP_LIB_PATH");
    println!("cargo:rerun-if-env-changed=DELSUM_NTL_INCLUDE");
    let mut build = cxx_build::bridge("src/lib.rs");
    for (key, value) in std::env::vars() {
        match key.as_str() {
            "DELSUM_STATIC_LIBS" => {
                if value == "1" {
                    link_type = "static";
                }
            },
            "DELSUM_NTL_LIB_PATH" | "DELSUM_GF2X_LIB_PATH" | "DELSUM_GMP_LIB_PATH" => {
                println!("cargo:rustc-link-search=native={}", value);
            },
            "DELSUM_NTL_INCLUDE" => {
                build.include(value);
            },
            _ => continue,
        }
    }
    println!("cargo:rustc-link-lib={}=ntl", link_type);
    println!("cargo:rustc-link-lib={}=gf2x", link_type);
    // gmp is required for thread safety apparently?
    println!("cargo:rustc-link-lib={}=gmp", link_type);
    build
        // needed for NTL
        .flag_if_supported("-fpermissive")
        .file("src/poly.cc")
        .flag_if_supported("-std=c++14")
        .flag_if_supported("-Wno-deprecated-copy")
        .flag_if_supported("-Wno-unused-parameter");
    build.compile("delsum_poly");
}

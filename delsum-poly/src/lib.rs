//! This is just a wrapper for the ntl gf2x polynomials for use in delsum
use std::fmt::Display;
use std::ops;

pub use cxx::UniquePtr;
pub use ffi::*;
pub type PolyPtr = UniquePtr<Poly>;
pub type PolyRemPtr = UniquePtr<PolyRem>;
impl Display for Poly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in (0..=deg(self)).rev() {
            write!(f, "{}", u8::from(self.coeff(i)))?;
        }
        if self.is_zero() {
            write!(f, "0")?;
        }
        Ok(())
    }
}

pub fn div_checked(a: &Poly, b: &Poly) -> Option<PolyPtr> {
    let mut r = copy_poly(a);
    let succ = r.pin_mut().div_to_checked(b);
    if succ {
        Some(r)
    } else {
        None
    }
}

impl ops::Add for &Poly {
    type Output = PolyPtr;

    fn add(self, rhs: Self) -> Self::Output {
        crate::add(self, rhs)
    }
}
impl ops::AddAssign<&Poly> for PolyPtr {
    fn add_assign(&mut self, rhs: &Poly) {
        self.pin_mut().add_to(rhs);
    }
}
impl ops::Mul for &Poly {
    type Output = PolyPtr;

    fn mul(self, rhs: Self) -> Self::Output {
        crate::mul(self, rhs)
    }
}
impl ops::MulAssign<&Poly> for PolyPtr {
    fn mul_assign(&mut self, rhs: &Poly) {
        self.pin_mut().mul_to(rhs);
    }
}
impl ops::Div for &Poly {
    type Output = PolyPtr;

    fn div(self, rhs: Self) -> Self::Output {
        crate::div(self, rhs)
    }
}
impl ops::DivAssign<&Poly> for PolyPtr {
    fn div_assign(&mut self, rhs: &Poly) {
        self.pin_mut().div_to(rhs);
    }
}
impl ops::Rem for &Poly {
    type Output = PolyPtr;

    fn rem(self, rhs: Self) -> Self::Output {
        crate::rem(self, rhs)
    }
}
impl ops::RemAssign<&Poly> for PolyPtr {
    fn rem_assign(&mut self, rhs: &Poly) {
        self.pin_mut().rem_to(rhs);
    }
}
impl ops::AddAssign<&PolyRem> for PolyRemPtr {
    fn add_assign(&mut self, rhs: &PolyRem) {
        self.pin_mut().add_to(rhs);
    }
}
impl ops::MulAssign<&PolyRem> for PolyRemPtr {
    fn mul_assign(&mut self, rhs: &PolyRem) {
        self.pin_mut().mul_to(rhs);
    }
}
impl ops::DivAssign<&PolyRem> for PolyRemPtr {
    fn div_assign(&mut self, rhs: &PolyRem) {
        self.pin_mut().div_to(rhs);
    }
}

// ntl author says it is thread safe
unsafe impl Send for Poly {}
unsafe impl Sync for Poly {}
unsafe impl Send for PolyRem {}
unsafe impl Sync for PolyRem {}

#[cxx::bridge(namespace = poly)]
mod ffi {
    struct PolyI64Pair {
        poly: UniquePtr<Poly>,
        l: i64,
    }

    unsafe extern "C++" {
        include!("delsum-poly/include/poly.hh");

        type Poly;
        fn new_poly_shifted(bytes: &[u8], shift: i64, msb_first: bool) -> UniquePtr<Poly>;
        fn new_poly(bytes: &[u8]) -> UniquePtr<Poly>;
        fn new_zero() -> UniquePtr<Poly>;
        fn to_bytes(self: &Poly, min_bytes: i64) -> UniquePtr<CxxVector<u8>>;
        fn deg(a: &Poly) -> i64;
        fn add_to(self: Pin<&mut Poly>, b: &Poly);
        fn mul_to(self: Pin<&mut Poly>, b: &Poly);
        fn div_to(self: Pin<&mut Poly>, b: &Poly);
        fn gcd_to(self: Pin<&mut Poly>, b: &Poly);
        fn rem_to(self: Pin<&mut Poly>, b: &Poly);
        fn div_to_checked(self: Pin<&mut Poly>, b: &Poly) -> bool;
        fn sqr(self: Pin<&mut Poly>);
        fn coeff(self: &Poly, idx: i64) -> bool;
        fn eq(self: &Poly, b: &Poly) -> bool;
        fn is_zero(self: &Poly) -> bool;
        fn add(b: &Poly, c: &Poly) -> UniquePtr<Poly>;
        fn mul(b: &Poly, c: &Poly) -> UniquePtr<Poly>;
        fn div(b: &Poly, c: &Poly) -> UniquePtr<Poly>;
        fn gcd(b: &Poly, c: &Poly) -> UniquePtr<Poly>;
        fn xgcd(x: Pin<&mut Poly>, y: Pin<&mut Poly>, b: &Poly, c: &Poly) -> UniquePtr<Poly>;
        fn rem(b: &Poly, c: &Poly) -> UniquePtr<Poly>;
        fn power(p: &Poly, n: i64) -> UniquePtr<Poly>;
        fn shift(p: &Poly, n: i64) -> UniquePtr<Poly>;
        fn copy_poly(p: &Poly) -> UniquePtr<Poly>;
        fn factor(p: &Poly, verbosity: i64) -> UniquePtr<CxxVector<PolyI64Pair>>;

        type PolyRem;
        fn new_polyrem(rem: &Poly, m: &Poly) -> UniquePtr<PolyRem>;
        fn add_to(self: Pin<&mut PolyRem>, b: &PolyRem);
        fn mul_to(self: Pin<&mut PolyRem>, b: &PolyRem);
        fn div_to(self: Pin<&mut PolyRem>, b: &PolyRem);
        fn sqr(self: Pin<&mut PolyRem>);
        fn rep(self: &PolyRem) -> UniquePtr<Poly>;
        fn powermod(p: &PolyRem, n: i64) -> UniquePtr<PolyRem>;
        fn copy_polyrem(p: &PolyRem) -> UniquePtr<PolyRem>;
    }
}

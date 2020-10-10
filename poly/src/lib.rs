use std::fmt::Display;
use std::ops;

pub use ffi::*;
pub use cxx::UniquePtr;
impl Display for Poly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in (0..=deg(self)).rev() {
            write!(f, "{}", u8::from(self.coeff(i)))?;
        }
        Ok(())
    }
}

pub fn div_checked(a: &Poly, b: &Poly) -> Option<UniquePtr<Poly>> {
    let mut r = copy_poly(a);
    let succ = r.div_to_checked(b);
    if succ {
        Some(r)
    } else {
        None
    }
}

impl ops::Add for &Poly {
    type Output = UniquePtr<Poly>;

    fn add(self, rhs: Self) -> Self::Output {
        crate::add(self, rhs)
    }
}
impl ops::AddAssign<&Poly> for UniquePtr<Poly> {
    fn add_assign(&mut self, rhs: &Poly) {
        self.add_to(rhs);
    }
}
impl ops::Mul for &Poly {
    type Output = UniquePtr<Poly>;

    fn mul(self, rhs: Self) -> Self::Output {
        crate::mul(self, rhs)
    }
}
impl ops::MulAssign<&Poly> for UniquePtr<Poly> {
    fn mul_assign(&mut self, rhs: &Poly) {
        self.mul_to(rhs);
    }
}
impl ops::Div for &Poly {
    type Output = UniquePtr<Poly>;

    fn div(self, rhs: Self) -> Self::Output {
        crate::div(self, rhs)
    }
}
impl ops::DivAssign<&Poly> for UniquePtr<Poly> {
    fn div_assign(&mut self, rhs: &Poly) {
        self.div_to(rhs);
    }
}
impl ops::Rem for &Poly {
    type Output = UniquePtr<Poly>;

    fn rem(self, rhs: Self) -> Self::Output {
        crate::rem(self, rhs)
    }
}
impl ops::RemAssign<&Poly> for UniquePtr<Poly> {
    fn rem_assign(&mut self, rhs: &Poly) {
        self.rem_to(rhs);
    }
}
impl ops::AddAssign<&PolyRem> for UniquePtr<PolyRem> {
    fn add_assign(&mut self, rhs: &PolyRem) {
        self.add_to(rhs);
    }
}
impl ops::MulAssign<&PolyRem> for UniquePtr<PolyRem> {
    fn mul_assign(&mut self, rhs: &PolyRem) {
        self.mul_to(rhs);
    }
}
impl ops::DivAssign<&PolyRem> for UniquePtr<PolyRem> {
    fn div_assign(&mut self, rhs: &PolyRem) {
        self.div_to(rhs);
    }
}

#[cxx::bridge(namespace = poly)]
mod ffi {
    struct PolyI64Pair {
        poly: UniquePtr<Poly>,
        l: i64,
    }
    extern "C" {
        // somehow this is not from workspace root but from whole package root?
        include!("poly/poly_ntl/poly.hh");
        type Poly;
        fn new_poly_shifted(bytes: &[u8], shift: i64) -> UniquePtr<Poly>;
        fn new_poly(bytes: &[u8]) -> UniquePtr<Poly>;
        fn new_zero() -> UniquePtr<Poly>;
        fn deg(a: &Poly) -> i64;
        fn add_to(self: &mut Poly, b: &Poly);
        fn mul_to(self: &mut Poly, b: &Poly);
        fn div_to(self: &mut Poly, b: &Poly);
        fn gcd_to(self: &mut Poly, b: &Poly);
        fn rem_to(self: &mut Poly, b: &Poly);
        fn div_to_checked(self: &mut Poly, b: &Poly) -> bool;
        fn coeff(self: &Poly, idx: i64) -> bool;
        fn eq(self: &Poly, b: &Poly) -> bool;
        fn is_zero(self: &Poly) -> bool;
        fn add(b: &Poly, c: &Poly) -> UniquePtr<Poly>;
        fn mul(b: &Poly, c: &Poly) -> UniquePtr<Poly>;
        fn div(b: &Poly, c: &Poly) -> UniquePtr<Poly>;
        fn gcd(b: &Poly, c: &Poly) -> UniquePtr<Poly>;
        fn rem(b: &Poly, c: &Poly) -> UniquePtr<Poly>;
        fn power(p: &Poly, n: i64) -> UniquePtr<Poly>;
        fn copy_poly(p: &Poly) -> UniquePtr<Poly>;
        fn squarefree_decomp(p: &Poly) -> UniquePtr<CxxVector<PolyI64Pair>>;
        fn equdeg_decomp(p: &Poly, d: i64) -> UniquePtr<CxxVector<Poly>>;
        type PolyRem;
        fn new_polyrem(rem: &Poly, m: &Poly) -> UniquePtr<PolyRem>;
        fn add_to(self: &mut PolyRem, b: &PolyRem);
        fn mul_to(self: &mut PolyRem, b: &PolyRem);
        fn div_to(self: &mut PolyRem, b: &PolyRem);
        fn sqr(self: &mut PolyRem);
        fn rep(self: &PolyRem) -> UniquePtr<Poly>;
        fn powermod(p: &PolyRem, n: i64) -> UniquePtr<PolyRem>;
        fn copy_polyrem(p: &PolyRem) -> UniquePtr<PolyRem>;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn degree() {
        let p = new_poly(&[0xff, 0xff, 0xff, 0xff]);
        assert_eq!(deg(&p), 31);
        let q = add(&p, &p);
        assert_eq!(deg(&q), -1);
        let q = mul(&p, &p);
        assert_eq!(deg(&q), 62);
        assert_eq!(deg(&new_poly(&[0x80, 0, 0])), 23);
    }
}
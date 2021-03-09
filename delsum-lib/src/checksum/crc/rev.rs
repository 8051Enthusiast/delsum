//! This module contains the function(s) for reversing the parameters for a CRC algorithm with given bytes and checksums.
//!
//! Generally, to find out the parameters, the checksums and their width are needed, and 3 of the following (with at least one file):
//! * value of `init`
//! * value of `xorout`
//! * value of `module`
//! * a file with checksum
//! * a different file with checksum
//! * yet another different file checksum
//!
//! If `init` is not known, it is neccessary to know two checksums of files with different lengths.
//! In case only checksums of files with a set length are required, setting `init = 0` is sufficient.
use super::{CRCBuilder, CRC};
use crate::checksum::endian::{bytes_to_int, int_to_bytes, wordspec_combos, Endian, WordSpec};
use crate::checksum::CheckReverserError;
use crate::utils::{cart_prod, unresult_iter};
use delsum_poly::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::convert::TryInto;
use std::pin::Pin;

/// Find the parameters of a CRC algorithm.
///
/// `spec` contains the known parameters of the algorithm (by setting the corresponding values in the builder).
/// `chk_bytes` are pairs of files and their checksums.
/// `verbosity` makes the function output what it is doing.
///
/// The `width` parameter of the builder has to be set.
pub fn reverse_crc<'a>(
    spec: &CRCBuilder<u128>,
    chk_bytes: &'a [(&[u8], Vec<u8>)],
    verbosity: u64,
    extended_search: bool,
) -> impl Iterator<Item = Result<CRC<u128>, CheckReverserError>> + 'a {
    let spec = spec.clone();
    let mut files = chk_bytes.to_owned();
    files.sort_unstable_by(|a, b| a.0.len().cmp(&b.0.len()).reverse());
    let ref_combinations: Vec<_> = discrete_combos(&spec, extended_search);
    ref_combinations
        .into_iter()
        .map(move |((refin, refout), wordspec)| {
            unresult_iter(reverse(
                &spec,
                files.clone(),
                verbosity,
                refin,
                refout,
                wordspec,
            ))
        })
        .flatten()
        .filter_map(|x| {
            // .transpose() but for Err
            match x {
                Ok(a) => Some(Ok(a)),
                Err(Some(e)) => Some(Err(e)),
                Err(None) => None,
            }
        })
}

/// Parallel version of reverse_crc.
#[cfg(feature = "parallel")]
pub fn reverse_crc_para<'a>(
    spec: &CRCBuilder<u128>,
    chk_bytes: &'a [(&[u8], Vec<u8>)],
    verbosity: u64,
    extended_search: bool,
) -> impl ParallelIterator<Item = Result<CRC<u128>, CheckReverserError>> + 'a {
    let spec = spec.clone();
    let mut files = chk_bytes.to_owned();
    files.sort_unstable_by(|a, b| a.0.len().cmp(&b.0.len()).reverse());
    let ref_combinations: Vec<_> = discrete_combos(&spec, extended_search);
    ref_combinations
        .into_par_iter()
        .map(move |((refin, refout), wordspec)| {
            unresult_iter(reverse(
                &spec,
                files.clone(),
                verbosity,
                refin,
                refout,
                wordspec,
            ))
            .par_bridge()
        })
        .flatten()
        .filter_map(|x| match x {
            Ok(a) => Some(Ok(a)),
            Err(Some(e)) => Some(Err(e)),
            Err(None) => None,
        })
}

// find all combinations of refin, refout and wordspecs using all values when a parameter is not given
fn discrete_combos(
    spec: &CRCBuilder<u128>,
    extended_search: bool,
) -> Vec<((bool, bool), WordSpec)> {
    let width = spec.width.expect("Missing width argument");
    let refins = spec
        .refin
        .map(|x| vec![x])
        .unwrap_or_else(|| vec![false, true]);
    let mut ret = Vec::new();
    for refin in refins {
        let refouts = spec.refout.map(|x| vec![x]).unwrap_or_else(|| {
            if extended_search {
                vec![false, true]
            } else {
                vec![refin]
            }
        });
        let input_endian = spec.input_endian.or_else(|| {
            Some(match refin {
                // big if true since for little, it is equivalent to wordsize=8 which is the same as big
                true => Endian::Big,
                // same for the reverse
                false => Endian::Little,
            })
        });
        let refs = cart_prod(&[refin], &refouts);
        let wordspecs = wordspec_combos(
            spec.wordsize,
            input_endian,
            spec.output_endian,
            width,
            extended_search,
        );
        ret.append(&mut cart_prod(&refs, &wordspecs));
    }
    ret
}

// wrapper to call rev_from_polys with polynomial arguments
fn reverse<'a>(
    spec: &CRCBuilder<u128>,
    chk_bytes: Vec<(&'a [u8], Vec<u8>)>,
    verbosity: u64,
    refin: bool,
    refout: bool,
    wordspec: WordSpec,
) -> Result<impl Iterator<Item = CRC<u128>> + 'a, Option<CheckReverserError>> {
    let width = match spec.width {
        Some(x) => x,
        None => return Err(Some(CheckReverserError::MissingParameter("width"))),
    };
    // check for errors in the parameters
    if 3 > chk_bytes.len()
        + spec.init.is_some() as usize
        + spec.xorout.is_some() as usize
        + spec.poly.is_some() as usize
    {
        return Err(Some(CheckReverserError::MissingParameter(
            "at least 3 parameters/files",
        )));
    }
    if spec.init.is_none()
        && chk_bytes.iter().map(|x| x.0.len()).max() == chk_bytes.iter().map(|x| x.0.len()).min()
    {
        return Err(Some(CheckReverserError::UnsuitableFiles(
            "need at least one file with different length",
        )));
    }
    // convert the files to polynomials
    let maybe_polys: Option<Vec<_>> = chk_bytes
        .iter()
        .map(|(b, c)| {
            if b.len() * 8 % wordspec.wordsize != 0 {
                None
            } else {
                bytes_to_poly(b, c, width as u8, refin, refout, wordspec).map(|p| (p, b.len()))
            }
        })
        .collect();
    let mut polys = match maybe_polys {
        Some(x) => x,
        None => return Err(None),
    };
    // sort by reverse file length
    polys.sort_by(|(fa, la), (fb, lb)| la.cmp(&lb).then(deg(fa).cmp(&deg(fb)).reverse()));
    // convert parameters to polynomials
    let revinfo = RevInfo::from_builder(spec, refin, refout, wordspec);
    rev_from_polys(&revinfo, &polys, verbosity).map(|x| x.iter())
}

struct RevInfo {
    width: usize,
    init: Option<PolyPtr>,
    xorout: Option<PolyPtr>,
    poly: Option<PolyPtr>,
    refin: bool,
    refout: bool,
    wordspec: WordSpec,
}

impl RevInfo {
    // this is responsible for converting integer values to polynomial values
    // and returning a RevInfo that can be used for further reversing
    fn from_builder(
        spec: &CRCBuilder<u128>,
        refin: bool,
        refout: bool,
        wordspec: WordSpec,
    ) -> Self {
        let width = spec.width.unwrap();
        let init = spec.init.map(|i| new_poly(&i.to_le_bytes()));
        let poly = spec.poly.map(|p| {
            let mut p = new_poly(&p.to_le_bytes());
            // add leading coefficient, which is omitted in binary form
            p.pin_mut().add_to(&new_poly_shifted(&[1], width as i64));
            p
        });
        // while init and poly are unaffected by refout, xorout is not
        let xorout = spec
            .xorout
            .map(|x| new_poly(&cond_reverse(width as u8, x, refout).to_le_bytes()));
        RevInfo {
            width,
            init,
            xorout,
            poly,
            refin,
            refout,
            wordspec,
        }
    }
}

struct RevResult {
    polys: Vec<PolyPtr>,
    inits: PrefactorMod,
    xorout: InitPoly,
    width: usize,
    refin: bool,
    refout: bool,
    wordspec: WordSpec,
}

impl RevResult {
    // iterate over all possible parameters
    fn iter(self) -> impl Iterator<Item = CRC<u128>> {
        let RevResult {
            polys,
            inits,
            xorout,
            width,
            refin,
            refout,
            wordspec,
        } = self;
        polys
            .into_iter()
            .map(move |pol| {
                // for each polynomial of degree width, iterate over all solutions of the PrefactorMod
                inits
                    .iter_inits(&pol, &xorout)
                    .map(move |(poly_p, init_p, xorout_p)| {
                        // convert polynomial parameters to a CRC<u128>
                        let poly =
                            poly_to_u128(&add(&poly_p, &new_poly_shifted(&[1], width as i64)));
                        let init = poly_to_u128(&init_p);
                        let xorout = cond_reverse(width as u8, poly_to_u128(&xorout_p), refout);
                        CRC::<u128>::with_options()
                            .width(width)
                            .poly(poly)
                            .init(init)
                            .xorout(xorout)
                            .refin(refin)
                            .refout(refout)
                            .inendian(wordspec.input_endian)
                            .outendian(wordspec.output_endian)
                            .wordsize(wordspec.wordsize)
                            .build()
                            .unwrap()
                    })
            })
            .flatten()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitPlace {
    None,
    Single(usize),
    Pair(usize, usize),
}

type InitPoly = (PolyPtr, InitPlace);

// The parameter reversing for CRC is quite similar and it may be easier to try to understand that implementation first,
// since it uses integers instead of 𝔽₂[X].
//
// If f is a file of length l (in bits) interpreted as a polynomial in 𝔽₂[X], then the crc is just
//  (init*X^l + f*X^width + xorout) % poly
//
// If we have a file with a crc checksum, we can calculate
//  checksum - f*X^width ≡ init*X^l + xorout mod poly
// Note that poly is not yet known, so we can't reduce by poly yet and have a giant degree l polynomial,
// with a file that is a few MB, this is a polynomial whose degree is a few millions, so each operation
// can be expensive.
//
// By using multiple files, we can also cancel xorout and init:
// Given three files of len l₁, l₂, l₃, we have calculated init*X^lₐ + xorout mod poly before, so by subtracting
// the first two, we get `a = init*(X^l₁ + X^l₂) mod poly`. Doing the 2nd and 3rd, we get similarly get `b = init*(X^l₂ + X^l₃) mod poly`.
// For simplicity, let's assume l₁ < l₂ < l₃ (if there are two of the same length, init is already cancelled).
// If we multiply a by (X^(l₃ - l₂) + 1), we get init*(X^(l₃ + l₁ - l₂) + X^l₃ + X^l₂ + X^l₁) mod poly.
// When we multiply b by (X^(l₂ - l₁) + 1), we also get that, so by subtracting both, we get 0 mod poly, meaning that
// poly divides the result, which we can use to determine poly later.
//
// If we have more than three files, we can also get more results, but since poly has to divide all of them, we can gcd them
// together to get a smaller polynomial that is divided by poly.
// If we don't have that, we still know that the highest prime factor of poly that we care about has degree width,
// which we can use to construct a polynomial that only has factors of degree <= width and gcd with that.
//
// One could think that doing a gcd between million-degree polynomials could be very slow.
// And if a naive implementation of multiplication and gcd were used, that would be correct.
// However this program uses two excellent libraries, NTL and gf2x, with which the gcd can be calculated in
// around O(n*log^2(n)) time, thanks to the FFT-based Schönhage-Strassen multiplication and a clever
// gcd implementation called half-gcd.
//
// Now we just assume that the result we got in the previous step is already our poly.
// We can just adjust it to be a divisor of that if we found it to be wrong later.
// With that, we can solve init*(X^l₂ + X^l₁) ≡ x mod poly for init using number theory®
// and from that, we get xorout by subtracting e.g. init*X^l₁.
//
// If our poly is still of degree higher than width, we can then factorize it.
// Note that factoring 𝔽₂[X] polynoials is suprisingly feasable (people have factored
// such polynomials in the degree of millions) and because the factors all have degree <= width,
// due to the way distinct degree factorization works, it should still work quite fast.
// However, by this point poly should be very close in degree to width, so it's not a very big issue anyway.
//
// Using the factorization, we can then iterate over all divisors of degree width.
fn rev_from_polys(
    spec: &RevInfo,
    arg_polys: &[(PolyPtr, usize)],
    verbosity: u64,
) -> Result<RevResult, Option<CheckReverserError>> {
    let log = |s| {
        if verbosity > 0 {
            eprintln!(
                "<crc, refin = {}, refout = {}> {}",
                spec.refin, spec.refout, s
            );
        }
    };
    // InitPlace is essentially a sparse polynomial with at most 2 coefficients being 1
    // note that it has an implied factor of 8, because it uses the byte position instead of bit position
    let mut polys: Vec<_> = arg_polys
        .iter()
        .rev()
        .map(|(p, l)| (copy_poly(p), InitPlace::Single(*l)))
        .collect();
    if let Some(init) = &spec.init {
        log("removing inits");
        remove_inits(&init, &mut polys);
    }
    log("removing xorouts");
    let (polys, mut xorout) = remove_xorouts(&spec.xorout, polys);
    log("finding poly");
    let (polys, mut hull) = find_polyhull(spec, polys, verbosity)?;
    log("finding init and refining poly");
    let init = find_init(&spec.init, hull.pin_mut(), polys);
    let polyhull_factors: Vec<_>;
    if deg(&hull) > 0 {
        xorout.0.pin_mut().rem_to(&hull);
        log("factoring poly");
        polyhull_factors = factor(&hull, if verbosity > 1 { 1 } else { 0 })
            .into_iter()
            .map(|PolyI64Pair { poly, l }| (copy_poly(poly), *l))
            .collect();
    } else {
        log("could not find any fitting factors for poly");
        xorout.0 = new_zero();
        polyhull_factors = vec![];
    }
    log("finding all factor combinations for poly and finishing");
    Ok(RevResult {
        polys: find_prod_comb(spec.width, &polyhull_factors),
        inits: init,
        xorout,
        width: spec.width,
        refin: spec.refin,
        refout: spec.refout,
        wordspec: spec.wordspec,
    })
}

fn remove_inits(init: &Poly, polys: &mut [InitPoly]) {
    for (p, l) in polys {
        match l {
            InitPlace::Single(d) => {
                p.pin_mut().add_to(&shift(init, 8 * *d as i64));
                *l = InitPlace::None;
            }
            // note: this branch shouldn't happen, but it is also no problem if it happens
            InitPlace::None => (),
            // this is not really a problem either, but I will not bother implementing it
            InitPlace::Pair(_, _) => {
                panic!("Internal Error: remove_inits should not receive Pair Inits")
            }
        }
    }
}

fn remove_xorouts(
    maybe_xorout: &Option<PolyPtr>,
    mut polys: Vec<InitPoly>,
) -> (Vec<InitPoly>, InitPoly) {
    let mut ret_vec = Vec::new();
    let mut prev = polys
        .pop()
        .expect("Internal Error: Zero-length vector given to remove_xorouts");
    let xor_ret = match maybe_xorout {
        Some(xorout) => {
            // if we already have xorout, we can subtract it from the files themselves so
            // that we have one more to get parameters from
            ret_vec.push((add(&prev.0, xorout), prev.1));
            (copy_poly(&xorout), InitPlace::None)
        }
        None => (copy_poly(&prev.0), prev.1),
    };
    for (p, l) in polys.into_iter().rev() {
        let appendix = match (maybe_xorout, l != InitPlace::None && l == prev.1) {
            (None, _) | (_, true) => {
                let poly_diff = add(&p, &prev.0);
                let new_init_place = match (prev.1, l) {
                    // no coefficients being one means it is zero and therefore the neutral element
                    (InitPlace::None, other) | (other, InitPlace::None) => other,
                    (InitPlace::Single(l1), InitPlace::Single(l2)) => {
                        if l1 == l2 {
                            // they cancel out
                            InitPlace::None
                        } else {
                            InitPlace::Pair(l1, l2)
                        }
                    }
                    (InitPlace::Pair(_, _), _) | (_, InitPlace::Pair(_, _)) => {
                        panic!("Internal Error: init pair in the input array of remove_xorouts")
                    }
                };
                (poly_diff, new_init_place)
            }
            (Some(xorout), false) => {
                let poly_no_xorout = add(&p, xorout);
                (poly_no_xorout, l)
            }
        };
        ret_vec.push(appendix);
        prev = (p, l);
    }
    (ret_vec, xor_ret)
}

fn find_polyhull(
    spec: &RevInfo,
    polys: Vec<InitPoly>,
    verbosity: u64,
) -> Result<(Vec<InitPoly>, PolyPtr), Option<CheckReverserError>> {
    let log = |s| {
        if verbosity > 1 {
            eprintln!(
                "<crc poly, refin = {}, refout = {}> {}",
                spec.refin, spec.refout, s
            );
        }
    };
    let mut contain_init_vec = Vec::new();
    let mut hull = spec
        .poly
        .as_ref()
        .map(|x| copy_poly(x))
        .unwrap_or_else(new_zero);
    log("gcd'ing same length files together");
    for (p, l) in polys {
        match l {
            InitPlace::None => {
                // if init is multiplied by 0, this is already a multiple of poly so we can gcd it to our estimate
                hull.pin_mut().gcd_to(&p);
            }
            _ => {
                contain_init_vec.push((p, l));
            }
        }
        if deg(&hull) == 0 {
            return Ok((contain_init_vec, hull));
        }
    }

    log("gcd'ing different length files together");
    for ((p, l), (q, m)) in contain_init_vec.iter().zip(contain_init_vec.iter().skip(1)) {
        let power_8n = |n: usize| new_poly_shifted(&[1], 8 * n as i64);
        // this essentially tries to cancel out the init in the checksums
        // if you have a*init and b*init, you can get 0 by calculating b*a*init - a*b*init
        // this is almost done here, except for cancelling unneccessary common X^k between a and b
        let (mut p_fac, mut q_fac) = match (l, m) {
            (InitPlace::None, _) | (_, InitPlace::None) => unreachable!(),
            (InitPlace::Single(d), InitPlace::Single(e)) => {
                let min = d.min(e);
                (power_8n(d - min), power_8n(e - min))
            }
            (InitPlace::Single(d), InitPlace::Pair(e1, e2)) => {
                let min = d.min(e1).min(e2);
                let p_fac = power_8n(d - min);
                let mut q_fac = power_8n(e2 - min);
                q_fac += &power_8n(e1 - min);
                (p_fac, q_fac)
            }
            (InitPlace::Pair(d1, d2), InitPlace::Single(e)) => {
                let min = d1.min(d1).min(e);
                let mut p_fac = power_8n(d2 - min);
                p_fac += &power_8n(d1 - min);
                let q_fac = power_8n(e - min);
                (p_fac, q_fac)
            }
            (InitPlace::Pair(d1, d2), InitPlace::Pair(e1, e2)) => {
                let min = d1.min(d2).min(e1).min(e2);
                let mut p_fac = power_8n(d2 - min);
                p_fac += &power_8n(d1 - min);
                let mut q_fac = power_8n(e2 - min);
                q_fac += &power_8n(e1 - min);
                (p_fac, q_fac)
            }
        };
        p_fac *= q;
        q_fac *= p;
        q_fac += &p_fac;
        // q_fac should now contain no init, so we can gcd it to the hull
        hull.pin_mut().gcd_to(&q_fac);
        if deg(&hull) == 0 {
            return Ok((contain_init_vec, hull));
        }
    }

    if hull.is_zero() {
        // nothing i can do to help now really
        return Err(Some(CheckReverserError::UnsuitableFiles(
            "Duplicate files or unluckiest person alive",
        )));
    }

    log("removing factors with degree*multiplicity > width");
    // You may remember from your course in abstract algebra that in GF(q)[X],
    // the polynomial p_d = X^(q^d) - X contains all primes with degrees dividing d (with multiplicty 1)
    // Here, we multiply all such polynomials for d from 1 to width together.
    // In this product, each prime of degree k has multiplicity floor(width/k) since there are
    // exactly floor(width/k) p_d where k divides d.
    // Now, that polynomial would be quite large, but we only care about the gcd of this polynomial
    // with hull, so we can evaluated this modulo hull.
    let mut cumulative_prod = new_polyrem(&new_poly(&[1]), &hull);
    let x = new_polyrem(&new_poly(&[1 << 1]), &hull);
    let mut x_to_2_to_n = copy_polyrem(&x);
    for i in 0..spec.width {
        if verbosity > 1 {
            eprintln!(
                "<crc poly, refin = {}, refout = {}> step {} of {}",
                spec.refin, spec.refout, i, spec.width
            )
        }
        x_to_2_to_n.pin_mut().sqr();
        let mut fac = copy_polyrem(&x_to_2_to_n);
        fac += &x;
        // (fac = x^(2^n) + x)
        cumulative_prod *= &fac;
    }
    drop(x_to_2_to_n);
    let reduced_prod = cumulative_prod.rep();
    drop(cumulative_prod);
    log("doing final gcd");
    hull.pin_mut().gcd_to(&reduced_prod);
    log("removing trailing zeros");
    // we don't care about the factor X^k in the hull, since crc polys should
    // have the lowest bit set (why would you not??)
    // it is also assumed later that this holds, so this can not just be removed
    for i in 0..=spec.width {
        if hull.coeff(i as i64) {
            hull = shift(&hull, -(i as i64));
            break;
        }
    }
    Ok((contain_init_vec, hull))
}

// we don't actually ever convert the factors represented by a
// InitPlaces struct into a full polynomial, we just evaluate it modulo the hull
// to do this faster, we save X^k mod hull and evaluate them from smallest to largest
// so we can reuse it later
struct MemoPower {
    prev_power: usize,
    prev_ppoly: PolyRemPtr,
    init_fac: PolyPtr,
    hull: PolyPtr,
}
impl MemoPower {
    fn new(hull: &Poly) -> Self {
        MemoPower {
            prev_power: 0,
            prev_ppoly: new_polyrem(&new_poly(&[1]), &hull),
            init_fac: new_zero(),
            hull: copy_poly(hull),
        }
    }
    fn update_init_fac(&mut self, place: &InitPlace) -> &Poly {
        let mut update_power = |&new_level: &usize| {
            if new_level < self.prev_power {
                panic!("Internal Error: Polynomials non-ascending");
            }
            let x = new_polyrem(&new_poly(&[1 << 1]), &self.hull);
            let power_diff = powermod(&x, (new_level - self.prev_power) as i64 * 8);
            self.prev_power = new_level;
            self.prev_ppoly *= &power_diff;
            self.prev_ppoly.rep()
        };
        self.init_fac = match place {
            InitPlace::None => new_zero(),
            InitPlace::Single(d) => update_power(d),
            InitPlace::Pair(d1, d2) => {
                let mut current_power = update_power(d1);
                current_power += &update_power(d2);
                current_power
            }
        };
        &self.init_fac
    }
    fn get_init_fac(&self) -> &Poly {
        &self.init_fac
    }
    fn update_hull(&mut self, hull: &Poly) {
        self.hull = copy_poly(hull);
        self.prev_ppoly = new_polyrem(&self.prev_ppoly.rep(), &hull)
    }
}
// describes a set of solutions for unknown*possible % hull
struct PrefactorMod {
    unknown: PolyPtr,
    possible: PolyPtr,
    hull: PolyPtr,
}

impl PrefactorMod {
    fn empty() -> Self {
        PrefactorMod {
            unknown: new_poly(&[1]),
            possible: new_zero(),
            hull: new_poly(&[1]),
        }
    }
    fn new_init(maybe_init: &Option<PolyPtr>, hull: &Poly) -> Self {
        // if we already have init, we can use that for our solution here, otherwise use the
        // set of all possible solutions
        let (unknown, possible) = match maybe_init {
            None => (copy_poly(hull), new_zero()),
            Some(init) => (new_poly(&[1]), copy_poly(init)),
        };
        PrefactorMod {
            unknown,
            possible,
            hull: copy_poly(&hull),
        }
    }

    fn new_file(
        mut file: PolyPtr,
        power: &mut MemoPower,
        mut hull: Pin<&mut Poly>,
    ) -> Option<Self> {
        file.pin_mut().rem_to(&hull);
        let file_float = gcd(&file, &hull);
        let power_float = gcd(power.get_init_fac(), &hull);
        let common_float = gcd(&power_float, &file_float);
        // power_float has to divide file_float in the hull
        let discrepancy = div(&power_float, &common_float);
        if !discrepancy.eq(&new_poly(&[1])) {
            // if it does not, we change the hull so that it does
            // by replacing the hull_part with the file_part in the hull
            let hull_part = highest_power_gcd(&hull, &discrepancy);
            let file_part = gcd(&file_float, &hull_part);
            // since discrepancy divides file_part and file_part divides hull, resue file_part here
            hull.as_mut().div_to(&hull_part);
            hull.as_mut().mul_to(&file_part);
            if deg(&hull) <= 0 {
                return None;
            }
            power.update_hull(&hull);
        }
        drop(discrepancy);
        drop(power_float);
        drop(file_float);
        // since we only have power*init ≡ mod hull, but want to calculate init,
        // we need to calculate the modular inverse
        let possible = inverse_fixed(file, power.get_init_fac(), &common_float, &hull);
        Some(PrefactorMod {
            unknown: common_float,
            possible,
            hull: copy_poly(&hull),
        })
    }

    fn update_hull(&mut self, hull: &Poly) {
        if self.hull.eq(hull) {
            return;
        }
        self.hull = copy_poly(hull);
        self.unknown.pin_mut().gcd_to(hull);
        self.possible %= &self.valid();
    }

    // merge two different sets of solutions into one where the hull is the gcd of both
    // and all solutions are valid in both
    fn merge(mut self, mut other: Self, mut hull: Pin<&mut Poly>) -> Option<Self> {
        self.update_hull(&hull);
        other.update_hull(&hull);
        self.adjust_compability(&mut other, hull.as_mut());
        if deg(&hull) <= 0 {
            return None;
        }
        let mut self_fac = new_zero();
        let mut other_fac = new_zero();
        let self_valid = self.valid();
        let other_valid = other.valid();
        // this is the chinese remainder theorem for non-coprime ideals
        let common_valid = xgcd(
            self_fac.pin_mut(),
            other_fac.pin_mut(),
            &self_valid,
            &other_valid,
        );
        self_fac *= &self_valid;
        self_fac *= &other.possible;
        other_fac *= &other_valid;
        other_fac *= &self.possible;
        self_fac += &other_fac;
        self_fac /= &common_valid;
        self.possible = self_fac;
        self.unknown = gcd(&self.unknown, &other.unknown);
        Some(self)
    }

    // in order to chinese remainder with a common factor, both polynomials modulo
    // the common factor need to be the same
    // if this is not the case, the hull is adjusted
    fn adjust_compability(&mut self, other: &mut Self, mut hull: Pin<&mut Poly>) {
        let common_valid = gcd(&self.valid(), &other.valid());
        let actual_valid = gcd(&add(&self.possible, &other.possible), &common_valid);
        hull.as_mut().div_to(&common_valid);
        hull.as_mut().mul_to(&actual_valid);
        if deg(&hull) <= 0 {
            return;
        }
        self.update_hull(&hull);
        other.update_hull(&hull);
    }

    fn valid(&self) -> PolyPtr {
        div(&self.hull, &self.unknown)
    }

    fn iter_inits(
        &self,
        red_poly: &Poly,
        xorout: &InitPoly,
    ) -> impl Iterator<Item = (PolyPtr, PolyPtr, PolyPtr)> {
        let red_unknown = gcd(&self.unknown, red_poly);
        let red_valid = div(red_poly, &red_unknown);
        let red_init = rem(&self.possible, &red_valid);
        let mod_valid = new_polyrem(&red_valid, red_poly);
        let mod_init = new_polyrem(&red_init, red_poly);
        let mod_xorout = new_polyrem(&xorout.0, red_poly);
        let x = new_polyrem(&new_poly(&[&1 << 1]), red_poly);
        let mod_power = match xorout.1 {
            InitPlace::None => new_polyrem(&new_zero(), red_poly),
            InitPlace::Single(l) => powermod(&x, 8 * l as i64),
            _ => panic!("Internal Error: Double"),
        };
        let poly_copy = copy_poly(red_poly);
        // iterate over all polynomials p mod red_unknown and calculate possible + valid*p
        (0u128..1 << deg(&red_unknown)).map(move |p| {
            let mut current_init = new_polyrem(&new_poly(&p.to_le_bytes()), &poly_copy);
            current_init *= &mod_valid;
            current_init += &mod_init;
            // also calculate the corresponding xorouts while we're at it
            let mut current_xorout = copy_polyrem(&mod_power);
            current_xorout *= &current_init;
            current_xorout += &mod_xorout;
            (
                copy_poly(&poly_copy),
                current_init.rep(),
                current_xorout.rep(),
            )
        })
    }
}

fn find_init(
    maybe_init: &Option<PolyPtr>,
    mut hull: Pin<&mut Poly>,
    polys: Vec<InitPoly>,
) -> PrefactorMod {
    if deg(&hull) <= 0 {
        return PrefactorMod::empty();
    }
    let mut ret = PrefactorMod::new_init(maybe_init, &hull);
    let mut power = MemoPower::new(&hull);
    for (p, l) in polys {
        power.update_init_fac(&l);
        let file_solutions = PrefactorMod::new_file(p, &mut power, hull.as_mut());
        ret = match file_solutions
            .map(|f| ret.merge(f, hull.as_mut()))
            .flatten()
        {
            Some(valid) => valid,
            None => return PrefactorMod::empty(),
        }
    }
    ret
}

// calculates lim_{n to inf} gcd(a, b^n)
fn highest_power_gcd(a: &Poly, b: &Poly) -> PolyPtr {
    let mut prev = new_poly(&[1]);
    let mut cur = b % a;
    while !cur.eq(&prev) {
        prev = copy_poly(&cur);
        cur.pin_mut().sqr();
        cur.pin_mut().gcd_to(a);
    }
    cur
}

// ntl's modular division doesn't account for common factors between
// the arguments, so this is a version which does
fn inverse_fixed(mut a: PolyPtr, b: &Poly, common: &Poly, hull: &Poly) -> PolyPtr {
    a /= common;
    let mut b = copy_poly(b);
    b /= common;
    let module = div(hull, common);
    if module.eq(&new_poly(&[1])) {
        return new_zero();
    }
    let mut ma = new_polyrem(&a, &module);
    let mb = new_polyrem(&b, &module);
    ma /= &mb;
    ma.rep()
}

fn find_prod_comb(
    width: usize,
    // (degree, multiplicity)
    gens: &[(PolyPtr, i64)],
) -> Vec<PolyPtr> {
    // there's no reason i implemented it like this in particular; the problem is NP complete
    // and i've got no clue how to efficiently solve it anyway and this seemed like a simple solution
    let mut ret: Vec<Vec<PolyPtr>> = (0..=width).map(|_| Vec::new()).collect();
    for (p, l) in gens.iter() {
        // since Poly doesn't implement clone, this will have to do for now
        let retcopy: Vec<Vec<_>> = ret
            .iter()
            .map(|v| v.iter().map(|q| copy_poly(q)).collect())
            .collect();
        let mut q = copy_poly(p);
        for _ in 1..=*l {
            let inc_deg = deg(&q) as usize;
            if inc_deg > width {
                break;
            }
            ret[inc_deg].push(copy_poly(&q));
            for (j, el) in retcopy[0..=width as usize - inc_deg].iter().enumerate() {
                for m in el {
                    ret[j + inc_deg].push(mul(&q, m));
                }
            }
            q *= &p;
        }
    }
    ret.pop().unwrap()
}

fn bytes_to_poly(
    bytes: &[u8],
    checksum: &[u8],
    width: u8,
    refin: bool,
    refout: bool,
    wordspec: WordSpec,
) -> Option<PolyPtr> {
    let new_bytes = reorder_poly_bytes(bytes, refin, wordspec);
    let mut poly = new_poly_shifted(&new_bytes, width as i64);
    let sum = bytes_to_int(&checksum, wordspec.output_endian);
    let check_mask = 1u128.wrapping_shl(width as u32).wrapping_sub(1);
    if (!check_mask & sum) != 0 {
        return None;
    }
    let check = cond_reverse(width, sum, refout);
    poly += &new_poly(&check.to_le_bytes());
    Some(poly)
}

fn reorder_poly_bytes(bytes: &[u8], refin: bool, wordspec: WordSpec) -> Vec<u8> {
    wordspec
        .iter_words(bytes)
        .rev()
        .map(|n| {
            let n_ref = if refin {
                n.reverse_bits() >> (64 - wordspec.wordsize)
            } else {
                n
            };
            int_to_bytes(n_ref, Endian::Little, wordspec.wordsize)
        })
        .flatten()
        .collect()
}

fn cond_reverse(width: u8, value: u128, refout: bool) -> u128 {
    if refout {
        value.reverse_bits() >> (128 - width)
    } else {
        value
    }
}

fn poly_to_u128(poly: &Poly) -> u128 {
    u128::from_be_bytes(
        poly.to_bytes(16)
            .as_ref()
            .unwrap()
            .as_slice()
            .try_into()
            .unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::{
        crc::{CRCBuilder, CRC},
        tests::ReverseFileSet,
    };
    use quickcheck::{Arbitrary, TestResult, Gen};
    impl Arbitrary for CRCBuilder<u128> {
        fn arbitrary(g: &mut Gen) -> Self {
            let width = (u8::arbitrary(g) % 128) + 1;
            let poly;
            let init;
            let xorout;
            if width != 128 {
                poly = (u128::arbitrary(g) % (1 << width)) | 1;
                init = u128::arbitrary(g) % (1 << width);
                xorout = u128::arbitrary(g) % (1 << width);
            }
            else {
                poly = u128::arbitrary(g) | 1;
                init = u128::arbitrary(g);
                xorout = u128::arbitrary(g);
            };
            let mut builder = CRC::<u128>::with_options();
            builder
                .width(usize::from(width))
                .poly(poly)
                .init(init)
                .xorout(xorout);
            let combos = discrete_combos(&builder, false);
            let ((refin, refout), wordspec) = *g.choose(&combos).unwrap();
            builder
                .refout(refout)
                .refin(refin)
                .wordsize(wordspec.wordsize)
                .outendian(wordspec.output_endian)
                .inendian(wordspec.input_endian);
            builder
        }
    }
    #[quickcheck]
    fn qc_crc_rev(
        files: ReverseFileSet,
        crc_build: CRCBuilder<u128>,
        known: (bool, bool, bool, bool, bool),
        wordspec_known: (bool, bool),
    ) -> TestResult {
        let crc = crc_build.build().unwrap();
        let mut naive = CRC::<u128>::with_options();
        naive.width(crc_build.width.unwrap());
        if known.0 {
            naive.poly(crc_build.poly.unwrap());
        }
        if known.1 {
            naive.init(crc_build.init.unwrap());
        }
        if known.2 {
            naive.xorout(crc_build.xorout.unwrap());
        }
        if known.3 {
            naive.refin(crc_build.refin.unwrap());
        }
        if known.4 {
            naive.refout(crc_build.refout.unwrap());
        }
        if wordspec_known.0 {
            naive.wordsize(crc_build.wordsize.unwrap());
        }
        if wordspec_known.1 {
            naive.outendian(crc_build.output_endian.unwrap());
        }
        let chk_files = files.with_checksums(&crc);
        let reverser = reverse_crc(&naive, &chk_files, 0, false);
        files.check_matching(&crc, reverser)
    }
    #[test]
    fn test_crc32() {
        let crc = CRC::<u128>::with_options()
            .poly(0x04c11db7)
            .width(32)
            .init(0xffffffff)
            .xorout(0xffffffff)
            .refout(true)
            .refin(true)
            .build()
            .unwrap();
        let files = ReverseFileSet(vec![
            vec![0x12u8, 0x34u8, 0x56u8],
            vec![0x67u8, 0x41u8, 0xffu8],
            vec![0x15u8, 0x56u8, 0x76u8, 0x1fu8],
            vec![0x14u8, 0x62u8, 0x51u8, 0xa4u8, 0xd3u8],
        ]);
        let chk_files: Vec<_> = files.with_checksums(&crc);
        let mut crc_naive = CRC::<u128>::with_options();
        crc_naive.width(32).refin(true).refout(true);
        let reverser = reverse_crc(&crc_naive, &chk_files, 0, false);
        assert!(!files.check_matching(&crc, reverser).is_failure())
    }
    #[test]
    fn test_crc16() {
        let crc = CRC::<u128>::with_options()
            .poly(0x8005)
            .width(16)
            .refin(true)
            .refout(true)
            .build()
            .unwrap();
        let files = ReverseFileSet(vec![
            vec![0x12u8, 0x34u8, 0x56u8],
            vec![0x67u8, 0x41u8, 0xffu8],
            vec![0x15u8, 0x56u8, 0x76u8, 0x1fu8],
            vec![0x14u8, 0x62u8, 0x51u8, 0xa4u8, 0xd3u8],
        ]);
        let chk_files = files.with_checksums(&crc);
        let mut crc_naive = CRC::<u128>::with_options();
        crc_naive.width(16).refin(true).refout(true);
        let reverser = reverse_crc(&crc_naive, &chk_files, 0, false);
        assert!(!files.check_matching(&crc, reverser).is_failure())
    }
    #[test]
    fn error1() {
        let crc = CRC::with_options()
            .width(32)
            .poly(0x04c11db7)
            .init(0xffffffff)
            .xorout(0xffffffff)
            .build()
            .unwrap();
        let files = ReverseFileSet(vec![
            vec![63, 6, 30, 42, 34, 48, 87, 50, 69, 26, 23, 98, 49, 3, 99, 86],
            vec![13, 13, 41, 51, 13, 62, 99, 11],
            vec![],
        ]);
        let chk_files = files.with_checksums(&crc);
        let mut crc_naive = CRC::<u128>::with_options();
        crc_naive.width(32);
        let reverser = reverse_crc(&crc_naive, &chk_files, 0, false);
        assert!(!files.check_matching(&crc, reverser).is_failure())
    }
    #[test]
    fn error2() {
        let crc = CRC::with_options()
            .width(50)
            .poly(0x2f)
            .init(0x5)
            .xorout(0x58)
            .refout(true)
            .build()
            .unwrap();
        let files = ReverseFileSet(vec![
            vec![
                8, 86, 64, 28, 64, 25, 99, 40, 15, 92, 15, 66, 42, 12, 66, 80,
            ],
            vec![73, 68, 27, 35, 16, 69, 9, 24],
            vec![51, 37, 13, 18, 50, 23, 49, 9],
            vec![49, 81, 64, 26, 24, 45, 36, 62],
            vec![18, 42, 24, 61, 15, 22, 41, 28],
            vec![16, 53, 20, 62, 50, 48, 25, 35],
            vec![4, 73, 5, 16, 37, 83, 40, 91],
        ]);
        let chk_files = files.with_checksums(&crc);
        let mut crc_naive = CRC::<u128>::with_options();
        crc_naive.width(50).xorout(0x58);
        let reverser = reverse_crc(&crc_naive, &chk_files, 0, true);
        assert!(!files.check_matching(&crc, reverser).is_failure())
    }
    #[test]
    fn error3() {
        let crc = CRC::with_options()
            .width(17)
            .poly(0x1)
            .init(0xb)
            .xorout(0x56)
            .refout(true)
            .build()
            .unwrap();
        // note that this is randomly generated, which is why there are so many bytes since i
        // didn't want to go to the trouble of reducing it
        let files = ReverseFileSet(vec![
            vec![
                70, 49, 21, 1, 7, 18, 4, 38, 11, 47, 30, 90, 49, 69, 17, 26, 48, 51, 91, 81, 31,
                61, 35, 3, 8, 94, 90, 10, 28, 30, 22, 31, 42, 87, 67, 28, 69, 49, 85, 95, 84, 81,
                30, 40, 81, 99, 92, 53, 95, 87, 17, 2, 56, 43, 33, 2, 5, 99, 32, 74, 68, 59, 47,
                88, 70, 42, 34, 68, 49, 41, 13, 84, 67, 5, 45, 31, 94, 80, 92, 36, 73, 4, 29, 36,
                14, 95, 79, 71, 93, 14, 39, 64, 15, 65, 56, 72, 0, 93, 35, 27, 66, 94, 21, 84, 72,
                69, 87, 44, 28, 75, 76, 25, 61, 41, 50, 44, 7, 42, 56, 63, 11, 20, 48, 55, 44, 44,
                45, 58, 77, 1, 83, 87, 30, 65, 27, 53, 42, 9, 39, 79, 47, 29, 78, 96, 9, 65, 98,
                55, 95, 85, 0, 55, 69, 33, 63, 5, 88, 13, 61, 21, 47, 90, 16, 65, 73, 38, 95, 45,
                95, 15, 54, 67, 11, 98, 50, 80, 95, 69, 88, 33, 24, 1, 8, 73, 39, 74, 59, 51, 7,
                64, 51, 58, 80, 99, 1, 68, 70, 45, 77, 5, 6, 4, 32, 33, 93, 95, 13, 98, 76, 87, 96,
                99, 74, 18, 23, 23, 4, 51, 47, 76, 14, 62, 91, 18, 68, 64, 38, 71, 15, 35, 51, 46,
                18, 53, 38, 88, 76, 42, 28, 25, 86, 17, 10, 73, 74, 45, 98, 20, 72, 12, 64, 24, 15,
                15, 87, 75, 12, 29, 16, 11, 16, 11, 16, 94, 41, 29, 97, 32, 88, 27, 96, 78, 70, 54,
                49, 82, 73, 0, 19, 15, 19, 93, 36, 74, 90, 33, 94, 70, 72, 6, 73, 54, 50, 4, 9, 0,
                81, 44, 72, 76, 29, 94, 85, 10, 43, 88, 15, 53, 17, 31, 53, 0, 3, 81, 84, 72, 29,
                74, 66, 71, 97, 66, 44, 81, 26, 57, 8, 49, 24, 18, 9, 89, 57, 37, 91, 86, 62, 16,
                41, 0, 88, 92, 54, 22, 8, 12, 12, 73, 38, 25, 67, 73, 87, 82, 48, 98, 43, 10, 65,
                66, 88, 76, 70, 3, 62, 22, 89, 99, 78, 79, 68, 23, 60, 60, 58, 14, 94, 52, 62, 75,
                10, 79, 89, 81, 81, 65, 71, 3, 74, 43, 14, 25, 21, 49, 1, 33, 64, 28, 72, 88, 56,
                46, 95, 93, 71, 24, 85, 2, 52, 29, 81, 0, 71, 25, 94, 91, 5, 77, 20, 81, 37, 85,
                74, 57, 96, 82, 17, 27, 81, 46, 96, 1, 19, 0, 51, 47, 63, 49, 65, 97, 12, 93, 78,
                84, 26, 26, 4, 73, 16, 43, 60, 95, 69, 29, 94, 97, 52, 83, 2, 10, 83, 80, 30, 91,
                3, 33, 72, 5, 44, 43, 44, 52, 34, 66, 99, 2, 15, 65, 52, 27, 39, 32, 58, 76, 5, 54,
                92, 27, 32, 52, 1, 10, 14, 70, 52, 74, 36, 12, 94, 34, 38, 46, 15, 91, 62, 84, 86,
                43, 83, 3, 56, 51, 96, 93, 14, 0, 34, 46, 79, 64, 50, 62, 25, 43, 47, 32, 36, 33,
                29, 25, 76, 39, 56, 27, 66, 17, 72, 84, 68, 56, 72, 10, 55, 37, 79, 54, 85, 48, 5,
                64, 14, 28, 40, 94, 30, 52, 43, 77, 47, 89, 30, 14, 43, 35, 87, 24, 14, 35,
            ],
            vec![
                2, 12, 30, 87, 4, 35, 39, 66, 11, 22, 93, 85, 87, 23, 75, 27, 91, 65, 26, 6, 29,
                56, 53, 56, 76, 30, 26, 68, 28, 60, 39, 35, 35, 31, 61, 72, 27, 44, 91, 77, 13, 47,
                85, 69, 91, 30, 8, 57, 59, 91, 41, 97, 34, 46, 57, 5, 94, 79, 80, 6, 97, 23, 53,
                64, 6, 6, 39, 35, 38, 98, 71, 30, 59, 15, 71, 69, 96, 8, 97, 41, 26, 76, 44, 61,
                79, 11, 57, 19, 83, 47, 15, 51, 14, 49, 99, 30, 3, 85, 45, 74, 85, 34, 68, 86, 90,
                86, 59, 29, 8, 8, 31, 13, 15, 40, 49, 2, 42, 71, 33, 87, 55, 63, 85, 74, 68, 71,
                18, 79, 86, 15, 13, 98, 89, 64, 94, 62, 3, 59, 34, 6, 42, 65, 25, 76, 22, 43, 69,
                82, 17, 48, 46, 87, 68, 43, 46, 0, 35, 61, 52, 22, 79, 35, 6, 66, 68, 4, 97, 16,
                76, 26, 49, 36, 90, 2, 85, 5, 46, 86, 28, 71, 4, 22, 97, 67, 7, 10, 77, 89, 5, 8,
                26, 17, 25, 6, 12, 3, 22, 50, 93, 35, 80, 61, 43, 62, 45, 19, 27, 44, 42, 64, 71,
                24, 90, 98, 63, 31, 96, 97, 45, 25, 74, 37, 64, 25, 38, 32, 6, 3, 91, 67, 31, 92,
                49, 41, 99, 29, 32, 78, 49, 26, 16, 81, 15, 27, 93, 4, 94, 1, 8, 9, 2, 10, 6, 55,
                11, 77, 51, 10, 53, 47, 91, 14, 62, 9, 87, 57, 83, 92, 3, 13, 54, 23, 77, 58, 73,
                2, 37, 88, 33, 88, 76, 0, 56, 1, 96, 74, 8, 75, 21, 84, 45, 19, 69, 60, 80, 16, 3,
                68, 46, 95, 38, 18, 18, 6, 18, 49, 84, 7, 64, 12, 90, 62, 90, 15, 16, 15, 20, 45,
                54, 13, 54, 13, 74, 88, 22, 28, 99, 47, 13, 52, 61, 36, 66, 93, 43, 17, 33, 78, 38,
                56, 87, 54, 88, 0, 58, 98, 35, 28, 19, 33, 93, 19, 68, 5, 65, 87, 30, 2, 34, 48,
                15, 42, 99, 48, 29, 13, 28, 46, 83, 96, 74, 39, 14, 83, 96, 83, 93, 26, 5, 2, 51,
                44, 10, 72, 24, 37, 7, 84, 74, 83, 50, 31, 41, 76, 73, 82, 9, 33, 39, 85, 11, 92,
                33, 36, 72, 75, 16, 22, 22, 54, 8, 57, 59, 60, 61, 66, 76, 77, 7, 66, 17, 30, 60,
                41, 87, 59, 50, 69, 3, 15, 90, 62,
            ],
            vec![
                98, 13, 33, 21, 95, 33, 29, 93, 66, 85, 9, 11, 64, 78, 41, 33, 96, 90, 94, 97, 7,
                36, 36, 67, 26, 48, 96, 96, 55, 24, 27, 45, 84, 44, 86, 69, 38, 80, 95, 18, 41, 72,
                56, 93, 19, 7, 66, 17, 20, 49, 85, 27, 13, 68, 49, 67, 64, 35, 56, 45, 77, 93, 52,
                31, 93, 35, 91, 11, 45, 44, 90, 35, 90, 94, 90, 86, 17, 51, 67, 11, 69, 21, 83, 19,
                17, 62, 0, 63, 1, 7, 97, 24, 88, 33, 12, 4, 16, 70, 69, 37, 77, 32, 66, 48, 8, 21,
                47, 14, 83, 57, 27, 86, 6, 75, 65, 26, 96, 1, 94, 72, 51, 15, 40, 86, 89, 32, 36,
                52, 58, 52, 58, 52, 39, 10, 18, 81, 56, 80, 49, 4, 59, 66, 53, 20, 71, 87, 25, 87,
                34, 2, 37, 11, 67, 80, 96, 2, 13, 80, 75, 17, 9, 48, 70, 19, 24, 61, 23, 31, 7, 6,
                42, 12, 63, 71, 21, 40,
            ],
        ]);
        let chk_files = files.with_checksums(&crc);
        let mut crc_naive = CRC::<u128>::with_options();
        crc_naive.width(17);
        let reverser = reverse_crc(&crc_naive, &chk_files, 0, true);
        assert!(!files.check_matching(&crc, reverser).is_failure())
    }
}

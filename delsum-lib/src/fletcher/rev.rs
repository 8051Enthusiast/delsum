//! This module contains the function(s) for reversing the parameters for a fletcher algorithm with given bytes and checksums.
//!
//! Generally, to find out the parameters, the checksums and their width are needed, and 3 of the following (with at least one file):
//! * value of `init`
//! * value of `addout`
//! * value of `modulus`
//! * a file with checksum
//! * a different file with checksum
//! * yet another different file checksum
//!
//! If `init` is not known, it is neccessary to know two checksums of files with different lengths.
//! In case only checksums of files with a set length are required, setting `init = 0` is sufficient.
//!
//! It is probable that giving just two files + checksum might already be enough, but there will
//! probably also be many some false positives.
use super::{Fletcher, FletcherBuilder};
use crate::checksum::{CheckReverserError, Checksum, filter_opt_err};
use crate::divisors::divisors_range;
use crate::endian::{Endian, SignedInt, WordSpec, wordspec_combos};
use crate::utils::{cart_prod, unresult_iter};
use num_bigint::BigInt;
use num_traits::{One, Signed, Zero, one, zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::convert::{TryFrom, TryInto};
use std::iter::Iterator;
/// Find the parameters of a fletcher algorithm.
///
/// `spec` contains the known parameters of the algorithm (by setting the corresponding values in the builder).
/// `chk_bytes` are pairs of files and their checksums.
/// `verbosity` makes the function output what it is doing.
///
/// The `width` parameter of the builder has to be set.
pub fn reverse_fletcher<'a>(
    spec: &FletcherBuilder<u64>,
    chk_bytes: &[(&'a [u8], Vec<u8>)],
    verbosity: u64,
    extended_search: bool,
) -> impl Iterator<Item = Result<Fletcher<u64>, CheckReverserError>> + use<'a> {
    let spec = spec.clone();
    let mut files = chk_bytes.to_owned();
    files.sort_unstable_by(|a, b| a.0.len().cmp(&b.0.len()).reverse());
    discrete_combos(spec.clone(), extended_search)
        .into_iter()
        .flat_map(move |x| {
            filter_opt_err(unresult_iter(reverse_discrete(
                spec.clone(),
                files.clone(),
                x,
                verbosity,
            )))
        })
}

/// Parallel version of reverse_fletcher.
///
/// It is parallel in the sense that there are two threads, for swap=false and swap=true, if it is not given,
/// so don't expect too much speedup.
#[cfg(feature = "parallel")]
pub fn reverse_fletcher_para<'a>(
    spec: &FletcherBuilder<u64>,
    chk_bytes: &[(&'a [u8], Vec<u8>)],
    verbosity: u64,
    extended_search: bool,
) -> impl ParallelIterator<Item = Result<Fletcher<u64>, CheckReverserError>> + use<'a> {
    let spec = spec.clone();
    let mut files = chk_bytes.to_owned();
    files.sort_unstable_by(|a, b| a.0.len().cmp(&b.0.len()).reverse());
    discrete_combos(spec.clone(), extended_search)
        .into_par_iter()
        .map(move |x| {
            filter_opt_err(unresult_iter(reverse_discrete(
                spec.clone(),
                files.clone(),
                x,
                verbosity,
            )))
            .par_bridge()
        })
        .flatten()
}

fn discrete_combos(spec: FletcherBuilder<u64>, extended_search: bool) -> Vec<(bool, WordSpec)> {
    let swap = spec
        .swap
        .map(|x| vec![x])
        .unwrap_or_else(|| vec![false, true]);
    let wordspecs = wordspec_combos(
        spec.wordsize,
        spec.input_endian,
        spec.output_endian,
        spec.signedness,
        spec.width.unwrap(),
        extended_search,
    );
    cart_prod(&swap, &wordspecs)
}

fn reverse_discrete(
    spec: FletcherBuilder<u64>,
    chk_bytes: Vec<(&[u8], Vec<u8>)>,
    loop_element: (bool, WordSpec),
    verbosity: u64,
) -> Result<impl Iterator<Item = Fletcher<u64>>, Option<CheckReverserError>> {
    let width = spec
        .width
        .ok_or(CheckReverserError::MissingParameter("width"))?;
    let wordspec = loop_element.1;
    let chk_words: Vec<_> = chk_bytes
        .iter()
        .map(|(f, c)| (wordspec.iter_words(f), c.clone()))
        .collect();
    let rev = RevSpec {
        width,
        addout: spec.addout,
        init: spec.init,
        modulus: spec.modulus,
        swap: loop_element.0,
        wordspec: loop_element.1,
    };
    reverse(rev, chk_words, verbosity).map(|x| x.iter())
}

struct RevSpec {
    width: usize,
    addout: Option<u128>,
    init: Option<u64>,
    modulus: Option<u64>,
    swap: bool,
    wordspec: WordSpec,
}

// contains the information needed for iterating over the possible algorithms
// `inits` is a solution set of the form `a*x ≡ y mod m`, and init still has to
// be subtracted from addout1 and addout2
#[derive(Debug, Clone)]
struct RevResult {
    inits: PrefactorMod,
    addout1: BigInt,
    addout2: (BigInt, usize),
    moduli: Vec<BigInt>,
    width: usize,
    swap: bool,
    wordspec: WordSpec,
}

impl RevResult {
    fn iter(self) -> impl Iterator<Item = Fletcher<u64>> {
        let RevResult {
            inits,
            addout1,
            addout2,
            moduli,
            width,
            swap,
            wordspec,
        } = self;
        moduli.into_iter().flat_map(move |m| {
            let modulus = if m.is_zero() {
                0u64
            } else {
                (&m).try_into().unwrap()
            };
            inits.iter(&addout1, &addout2, &m).map(move |(i, s1, s2)| {
                let addout = glue_sum(s1, s2, width, swap);
                Fletcher::with_options()
                    .addout(addout)
                    .init(i)
                    .modulus(modulus)
                    .width(width)
                    .swap(swap)
                    .inendian(wordspec.input_endian)
                    .outendian(wordspec.output_endian)
                    .wordsize(wordspec.wordsize)
                    .signedness(wordspec.signedness)
                    .build()
                    .unwrap()
            })
        })
    }
}

// For understanding the reversing process, one must first look what the sum looks like.
// If one has a file [a, b, c, d, e] of 5 bytes, then the regular checksum will be in the form
//      (init + a + b + c + d + e + addout1) mod m
// and the cumulative one will be
//      (5*init + 5*a + 4*b + 3*c + 2*d * e + addout2) mod m
// Because we also know the file, we can subtract a + b + c + d + e or 5a + 4b + 3c + 2d + 1e and get
//      (init + addout1) mod m
//      (5*init + addout2) mod m
// Note that the notation `mod` here does not mean that the result is 0 <= x < m, just that the difference
// to the unreduced form is a multiple of `m`.
// So we can just subtract these values without knowing the value of m.
//
// We will ignore the regular sum for now because that's the easy part and is the same as in modsum with addout = addout1 + init.
//
// Now assume that we do not know addout2 or init, but have three files of lengths 2, 3 and 5.
// We can therefore get the values (5*init + addout2) mod m, (3*init + addout2) mod m and (2*init + addout2) mod m.
// If we take the differences of the first two and the last two, we get x = 2*init mod m and y = init mod m.
//
// So now we can calculate x - 2*y to get a result that is (0 mod m), which means that the result is divisible by m.
// We can just assume for now that we actually found m and adjust m later if we found that to be wrong.
// If m is zero, that is bad luck and we return an error that the files were likely too similar.
//
// Now that we have a candidate for m, we can start looking for init and addout.
// Say we found m = 4 and x = 2*init = 2 mod 4.
// From that, we can infer that init = 1 mod 4 or init = 3 mod 4 (see PrefactorMod for how exactly that is done).
// Finding out addout2 is now as easy as subtracting 5*init from (5*init + addout2) mod m.
fn reverse(
    spec: RevSpec,
    chk_bytes: Vec<(impl Iterator<Item = SignedInt<u64>>, Vec<u8>)>,
    verbosity: u64,
) -> Result<RevResult, Option<CheckReverserError>> {
    let log = |s| {
        if verbosity > 0 {
            eprintln!("<fletcher> {}", s);
        }
    };
    let width = spec.width;
    let swap = spec.swap;
    let wordspec = spec.wordspec;
    let Some((min, max, mut cumusums, regsums)) =
        summarize(chk_bytes, width, swap, spec.wordspec.output_endian)
    else {
        return Err(None);
    };
    log("finding parameters of lower sum");
    // finding the parameters of the lower sum is pretty much a separate problem already
    // handled in modsum, so we delegate to that
    let modulus = spec.modulus.unwrap_or(0) as u128;
    let (modulus, addout1) = find_regular_sum(&spec, &regsums, modulus);
    let mut modulus = BigInt::from(modulus);
    let mut addout1 = BigInt::from(addout1);
    // here, we take the the checksums and remove the cumulative sum sums from them
    // the second value of each value is supposed to be the multiplicity of init in the sum
    if let Some(init) = spec.init {
        log("removing inits from upper sum");
        // if we have the parameter init already given, we can remove
        // it from the sums, so that the cumusums are now just (x, 0) pairs.
        remove_init(&mut cumusums, &BigInt::from(init));
    }
    log("removing upper sum addout");
    // take the difference between neighboring files if addout2 is not given, to remove the constant addout from the sums
    // if we already have addout2 given, we don't take the difference between two files, but between each file and given addout2
    let (red_files, mut addout2) = remove_addout2(&spec, cumusums, &modulus);
    log("refining the modulus");
    // here we try to find `modulus` and reduce the sums by modulus
    let boneless_files = refine_modulus(&mut modulus, red_files);
    if modulus.is_zero() {
        return Err(Some(CheckReverserError::UnsuitableFiles(
            "too short or too similar or too few files given",
        )));
    }
    log("attempting to find init");
    // now that we have the modulus, we can try to find init and reduce `modulus` some more
    let inits = find_init(&spec.init.map(BigInt::from), &mut modulus, boneless_files);
    let modulus = inits.modulus.clone();
    addout1 = mod_red(&addout1, &modulus);
    addout2.0 = mod_red(&addout2.0, &modulus);
    // if we have checksums of width 7 with values 0x24, 0x51 and 0x64 we know that `modulus` has to be between 2^7 = 0x80 and 0x64
    log("try to find all possible modulus values");
    // therefore we try to find all divisors of modulus in that range

    let moduli = match spec.modulus {
        None => divisors_range(modulus.try_into().unwrap(), min, max)
            .into_iter()
            .map(BigInt::from)
            .collect(),
        Some(m) => {
            if BigInt::from(m) == modulus && m as u128 >= min && m as u128 <= max {
                vec![modulus]
            } else {
                Vec::new()
            }
        }
    };
    Ok(RevResult {
        inits,
        addout1,
        addout2,
        moduli,
        width,
        swap,
        wordspec,
    })
}

fn split_sum(sum: u128, width: usize, swap: bool) -> (u64, u64) {
    let mut lower = sum & ((1 << (width / 2)) - 1);
    let mut upper = sum >> (width / 2);
    if swap {
        std::mem::swap(&mut lower, &mut upper);
    }
    (lower as u64, upper as u64)
}

fn glue_sum(mut s1: u64, mut s2: u64, width: usize, swap: bool) -> u128 {
    if swap {
        std::mem::swap(&mut s1, &mut s2);
    }
    (s1 as u128) | ((s2 as u128) << (width / 2))
}

fn summarize(
    chks: Vec<(impl Iterator<Item = SignedInt<u64>>, Vec<u8>)>,
    width: usize,
    swap: bool,
    out_endian: Endian,
) -> Option<(u128, u128, Vec<(BigInt, usize)>, Vec<i128>)> {
    let mut regsums = Vec::new();
    let mut cumusums = Vec::new();
    let mut min = 2;
    for (words, chk) in chks {
        let chk = Checksum::from_bytes(&chk, out_endian, width)?;
        let (s1, s2) = split_sum(chk, width, swap);
        min = min.max(s1 as u128 + 1);
        min = min.max(s2 as u128 + 1);
        let mut current_sum: BigInt = zero();
        let mut cumusum: BigInt = zero();
        let mut size = 0usize;
        for word in words {
            size += 1;
            if word.negative {
                current_sum -= BigInt::from(word.value);
            } else {
                current_sum += BigInt::from(word.value);
            }
            cumusum += &current_sum;
        }
        let (check1, check2) = split_sum(chk, width, swap);
        regsums.push(
            i128::try_from(current_sum).expect("Unexpected overflow in sum") - check1 as i128,
        );
        cumusums.push((BigInt::from(check2) - cumusum, size));
    }
    let max = 1 << (width / 2);
    Some((min, max, cumusums, regsums))
}

fn find_regular_sum(spec: &RevSpec, sums: &[i128], mut modulus: u128) -> (u128, i128) {
    let width = spec.width;
    // init is here actually addout1 + init, which we can only know if we have both values
    let maybe_init = spec.addout.and_then(|x| {
        spec.init
            .map(|y| y as i128 + split_sum(x, width, spec.swap).0 as i128)
    });
    // delegate to the corresponding modsum function
    let sum1_addout = super::super::modsum::find_largest_mod(sums, maybe_init, &mut modulus);
    (modulus, sum1_addout)
}

fn remove_init(sums: &mut [(BigInt, usize)], init: &BigInt) {
    for (s, l) in sums.iter_mut() {
        *s -= init * BigInt::from(*l);
        *l = 0;
    }
}

fn remove_addout2(
    spec: &RevSpec,
    mut sums: Vec<(BigInt, usize)>,
    modulus: &BigInt,
) -> (Vec<(BigInt, usize)>, (BigInt, usize)) {
    let width = spec.width;
    let swap = spec.swap;
    let maybe_addout = spec
        .addout
        .map(|x| BigInt::from(split_sum(x, width, swap).1));
    let mut ret_vec = Vec::new();
    let mut prev = sums
        .pop()
        .expect("Internal Error: Zero-length vector given to remove_addout2");
    // note: this variable is actually (x,y) where x = addout2 + y*init because we do not know
    // init yet
    let addout2 = match &maybe_addout {
        Some(addout) => {
            // if we already know addout, we can use the first file for determining
            // the modulus or init better
            ret_vec.push((mod_red(&(&prev.0 - addout), modulus), prev.1));
            (addout.clone(), 0)
        }
        None => prev.clone(),
    };
    // note that we reverse the order of the vector here
    for (p, l) in sums.into_iter().rev() {
        let appendix = match (&maybe_addout, l != 0 && l == prev.1) {
            // if we know addout, but the two files have the same length, we still
            // want to calculate the difference between the two files, as it will
            // make it easier to determine the modulus
            (None, _) | (_, true) => (mod_red(&(&p - prev.0), modulus), l - prev.1),
            // but if they're not the same size, we just subtract addout
            (Some(addout), false) => (mod_red(&(&p - addout), modulus), l),
        };
        ret_vec.push(appendix);
        prev = (p, l);
    }
    (ret_vec, addout2)
}

fn refine_modulus(modulus: &mut BigInt, sums: Vec<(BigInt, usize)>) -> Vec<(BigInt, usize)> {
    let mut non_zero = Vec::new();
    for (s, l) in sums {
        if l != 0 {
            non_zero.push((s, l));
            continue;
        }
        // if we have l == 0, they don't contain init, and because they also don't contain
        // addout, they have to be divisible by modulus
        *modulus = gcd(modulus, &s);
    }
    for ((sa, la), (sb, lb)) in non_zero.iter().zip(non_zero.iter().skip(1)) {
        // for x = a*init mod m, y = b*init mod m we can get 0 mod m by calculating
        // (b * x + a * y)/gcd(a, b)
        let bla = BigInt::from(*la);
        let blb = BigInt::from(*lb);
        let common = gcd(&bla, &blb);
        let mul_sa = sa * blb;
        let mul_sb = sb * bla;
        *modulus = gcd(modulus, &((mul_sa - mul_sb) / common));
    }
    non_zero
        .iter()
        .map(|(s, l)| (mod_red(s, modulus), *l))
        .collect()
}

// modular reduction, because % is just wrong
fn mod_red(n: &BigInt, modulus: &BigInt) -> BigInt {
    if modulus.is_zero() {
        // yes, n modulo 0 is n and i will die on this hill
        n.clone()
    } else {
        let k = n % modulus;
        if k < zero() { modulus + k } else { k }
    }
}
fn find_init(
    maybe_init: &Option<BigInt>,
    modulus: &mut BigInt,
    sums: Vec<(BigInt, usize)>,
) -> PrefactorMod {
    if modulus.is_one() {
        return PrefactorMod::empty();
    };
    let mut ret = PrefactorMod::new_init(maybe_init, modulus);
    for (p, l) in sums {
        // get the set of inits that solve l*init ≡ p mod modulus
        let file_solutions = PrefactorMod::from_sum(&p, l, modulus);
        // merge the solutions with the other solutions
        ret = match file_solutions.map(|f| ret.merge(f)) {
            Some(valid) => valid,
            None => return PrefactorMod::empty(),
        }
    }
    ret
}
// describes a set of solutions for unknown*possible % modulus
// the `unknown` parameter divides modulus and captures the fact that there
// can be multiple solutions for unknown*possible mod `modulus` because we only
// know possible modulo (modulus / unknown)
#[derive(Clone, Debug)]
struct PrefactorMod {
    unknown: BigInt,
    possible: BigInt,
    modulus: BigInt,
}

impl PrefactorMod {
    fn empty() -> PrefactorMod {
        PrefactorMod {
            modulus: one(),
            unknown: one(),
            possible: zero(),
        }
    }
    fn from_sum(sum: &BigInt, power: usize, modulus: &mut BigInt) -> Option<PrefactorMod> {
        let bpower = BigInt::from(power);
        // this basically calculates sum*power^-1, but adjusting modulus if there are no solutions
        // and keeping in mind that there can be multiple solutions (which the unknown var keeps track of)
        let (possible, unknown) = partial_mod_div(sum, &bpower, modulus);
        if modulus.is_one() {
            return None;
        }
        Some(PrefactorMod {
            unknown,
            possible,
            modulus: modulus.clone(),
        })
    }
    fn new_init(maybe_init: &Option<BigInt>, modulus: &BigInt) -> Self {
        let (unknown, possible) = match maybe_init {
            None => (modulus.clone(), zero()),
            Some(init) => (one(), init.clone()),
        };
        PrefactorMod {
            unknown,
            possible,
            modulus: modulus.clone(),
        }
    }
    fn merge(mut self, mut a: PrefactorMod) -> PrefactorMod {
        if self.modulus != a.modulus {
            let modulus = gcd(&self.modulus, &a.modulus);
            self.update_modulus(&modulus);
            a.update_modulus(&modulus);
        }
        // remove the set of incompatible solutions by adjusting modulus
        self.adjust_compability(&mut a);
        let self_valid = self.valid();
        let other_valid = a.valid();
        // this is how the chinese remainder theorem with two non-coprime parameters works
        let (common_valid, (mut self_fac, mut other_fac)) = xgcd(&self_valid, &other_valid);
        self_fac *= &self_valid;
        self_fac *= &a.possible;
        other_fac *= &other_valid;
        other_fac *= &self.possible;
        self_fac += &other_fac;
        self_fac /= &common_valid;
        self.possible = self_fac;
        self.unknown = gcd(&self.unknown, &a.unknown);
        self
    }
    fn update_modulus(&mut self, modulus: &BigInt) -> bool {
        if &self.modulus == modulus {
            return false;
        }
        self.modulus.clone_from(modulus);
        self.possible %= modulus;
        self.unknown = gcd(modulus, &self.unknown);
        true
    }
    fn valid(&self) -> BigInt {
        self.modulus.clone() / self.unknown.clone()
    }
    // in order to chinese remainder with a common factor, both polynomials modulo
    // the common factor need to be the same
    // if this is not the case, the modulus is adjusted
    fn adjust_compability(&mut self, other: &mut Self) {
        let common_valid = gcd(&self.valid(), &other.valid());
        let actual_valid = gcd(&(&self.possible - &other.possible), &common_valid);
        let modulus = &self.modulus / &common_valid * &actual_valid;
        self.update_modulus(&modulus);
        other.update_modulus(&modulus);
    }
    // iterate over all solutions in `modulus`, also calculating addout1, addout2
    fn iter(
        &self,
        addout1: &BigInt,
        addout2: &(BigInt, usize),
        modulus: &BigInt,
    ) -> impl Iterator<Item = (u64, u64, u64)> + use<> {
        let mut red = self.clone();
        red.update_modulus(modulus);
        let mod_addout1 = mod_red(addout1, modulus);
        let mod_addout2 = mod_red(&addout2.0, modulus);
        let mod_addfac = mod_red(&BigInt::from(addout2.1), modulus);
        let modulus = modulus.clone();
        (0u64..(&red.unknown).try_into().unwrap())
            .map(BigInt::from)
            .map(move |i| {
                let real_init: u64 = mod_red(&(i * red.valid() + &red.possible), &modulus)
                    .try_into()
                    .unwrap();
                let real_addout1: u64 = mod_red(&(&mod_addout1 - real_init), &modulus)
                    .try_into()
                    .unwrap();
                let real_addout2: u64 = mod_red(&(&mod_addout2 - &mod_addfac * real_init), &modulus)
                    .try_into()
                    .unwrap();
                (real_init, real_addout1, real_addout2)
            })
    }
}

// from b*x ≡ a mod m, try to calculate x mod m/y where y is the second return value
fn partial_mod_div(a: &BigInt, b: &BigInt, modulus: &mut BigInt) -> (BigInt, BigInt) {
    let common = gcd(b, modulus);
    // if we want b*x ≡ a mod m, and c divides both b and m,
    // then a must be divisible by c as well
    // if that is not the case, we determine the maximal modulus where this is true
    if !(a % &common).is_zero() {
        // assume for simplicity that modulus is a prime power p^k
        // then we have b = d*p^n, a = e*p^m with d, e not divisible by p
        // then gcd(b, p^k) = p^n (because n has to be smaller than k)
        // if m < n, then b doesn't divide a and we try to adjust k so that it does
        // this can be done by simply setting m = k so that we now have 0*x ≡ 0 mod p^m
        let mut x = common.clone() / gcd(a, &common);
        // this loop tries to calculate
        //    if m < n { k = m }
        // without having to factor the number to obtain the prime powers
        // this works by first determining p^m by squaring and gcd'ing the product of all p's where
        // m < n, so that we have the maximum powers that divide modulus
        loop {
            let new_x = gcd(&(&x * &x), modulus);
            if new_x == x {
                break;
            }
            x = new_x;
        }
        *modulus = modulus.clone() / &x * gcd(&x, a);
    }
    let (common, (b_inv_unmod, _)) = xgcd(b, modulus);
    let b_inv = mod_red(&b_inv_unmod, modulus);
    let inv = (a / &common * b_inv) % &*modulus;
    (inv, common)
}

// note: this can be replaced with a more efficient implementations, like the one in factor.rs, but
// i'm not feeling like doing it right now tbh
fn gcd(a: &BigInt, b: &BigInt) -> BigInt {
    xgcd(a, b).0
}

fn xgcd(a: &BigInt, b: &BigInt) -> (BigInt, (BigInt, BigInt)) {
    let mut a = a.abs();
    let mut b = b.abs();
    if a.is_zero() {
        return (b, (zero(), one()));
    }
    if b.is_zero() {
        return (a, (one(), zero()));
    }
    let mut a_fac = (one(), zero());
    let mut b_fac = (zero(), one());
    if a < b {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut a_fac, &mut b_fac);
    }
    while !b.is_zero() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut a_fac, &mut b_fac);
        let fac = &b / &a;
        let rem = &b % &a;
        b = rem;
        b_fac = (b_fac.0 - &fac * &a_fac.0, b_fac.1 - &fac * &a_fac.1);
    }
    (a, a_fac)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::tests::ReverseFileSet;
    use crate::endian::{Endian, Signedness, WordSpec};
    use quickcheck::{Arbitrary, Gen, TestResult};
    impl Arbitrary for FletcherBuilder<u64> {
        fn arbitrary(g: &mut Gen) -> Self {
            let mut new_fletcher = Fletcher::with_options();
            let width = ((u8::arbitrary(g) % 63 + 2) * 2) as usize;
            new_fletcher.width(width);
            let mut modulus = 0;
            while modulus <= 1 {
                modulus = u64::arbitrary(g);
                if width < 128 {
                    modulus %= 1 << (width / 2);
                }
            }
            new_fletcher.modulus(modulus);
            let init = u64::arbitrary(g) % modulus;
            new_fletcher.init(init);
            let swap = bool::arbitrary(g);
            new_fletcher.swap(swap);
            let addout1 = u64::arbitrary(g) % modulus;
            let addout2 = u64::arbitrary(g) % modulus;
            let addout = glue_sum(addout1, addout2, width, swap);
            new_fletcher.addout(addout);
            let wordspec = WordSpec::arbitrary(g);
            let max_word_width = ((width + 15) / 16).next_power_of_two() * 8;
            new_fletcher.wordsize(max_word_width.min(wordspec.wordsize));
            new_fletcher.inendian(wordspec.input_endian);
            new_fletcher.outendian(wordspec.output_endian);
            new_fletcher.signedness(wordspec.signedness);
            new_fletcher
        }
    }
    #[test]
    fn fletcher16() {
        let f16 = Fletcher::with_options()
            .width(16)
            .modulus(0xffu64)
            .addout(0x2233)
            .init(0x44)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![145u8, 43, 41, 159, 51, 200, 25, 53, 53, 75, 100, 41, 99],
            vec![238, 92, 59, 96, 189, 61, 241, 51],
            vec![33, 241, 149, 112, 184],
        ]);
        let mut naive = Fletcher::<u64>::with_options();
        naive.width(16).swap(false);
        let chk_files: Vec<_> = f.with_checksums(&f16);
        let reverser = reverse_fletcher(&naive, &chk_files, 0, false);
        assert!(!f.check_matching(&f16, reverser).is_failure());
    }
    #[quickcheck]
    fn qc_fletch_rev(
        files: ReverseFileSet,
        fletch_build: FletcherBuilder<u64>,
        known: (bool, bool, bool, bool),
        wordspec_known: (bool, bool, bool, bool),
    ) -> TestResult {
        let fletcher = fletch_build.build().expect("Could not build checksum");
        let mut naive = Fletcher::<u64>::with_options();
        naive.width(fletch_build.width.unwrap());
        if known.0 {
            naive.modulus(fletch_build.modulus.unwrap());
        }
        if known.1 {
            naive.init(fletch_build.init.unwrap());
        }
        if known.2 {
            naive.addout(fletch_build.addout.unwrap());
        }
        if known.3 {
            naive.swap(fletch_build.swap.unwrap());
        }
        if wordspec_known.0 {
            naive.wordsize(fletch_build.wordsize.unwrap());
        }
        if wordspec_known.1 {
            naive.inendian(fletch_build.input_endian.unwrap());
        }
        if wordspec_known.2 {
            naive.outendian(fletch_build.output_endian.unwrap());
        }
        if wordspec_known.3 {
            naive.signedness(fletch_build.signedness.unwrap());
        }
        let chk_files: Vec<_> = files.with_checksums(&fletcher);
        let reverser = reverse_fletcher(&naive, &chk_files, 0, false);
        files.check_matching(&fletcher, reverser)
    }
    #[test]
    fn error1() {
        let f16 = Fletcher::with_options()
            .width(32)
            .modulus(0x4d)
            .addout(0x110011)
            .init(0x35)
            .swap(true)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![
                0, 0, 0, 0, 0, 65, 0, 66, 59, 32, 3, 54, 55, 0, 58, 13, 66, 41, 0, 82, 29, 43, 35,
                20, 36, 50, 81, 10, 37, 33, 50, 21, 45, 70, 65, 18, 49, 22, 60, 35, 83, 0, 75, 87,
                59, 7, 76, 66, 44, 34, 23, 3, 1, 50, 71, 48, 30, 34, 41, 46, 6, 32, 5,
            ],
            vec![0, 0, 0, 0, 5, 55, 38, 55, 6, 50, 11, 43, 43, 16],
            vec![0, 0, 0, 0, 0, 0, 0, 10, 51, 59, 21, 29],
            vec![0, 0, 0, 37, 79, 42, 10],
        ]);
        let chk_files: Vec<_> = f.with_checksums(&f16);
        let mut naive = Fletcher::<u64>::with_options();
        naive.width(32);
        let reverser = reverse_fletcher(&naive, &chk_files, 0, false);
        assert!(!f.check_matching(&f16, reverser).is_failure());
    }
    #[test]
    fn error2() {
        let f16 = Fletcher::with_options()
            .width(102)
            .modulus(0x4d)
            .addout(0x170000000000042)
            .init(0x35)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 14, 54, 2, 0, 0, 3, 52, 23, 55, 10, 86, 40,
            ],
            vec![0, 48, 93, 15, 0, 0, 27, 58, 20, 22, 69, 42, 30, 74],
            vec![10, 94, 63, 27, 37, 41, 58, 33, 57, 77],
            vec![0, 11, 24],
        ]);
        let chk_files = f.with_checksums(&f16);
        let mut naive = Fletcher::<u64>::with_options();
        naive.width(102);
        let reverser = reverse_fletcher(&naive, &chk_files, 0, false);
        assert!(!f.check_matching(&f16, reverser).is_failure());
    }
    #[test]
    fn error3() {
        let f16 = Fletcher::with_options()
            .width(42)
            .modulus(0x3)
            .addout(0x200001)
            .init(0)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![48, 29, 22],
            vec![50, 0, 48],
            vec![47, 24],
            vec![],
        ]);
        let chk_files = f.with_checksums(&f16);
        let mut naive = Fletcher::<u64>::with_options();
        naive.width(42);
        let reverser = reverse_fletcher(&naive, &chk_files, 0, false);
        assert!(!f.check_matching(&f16, reverser).is_failure());
    }
    #[test]
    fn error4() {
        let f16 = Fletcher::with_options()
            .width(126)
            .modulus(0x5d)
            .addout(0x15000000000000001d)
            .init(0x31)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![
                0, 0, 0, 0, 0, 0, 37, 37, 10, 0, 63, 70, 18, 75, 57, 62, 64, 74, 87, 0, 20, 10, 76,
                65, 99, 19, 5, 22, 0, 69,
            ],
            vec![3, 23, 71, 58, 32, 10, 0, 51, 88, 59, 1, 85],
            vec![87, 21],
            vec![],
        ]);
        let chk_files = f.with_checksums(&f16);
        let mut naive = Fletcher::<u64>::with_options();
        naive.width(126);
        let reverser = reverse_fletcher(&naive, &chk_files, 0, false);
        assert!(!f.check_matching(&f16, reverser).is_failure());
    }
    #[test]
    fn error5() {
        let f16 = Fletcher::with_options()
            .width(42)
            .modulus(6u64)
            .addout(0x000000)
            .init(0)
            .swap(false)
            .inendian(Endian::Big)
            .outendian(Endian::Little)
            .wordsize(64)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![65, 51, 46, 37, 4, 12, 65, 44],
            vec![45, 78, 4, 14, 70, 24, 19, 45],
            vec![],
        ]);
        let chk_files = f.with_checksums(&f16);
        let mut naive = Fletcher::<u64>::with_options();
        naive
            .width(42)
            .swap(false)
            .inendian(Endian::Big)
            .outendian(Endian::Little)
            .wordsize(64);
        let reverser = reverse_fletcher(&naive, &chk_files, 0, false);
        assert!(!f.check_matching(&f16, reverser).is_failure());
    }
    #[test]
    fn error6() {
        // init + addout for the regular sum overflowed, changed to i128
        let f128 = Fletcher::with_options()
            .width(128)
            .modulus(0xcb80a6f9a8cd46f4u64)
            .init(0xb3ecf9878dbc2c93)
            .addout(0x9b8e3e2905a19ea31cb7d9ba3c8891fe)
            .swap(true)
            .inendian(Endian::Little)
            .wordsize(16)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![
                13, 255, 162, 255, 220, 220, 98, 238, 51, 0, 161, 137, 166, 137, 28, 37,
            ],
            vec![
                5, 38, 212, 62, 75, 1, 161, 207, 51, 50, 163, 94, 181, 67, 0, 206,
            ],
            vec![161, 64, 210, 72, 171, 168, 255, 226],
        ]);
        let chk_files = f.with_checksums(&f128);
        let mut naive = Fletcher::<u64>::with_options();
        naive
            .width(128)
            .init(0xb3ecf9878dbc2c93)
            .addout(0x9b8e3e2905a19ea31cb7d9ba3c8891fe);
        let reverser = reverse_fletcher(&naive, &chk_files, 0, false);
        assert!(!f.check_matching(&f128, reverser).is_failure());
    }
    #[test]
    fn error7() {
        let f16 = Fletcher::with_options()
            .width(10)
            .modulus(5u64)
            .addout(0x81)
            .init(0)
            .swap(false)
            .inendian(Endian::Big)
            .outendian(Endian::Big)
            .wordsize(8)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![
                76, 68, 237, 127, 232, 152, 85, 112, 110, 255, 145, 240, 6, 252, 63, 204, 86, 165,
                149, 217, 9, 213, 66, 0, 13, 216, 111, 138, 245, 52, 159, 24, 110, 131, 38, 197,
                218, 7, 228, 131, 24, 230, 195, 165, 240, 58, 75, 234, 67, 88, 3, 1, 166, 90, 71,
                59, 255, 232, 223, 88, 59, 239, 97, 144, 190, 1, 172, 164, 146, 246, 1, 1,
            ],
            vec![
                107, 214, 84, 116, 5, 236, 90, 81, 133, 86, 10, 1, 166, 0, 251, 98, 161, 235, 170,
                232, 1, 0, 213, 125, 199, 157, 90, 4, 84, 0, 95, 53, 33, 132, 43, 129, 128, 75, 92,
                23, 32, 255, 145, 40, 129, 137, 2, 0, 1, 103, 187, 86, 182, 155, 177, 223,
            ],
            vec![94, 1, 182, 0, 183, 61, 129, 141],
        ]);
        let chk_files = f.with_checksums(&f16);
        let mut naive = Fletcher::<u64>::with_options();
        naive.width(10).addout(0x81);
        let reverser = reverse_fletcher(&naive, &chk_files, 0, false).collect::<Vec<_>>();
        eprintln!("{:?}", reverser);
        assert!(
            !f.check_matching(&f16, reverser.iter().cloned())
                .is_failure()
        );
    }
    #[test]
    fn error8() {
        let f16 = Fletcher::with_options()
            .width(10)
            .modulus(3u64)
            .init(1)
            .addout(1)
            .swap(false)
            .inendian(Endian::Little)
            .outendian(Endian::Big)
            .wordsize(8)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![
                213, 135, 255, 136, 226, 145, 148, 73, 43, 34, 209, 239, 124, 73, 186, 105,
            ],
            vec![
                35, 73, 105, 174, 0, 214, 0, 82, 167, 23, 113, 58, 244, 201, 89, 235,
            ],
            vec![250, 199, 177, 8, 135, 127, 88, 148],
        ]);
        let chk_files = f.with_checksums(&f16);
        let mut naive = Fletcher::<u64>::with_options();
        naive.width(10).modulus(3);
        let reverser = reverse_fletcher(&naive, &chk_files, 0, false);
        assert!(!f.check_matching(&f16, reverser).is_failure());
    }

    #[test]
    fn error9() {
        let fletch = Fletcher::with_options()
            .modulus(2442387192987926634u64)
            .init(127264857433458109)
            .addout(1912567329028884011882136235663163392)
            .swap(true)
            .inendian(Endian::Little)
            .outendian(Endian::Little)
            .signedness(Signedness::Signed)
            .wordsize(64)
            .width(124)
            .build()
            .unwrap();

        let files = ReverseFileSet(vec![vec![48, 73, 55, 229, 255, 1, 103, 39], vec![], vec![]]);
        let chk_files = files.with_checksums(&fletch);
        let mut naive = Fletcher::<u64>::with_options();
        naive.width(124);
        let reverser = reverse_fletcher(&naive, &chk_files, 0, false);
        assert!(!files.check_matching(&fletch, reverser).is_failure());
    }
}

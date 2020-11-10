use super::{Fletcher, FletcherBuilder};
use crate::checksum::{unresult_iter, CheckReverserError};
use crate::factor::divisors_range;
use num_bigint::BigInt;
use num_traits::{one, zero, One, Signed, Zero};
use std::convert::TryInto;
use std::iter::Iterator;
pub fn reverse_fletcher<'a>(
    spec: &FletcherBuilder<u128>,
    chk_bytes: &'a [(&[u8], u128)],
    verbosity: u64,
) -> impl Iterator<Item = Result<Fletcher<u128>, CheckReverserError>> + 'a {
    let spec = spec.clone();
    let swap = spec
        .swap
        .map(|x| vec![x])
        .unwrap_or_else(|| vec![false, true]);
    swap.into_iter()
        .map(move |s| unresult_iter(reverse(&spec, chk_bytes, s, verbosity).map(|x| x.iter())))
        .flatten()
}

#[derive(Debug, Clone)]
struct ReversingResult {
    inits: PrefactorMod,
    addout1: BigInt,
    addout2: (BigInt, usize),
    modules: Vec<BigInt>,
    width: usize,
    swap: bool,
}

impl ReversingResult {
    fn iter(self) -> impl Iterator<Item = Fletcher<u128>> {
        let ReversingResult {
            inits,
            addout1,
            addout2,
            modules,
            width,
            swap,
        } = self;
        modules
            .into_iter()
            .map(move |m| {
                let module = if m.is_zero() {
                    0u128
                } else {
                    (&m).try_into().unwrap()
                };
                inits.iter(&addout1, &addout2, &m).map(move |(i, s1, s2)| {
                    let addout = glue_sum(s1, s2, width, swap);
                    Fletcher::with_options()
                        .addout(addout)
                        .init(i as u128)
                        .module(module)
                        .width(width)
                        .swap(swap)
                        .build()
                        .unwrap()
                })
            })
            .flatten()
    }
}

fn reverse(
    spec: &FletcherBuilder<u128>,
    chk_bytes: &[(&[u8], u128)],
    swap: bool,
    verbosity: u64,
) -> Result<ReversingResult, CheckReverserError> {
    let log = |s| {
        if verbosity > 0 {
            eprintln!("<fletcher> {}", s);
        }
    };
    let mut files = chk_bytes.to_owned();
    files.sort_unstable_by(|a, b| a.0.len().cmp(&b.0.len()).reverse());
    let spec = spec.clone();
    let width = match spec.width {
        Some(x) => x,
        None => return Err(CheckReverserError::MissingParameter("width")),
    };
    log("finding parameters of lower sum");
    let (module, addout1) = find_regular_sum(&spec, &files, swap);
    let mut module = BigInt::from(module);
    let mut addout1 = BigInt::from(addout1);
    let mut cumusums = cumusum(width, chk_bytes, &module, swap);
    if let Some(init) = spec.init {
        log("removing inits from upper sum");
        remove_init(&mut cumusums, &BigInt::from(init));
    }
    log("removing upper sum addout");
    let (red_files, mut addout2) = remove_addout2(&spec, cumusums, &module, swap);
    log("refining the module");
    let boneless_files = refine_module(&mut module, red_files);
    if module.is_zero() {
        return Err(CheckReverserError::UnsuitableFiles(
            "too short or too similar",
        ));
    }
    log("attempting to find init");
    let inits = find_init(&spec.init.map(BigInt::from), &mut module, boneless_files);
    let module = inits.module.clone();
    addout1 = mod_red(&addout1, &module);
    addout2.0 = mod_red(&addout2.0, &module);
    let (min, max) = chk_range(chk_bytes, width);
    log("try to find all possible module values");
    let modules = divisors_range(module.try_into().unwrap(), min, max)
        .into_iter()
        .map(BigInt::from)
        .collect();
    Ok(ReversingResult {
        inits,
        addout1,
        addout2,
        modules,
        width,
        swap,
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

fn chk_range(chks: &[(&[u8], u128)], width: usize) -> (u128, u128) {
    let mut min = 2;
    for (_, sum) in chks {
        // swap does not matter
        let (s1, s2) = split_sum(*sum, width, false);
        min = min.max(s1 as u128 + 1);
        min = min.max(s2 as u128 + 1);
    }
    let max = 1 << (width / 2);
    (min, max)
}

fn find_regular_sum(spec: &FletcherBuilder<u128>, files: &[(&[u8], u128)], swap: bool) -> (u128, i128) {
    let width = spec.width.unwrap();
    let mut sums = Vec::new();
    for (f, chk) in files.iter() {
        let (chk_lo, _) = split_sum(*chk, width, swap);
        sums.push(f.iter().copied().map(i128::from).sum::<i128>() - chk_lo as i128);
    }
    let mut module = 0;
    let maybe_init = spec
        .addout
        .map(|x| spec.init.map(|y| y as u64 + split_sum(x, width, swap).0))
        .flatten();
    let sum1_addout = super::super::modsum::rev::find_largest_mod(&sums, maybe_init, &mut module);
    (module, sum1_addout)
}

fn remove_init(sums: &mut Vec<(BigInt, usize)>, init: &BigInt) {
    for (s, l) in sums.iter_mut() {
        *s -= init * BigInt::from(*l);
        *l = 0;
    }
}

fn remove_addout2(
    spec: &FletcherBuilder<u128>,
    mut sums: Vec<(BigInt, usize)>,
    module: &BigInt,
    swap: bool,
) -> (Vec<(BigInt, usize)>, (BigInt, usize)) {
    let width = spec.width.unwrap();
    let maybe_addout = spec.addout.map(|x| BigInt::from(split_sum(x, width, swap).1));
    let mut ret_vec = Vec::new();
    let mut prev = sums
        .pop()
        .expect("Internal Error: Zero-length vector given to remove_addout2");
    let addout2 = match &maybe_addout {
        Some(addout) => {
            ret_vec.push((mod_red(&(&prev.0 - addout), module), prev.1));
            (addout.clone(), 0)
        }
        None => prev.clone(),
    };
    for (p, l) in sums.into_iter().rev() {
        let appendix = match (&maybe_addout, l != 0 && l == prev.1) {
            (None, _) | (_, true) => (mod_red(&(&p - prev.0), module), l - prev.1),
            (Some(addout), false) => (mod_red(&(&p - addout), module), l),
        };
        ret_vec.push(appendix);
        prev = (p, l);
    }
    (ret_vec, addout2)
}

fn refine_module(module: &mut BigInt, sums: Vec<(BigInt, usize)>) -> Vec<(BigInt, usize)> {
    let mut non_zero = Vec::new();
    for (s, l) in sums {
        if l != 0 {
            non_zero.push((s, l));
            continue;
        }
        *module = gcd(module, &s);
    }
    for ((sa, la), (sb, lb)) in non_zero.iter().zip(non_zero.iter().skip(1)) {
        let bla = BigInt::from(*la);
        let blb = BigInt::from(*lb);
        let common = gcd(&bla, &blb);
        let a_part = mod_red(&(bla / &common), module);
        let b_part = mod_red(&(blb / &common), module);
        let mut mul_sa = sa * b_part;
        let mut mul_sb = sb * a_part;
        mul_sa = mod_red(&mul_sa, module);
        mul_sb = mod_red(&mul_sb, module);
        *module = gcd(module, &(mul_sa - mul_sb));
    }
    non_zero
        .iter()
        .map(|(s, l)| (mod_red(s, module), *l))
        .collect()
}

fn cumusum(width: usize, chk_bytes: &[(&[u8], u128)], module: &BigInt, swap: bool) -> Vec<(BigInt, usize)> {
    let mut sums = Vec::new();
    for (bytes, chk) in chk_bytes {
        let mut current_sum: BigInt = zero();
        let mut cumusum: BigInt = zero();
        for byte in *bytes {
            current_sum += BigInt::from(*byte);
            cumusum += &current_sum;
        }
        cumusum = mod_red(&cumusum, module);
        let (_, check) = split_sum(*chk, width, swap);
        sums.push((BigInt::from(check) - cumusum, bytes.len()));
    }
    sums
}

// modular reduction, because % is just wrong
fn mod_red(n: &BigInt, module: &BigInt) -> BigInt {
    if module.is_zero() {
        // yes, n modulo 0 is n and i will die on this hill
        n.clone()
    } else {
        let k = n % module;
        if k < zero() {
            module + k
        } else {
            k
        }
    }
}
fn find_init(
    maybe_init: &Option<BigInt>,
    module: &mut BigInt,
    sums: Vec<(BigInt, usize)>,
) -> PrefactorMod {
    if module.is_one() {
        return PrefactorMod::empty();
    };
    let mut ret = PrefactorMod::new_init(maybe_init, module);
    for (p, l) in sums {
        let file_solutions = PrefactorMod::from_sum(&p, l, module);
        ret = match file_solutions.map(|f| ret.merge(f)).flatten() {
            Some(valid) => valid,
            None => return PrefactorMod::empty(),
        }
    }
    ret
}
// describes a set of solutions for unknown*possible % hull
#[derive(Clone, Debug)]
struct PrefactorMod {
    unknown: BigInt,
    possible: BigInt,
    module: BigInt,
}

impl PrefactorMod {
    fn empty() -> PrefactorMod {
        PrefactorMod {
            module: one(),
            unknown: one(),
            possible: zero(),
        }
    }
    fn from_sum(sum: &BigInt, power: usize, module: &mut BigInt) -> Option<PrefactorMod> {
        let bpower = BigInt::from(power);
        let (possible, unknown) = partial_mod_div(sum, &bpower, module);
        if module.is_one() {
            return None;
        }
        Some(PrefactorMod {
            unknown,
            possible,
            module: module.clone(),
        })
    }
    fn new_init(maybe_init: &Option<BigInt>, module: &BigInt) -> Self {
        let (unknown, possible) = match maybe_init {
            None => (module.clone(), zero()),
            Some(init) => (one(), init.clone()),
        };
        PrefactorMod {
            unknown,
            possible,
            module: module.clone(),
        }
    }
    fn merge(mut self, mut a: PrefactorMod) -> Option<PrefactorMod> {
        if self.module != a.module {
            let (module, _) = xgcd(&self.module, &a.module);
            self.update_module(&module);
            a.update_module(&module);
        }
        self.adjust_compability(&mut a);
        let self_valid = self.valid();
        let other_valid = a.valid();
        // bezout
        let (common_valid, (mut self_fac, mut other_fac)) = xgcd(&self_valid, &other_valid);
        self_fac *= &self_valid;
        self_fac *= &a.possible;
        other_fac *= &other_valid;
        other_fac *= &self.possible;
        self_fac += &other_fac;
        self_fac /= &common_valid;
        self.possible = self_fac;
        self.unknown = gcd(&self.unknown, &a.unknown);
        Some(self)
    }
    fn update_module(&mut self, module: &BigInt) -> bool {
        if &self.module == module {
            return false;
        }
        self.module = module.clone();
        self.possible %= module;
        self.unknown = gcd(module, &self.unknown);
        true
    }
    fn valid(&self) -> BigInt {
        self.module.clone() / self.unknown.clone()
    }
    // in order to chinese remainder with a common factor, both polynomials modulo
    // the common factor need to be the same
    // if this is not the case, the hull is adjusted
    fn adjust_compability(&mut self, other: &mut Self) {
        let common_valid = gcd(&self.valid(), &other.valid());
        let actual_valid = gcd(&(&self.possible - &other.possible), &common_valid);
        let module = &self.module / &common_valid * &actual_valid;
        if module.is_one() {
            return;
        }
        self.update_module(&module);
        other.update_module(&module);
    }
    fn iter(
        &self,
        addout1: &BigInt,
        addout2: &(BigInt, usize),
        module: &BigInt,
    ) -> impl Iterator<Item = (u64, u64, u64)> {
        let mut red = self.clone();
        red.update_module(&module);
        let mod_addout1 = mod_red(addout1, module);
        let mod_addout2 = mod_red(&addout2.0, module);
        let mod_addfac = mod_red(&BigInt::from(addout2.1), module);
        let module = module.clone();
        (0u64..(&red.unknown).try_into().unwrap())
            .into_iter()
            .map(BigInt::from)
            .map(move |i| {
                let real_init: u64 = mod_red(&(i * (&red).valid() + &red.possible), &module)
                    .try_into()
                    .unwrap();
                let real_addout1: u64 = mod_red(&(&mod_addout1 - real_init), &module)
                    .try_into()
                    .unwrap();
                let real_addout2: u64 = mod_red(&(&mod_addout2 - &mod_addfac * real_init), &module)
                    .try_into()
                    .unwrap();
                (real_init, real_addout1, real_addout2)
            })
    }
}

fn partial_mod_div(a: &BigInt, b: &BigInt, module: &mut BigInt) -> (BigInt, BigInt) {
    let common = gcd(&b, &module);
    if !(a % &common).is_zero() {
        let mut x = common.clone() / gcd(a, &common);
        loop {
            let new_x = gcd(&(&x * &x), &module);
            if new_x == x {
                break;
            }
            x = new_x;
        }
        *module = module.clone() / &x * gcd(&x, &a);
    }
    let (common, (b_inv_unmod, _)) = xgcd(&b, &module);
    let b_inv = mod_red(&b_inv_unmod, &module);
    let inv = (a / &common * b_inv) % &*module;
    (inv, common)
}

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
    use crate::checksum::Digest;
    use quickcheck::{Arbitrary, TestResult};
    impl Arbitrary for FletcherBuilder<u128> {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            let mut new_fletcher = Fletcher::with_options();
            let width = ((u8::arbitrary(g) % 63 + 2) * 2) as usize;
            new_fletcher.width(width as usize);
            let mut module = u128::arbitrary(g) % (1 << (width / 2));
            while module <= 1 {
                module = u128::arbitrary(g) % (1 << (width / 2));
            }
            new_fletcher.module(module);
            let init = u64::arbitrary(g) as u128 % module;
            new_fletcher.init(init);
            let swap = bool::arbitrary(g);
            new_fletcher.swap(swap);
            let addout1 = u64::arbitrary(g) as u128 % module;
            let addout2 = u64::arbitrary(g) as u128 % module;
            let addout = glue_sum(addout1 as u64, addout2 as u64, width, swap);
            new_fletcher.addout(addout);
            new_fletcher
        }
    }
    #[test]
    fn fletcher16() {
        let f16 = Fletcher::with_options()
            .width(16)
            .module(0xffu128)
            .addout(0x2233)
            .init(0x44)
            .build()
            .unwrap();
        let f = vec![
            &[145u8, 43, 41, 159, 51, 200, 25, 53, 53, 75, 100, 41, 99][..],
            &[238, 92, 59, 96, 189, 61, 241, 51][..],
            &[33, 241, 149, 112, 184][..],
        ];
        let chk_files: Vec<_> = f
            .iter()
            .map(|f| {
                let checksum: u128 = f16.digest(*f).unwrap();
                (*f, checksum)
            })
            .collect();
        let mut naive = Fletcher::<u128>::with_options();
        naive.width(16).swap(false);
        let m: Result<Vec<_>, _> = reverse_fletcher(&naive, &chk_files, 0).collect();
        if let Ok(x) = m {
            assert_eq!(vec![f16], x)
        }
    }
    #[quickcheck]
    fn qc_fletch_rev(
        mut files: Vec<Vec<u8>>,
        fletch_build: FletcherBuilder<u128>,
        known: (bool, bool, bool, bool),
    ) -> TestResult {
        files.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(&b)).reverse());
        if files.iter().zip(files.iter().skip(1)).any(|(a, b)| a == b) || files.len() <= 3 {
            return TestResult::discard();
        }
        let fletcher = fletch_build.build().unwrap();
        let mut naive = Fletcher::<u128>::with_options();
        naive.width(fletch_build.width.unwrap());
        if known.0 {
            naive.module(fletch_build.module.unwrap());
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
        let chk_files: Vec<_> = files
            .iter()
            .map(|f| {
                let checksum = fletcher.digest(f.as_slice()).unwrap();
                (f.as_slice(), checksum)
            })
            .collect();
        let reverser = reverse_fletcher(&naive, &chk_files, 0);
        let mut has_appeared = false;
        for fletch_loop in reverser {
            let fletch_loop = match fletch_loop {
                Ok(x) => x,
                Err(_) => return TestResult::discard(),
            };
            if !has_appeared && fletch_loop == fletcher {
                has_appeared = true;
            }
            for (file, original_check) in &chk_files {
                let checksum = fletch_loop.digest(*file).unwrap();
                if checksum != *original_check {
                    eprintln!("expected checksum: {:x}", original_check);
                    eprintln!("actual checksum: {:x}", checksum);
                    eprintln!("fletcher: {}", fletch_loop);
                    return TestResult::failed();
                }
            }
        }
        TestResult::from_bool(has_appeared)
    }
    #[test]
    fn error1() {
        let f16 = Fletcher::with_options()
            .width(32)
            .module(0x4d)
            .addout(0x110011)
            .init(0x35)
            .swap(true)
            .build()
            .unwrap();
        let f = vec![
            &[
                0, 0, 0, 0, 0, 65, 0, 66, 59, 32, 3, 54, 55, 0, 58, 13, 66, 41, 0, 82, 29, 43, 35,
                20, 36, 50, 81, 10, 37, 33, 50, 21, 45, 70, 65, 18, 49, 22, 60, 35, 83, 0, 75, 87,
                59, 7, 76, 66, 44, 34, 23, 3, 1, 50, 71, 48, 30, 34, 41, 46, 6, 32, 5,
            ][..],
            &[0, 0, 0, 0, 5, 55, 38, 55, 6, 50, 11, 43, 43, 16][..],
            &[0, 0, 0, 0, 0, 0, 0, 10, 51, 59, 21, 29][..],
            &[0, 0, 0, 37, 79, 42, 10][..],
        ];
        let chk_files: Vec<_> = f
            .iter()
            .map(|f| {
                let checksum: u128 = f16.digest(*f).unwrap();
                (*f, checksum)
            })
            .collect();
        let mut naive = Fletcher::<u128>::with_options();
        naive.width(32);
        let m: Result<Vec<_>, _> = reverse_fletcher(&naive, &chk_files, 0).collect();
        if let Ok(x) = m {
            assert_eq!(x, vec![f16]);
        }
    }
    #[test]
    fn error2() {
        let f16 = Fletcher::with_options()
            .width(102)
            .module(0x4d)
            .addout(0x170000000000042)
            .init(0x35)
            .build()
            .unwrap();
        let f = vec![
            &[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 14, 54, 2, 0, 0, 3, 52, 23, 55, 10, 86, 40,
            ][..],
            &[0, 48, 93, 15, 0, 0, 27, 58, 20, 22, 69, 42, 30, 74][..],
            &[10, 94, 63, 27, 37, 41, 58, 33, 57, 77][..],
            &[0, 11, 24][..],
        ];
        let chk_files: Vec<_> = f
            .iter()
            .map(|f| {
                let checksum: u128 = f16.digest(*f).unwrap();
                (*f, checksum)
            })
            .collect();
        let mut naive = Fletcher::<u128>::with_options();
        naive.width(102);
        let m: Result<Vec<_>, _> = reverse_fletcher(&naive, &chk_files, 0).collect();
        if let Ok(x) = m {
            assert_eq!(x, vec![f16]);
        }
    }
    #[test]
    fn error3() {
        let f16 = Fletcher::with_options()
            .width(42)
            .module(0x3)
            .addout(0x200001)
            .init(0)
            .build()
            .unwrap();
        let f = vec![&[48, 29, 22][..], &[50, 0, 48][..], &[47, 24][..], &[][..]];
        let chk_files: Vec<_> = f
            .iter()
            .map(|f| {
                let checksum: u128 = f16.digest(*f).unwrap();
                (*f, checksum)
            })
            .collect();
        let mut naive = Fletcher::<u128>::with_options();
        naive.width(42);
        let m: Result<Vec<_>, _> = reverse_fletcher(&naive, &chk_files, 0).collect();
        if let Ok(x) = m {
            assert!(x.contains(&f16))
        }
    }
    #[test]
    fn error4() {
        let f16 = Fletcher::with_options()
            .width(126)
            .module(0x5d)
            .addout(0x15000000000000001d)
            .init(0x31)
            .build()
            .unwrap();
        let f = vec![
            &[
                0, 0, 0, 0, 0, 0, 37, 37, 10, 0, 63, 70, 18, 75, 57, 62, 64, 74, 87, 0, 20, 10, 76,
                65, 99, 19, 5, 22, 0, 69,
            ][..],
            &[3, 23, 71, 58, 32, 10, 0, 51, 88, 59, 1, 85][..],
            &[87, 21][..],
            &[][..],
        ];
        let chk_files: Vec<_> = f
            .iter()
            .map(|f| {
                let checksum: u128 = f16.digest(*f).unwrap();
                (*f, checksum)
            })
            .collect();
        let mut naive = Fletcher::<u128>::with_options();
        naive.width(126);
        let m: Result<Vec<_>, _> = reverse_fletcher(&naive, &chk_files, 0).collect();
        if let Ok(x) = m {
            assert_eq!(x, vec![f16])
        }
    }
}

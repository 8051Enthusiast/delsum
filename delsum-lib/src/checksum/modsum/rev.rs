//! This module contains the function(s) for reversing the parameters for a modular sum.
//!
//! Generally, to find out the parameters, the checksums and their width are needed, and 2 of the following (with at least one file):
//! * value of `init`
//! * value of `module`
//! * a file with checksum
//! * a different file with checksum
//!
//! Of course, giving more files will result in fewer false positives.
use super::{ModSum, ModSumBuilder};
use crate::checksum::{unresult_iter, CheckReverserError};
use crate::factor::{divisors_range, gcd};
use std::iter::Iterator;
/// Find the parameters of a modsum algorithm.
///
/// `spec` contains the known parameters of the algorithm (by setting the corresponding values in the builder).
/// `chk_bytes` are pairs of files and their checksums.
/// `verbosity` makes the function output what it is doing
///
/// The `width` parameter of the builder has to be set.
pub fn reverse_modsum(
    spec: &ModSumBuilder<u64>,
    // note: even though all the sums are supposed to be u64,
    // for ease of intercompability with other reverse function,
    // we take u128
    chk_bytes: &[(&[u8], u128)],
    verbosity: u64,
) -> impl Iterator<Item = Result<ModSum<u64>, CheckReverserError>> {
    let res = reverse(spec, chk_bytes, verbosity).map(|x| x.iter());
    unresult_iter(res)
}
struct RevResult {
    modlist: Vec<u128>,
    init: i128,
    width: usize,
}

impl RevResult {
    // iterate over all possible modules and calculate the corresponding init values
    fn iter(self) -> impl Iterator<Item = ModSum<u64>> {
        let Self {
            modlist,
            init,
            width,
        } = self;
        modlist.into_iter().map(move |module| {
            let init_negative = init < 0;
            let mut init = init.abs() as u128 % module;
            if init_negative {
                init = module - init;
            }
            ModSum::with_options()
                .width(width)
                .module(module as u64)
                .init(init as u64)
                .build()
                .unwrap()
        })
    }
}

// If we have a file with the bytes [a, b, c, d] we have a checksum of the form (init + a + b + c + d) mod m.
// By subtracting a + b + c + d from the checksum (without mod'ing by m because we don't know m yet), we get
// init mod m.
// If we have two files, we can take their difference and have a number that is 0 mod m, which means m divides this number.
// The solutions are then the divisors m in the appropiate range.
fn reverse(
    spec: &ModSumBuilder<u64>,
    chk_bytes: &[(&[u8], u128)],
    verbosity: u64,
) -> Result<RevResult, CheckReverserError> {
    let log = |s| {
        if verbosity > 0 {
            eprintln!("<modsum> {}", s);
        }
    };
    let spec = spec.clone();
    let width = spec
        .width
        .ok_or(CheckReverserError::MissingParameter("width"))?;
    let mut sums = Vec::<i128>::new();
    let max_sum = 1u128 << width;
    let mut min_sum = 0;
    log("summing files up");
    for (f, chk) in chk_bytes {
        min_sum = min_sum.max(*chk as u128);
        // here we calculate (init mod m)
        sums.push(f.iter().copied().map(i128::from).sum::<i128>() - *chk as i128);
    }
    let mut module = 0;
    log("removing inits");
    // here we find module by gcd'ing between the differences (init - init == 0 mod m)
    let init = find_largest_mod(&sums, spec.init, &mut module);
    if module == 0 {
        return Err(CheckReverserError::UnsuitableFiles(
            "too short or too similar",
        ));
    }
    log("finding all possible factors");
    // find all possible divisors
    let modlist = divisors_range(module, min_sum + 1, max_sum);
    Ok(RevResult {
        modlist,
        init,
        width,
    })
}

pub(crate) fn find_largest_mod(sums: &[i128], maybe_init: Option<u64>, module: &mut u128) -> i128 {
    let init = match maybe_init {
        Some(i) => {
            // if we already have init, we can just subtract that from the sum and get a multiple of m
            let init = i as i128;
            for s in sums {
                *module = gcd(*module, (s + init).abs() as u128);
            }
            i as i128
        }
        None => {
            // otherwise their difference will do, but we do get one gcd less
            for (s1, s2) in sums.iter().zip(sums.iter().skip(1)) {
                *module = gcd(*module, (s1 - s2).abs() as u128);
            }
            -sums[0]
        }
    };
    init
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::Digest;
    use quickcheck::{Arbitrary, TestResult};
    impl Arbitrary for ModSumBuilder<u64> {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            let mut new_modsum = ModSum::with_options();
            let width = u8::arbitrary(g) % 64 + 1;
            new_modsum.width(width as usize);
            let module = if width < 64 {
                u64::arbitrary(g) % (1 << width)
            } else {
                u64::arbitrary(g)
            };
            new_modsum.module(module);
            let init = if module != 0 {
                u64::arbitrary(g) % module
            } else {
                u64::arbitrary(g)
            };
            new_modsum.init(init);
            new_modsum
        }
    }

    #[quickcheck]
    fn qc_modsum_rev(
        mut files: Vec<Vec<u8>>,
        modsum_build: ModSumBuilder<u64>,
        known: (bool, bool),
    ) -> TestResult {
        files.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(&b)).reverse());
        if files.iter().zip(files.iter().skip(1)).any(|(a, b)| a == b) || files.len() <= 2 {
            return TestResult::discard();
        }
        let modsum = modsum_build.build().unwrap();
        let mut naive = ModSum::<u64>::with_options();
        naive.width(modsum_build.width.unwrap());
        if known.0 {
            naive.module(modsum_build.module.unwrap());
        }
        if known.1 {
            naive.init(modsum_build.init.unwrap());
        }
        let chk_files: Vec<_> = files
            .iter()
            .map(|f| {
                let checksum = modsum.digest(f.as_slice()).unwrap();
                (f.as_slice(), checksum as u128)
            })
            .collect();
        let reverser = reverse_modsum(&naive, &chk_files, 0);
        let mut has_appeared = false;
        for modsum_loop in reverser {
            let modsum_loop = match modsum_loop {
                Err(_) => return TestResult::discard(),
                Ok(x) => x,
            };
            if modsum_loop == modsum {
                has_appeared = true;
            }
            for (file, original_check) in &chk_files {
                let checksum = modsum_loop.digest(*file).unwrap();
                if checksum as u128 != *original_check {
                    eprintln!("expected checksum: {:x}", original_check);
                    eprintln!("actual checksum: {:x}", checksum);
                    eprintln!("modsum: {}", modsum_loop);
                    return TestResult::failed();
                }
            }
        }
        TestResult::from_bool(has_appeared)
    }
    #[test]
    fn error1() {
        let modsum = ModSum::with_options()
            .width(38)
            .module(10)
            .init(1)
            .build()
            .unwrap();
        let f = vec![
            &[][..],
            &[0][..],
            &[30, 98, 74, 46, 90, 70, 18, 37, 44, 53, 53, 20, 47, 39][..],
        ];
        let chk_files: Vec<_> = f
            .iter()
            .map(|f| {
                let checksum = modsum.digest(*f).unwrap() as u128;
                (*f, checksum)
            })
            .collect();
        let mut naive = ModSum::<u64>::with_options();
        naive.width(38);
        let m: Result<Vec<_>, _> = reverse_modsum(&naive, &chk_files, 0).collect();
        if let Ok(x) = m {
            assert!(x.contains(&modsum))
        }
    }
    #[test]
    fn error2() {
        let modsum = ModSum::with_options()
            .width(38)
            .module(40)
            .init(2)
            .build()
            .unwrap();
        let f = vec![
            &[61, 25, 35, 56, 90, 96, 75][..],
            &[8, 94, 62, 74][..],
            &[82, 11, 99, 46][..],
        ];
        let chk_files: Vec<_> = f
            .iter()
            .map(|f| {
                let checksum = modsum.digest(*f).unwrap() as u128;
                (*f, checksum)
            })
            .collect();
        let mut naive = ModSum::<u64>::with_options();
        naive.width(38);
        let m: Result<Vec<_>, _> = reverse_modsum(&naive, &chk_files, 0).collect();
        if let Ok(x) = m {
            assert!(x.contains(&modsum))
        }
    }
}

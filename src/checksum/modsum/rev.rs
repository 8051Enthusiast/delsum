use super::{ModSumBuilder, ModSum};
use std::iter::Iterator;
use crate::factor::{gcd, divisors_range};
use crate::checksum::CheckReverserError;
pub fn reverse_modsum(
    spec: &ModSumBuilder<u64>,
    // note: even though all the sums are supposed to be u64,
    // for ease of intercompability with other reverse function,
    // we take u128
    chk_bytes: &[(&[u8], u128)],
    verbosity: u64,
) -> Result<impl Iterator<Item = ModSum<u64>>, CheckReverserError> {
    let log = |s| if verbosity > 0 {
        eprintln!("<modsum> {}", s);
    };
    let spec = spec.clone();
    let width = spec.width.ok_or(CheckReverserError::MissingParameter("width"))?;
    let mut sums = Vec::<i128>::new();
    let max_sum = 1u128 << width;
    let mut min_sum = 0;
    log("summing files up");
    for (f, chk) in chk_bytes {
        min_sum = min_sum.max(*chk as u128);
        sums.push(f.iter().copied().map(i128::from).sum::<i128>() - *chk as i128);
    }
    let mut hull = 0;
    log("removing inits");
    let init = match spec.init {
        Some(i) => {
            let init = i as i128;
            for s in sums {
                hull = gcd(hull, (s + init).abs() as u128);
            }
            i as i128
        },
        None => {
            for (s1, s2) in sums.iter().zip(sums.iter().skip(1)) {
                hull = gcd(hull, (s1 - s2).abs() as u128);
            }
            -sums[0]
        }
    };
    if hull == 0 {
        return Err(CheckReverserError::UnsuitableFiles("sum of files to small to wrap around"));
    }
    log("finding all possible factors");
    let modules = divisors_range(hull, min_sum + 1, max_sum);
    Ok(modules.into_iter().map(move |module| {
        let init_negative = init < 0;
        let mut init = init.abs() as u128 % module;
        if init_negative {
            init = module - init;
        }
        ModSum::with_options().width(width).module(module as u64).init(init as u64).build().unwrap()
    }))
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
        let reverser = match reverse_modsum(&naive, &chk_files, 0) {
            Ok(r) => r,
            Err(_) => return TestResult::discard()
        };
        let mut has_appeared = false;
        for modsum_loop in reverser {
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
}
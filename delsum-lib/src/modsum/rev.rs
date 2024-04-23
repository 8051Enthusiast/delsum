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
use crate::checksum::CheckReverserError;
use crate::endian::{bytes_to_int, wordspec_combos, WordSpec};
use crate::divisors::{divisors_range, gcd};
use crate::utils::unresult_iter;
use std::iter::Iterator;
/// Find the parameters of a modsum algorithm.
///
/// `spec` contains the known parameters of the algorithm (by setting the corresponding values in the builder).
/// `chk_bytes` are pairs of files and their checksums.
/// `verbosity` makes the function output what it is doing
///
/// The `width` parameter of the builder has to be set.
pub fn reverse_modsum<'a>(
    spec: &ModSumBuilder<u64>,
    chk_bytes: &'a [(&[u8], Vec<u8>)],
    verbosity: u64,
    extended_search: bool,
) -> impl Iterator<Item = Result<ModSum<u64>, CheckReverserError>> + 'a {
    let spec = spec.clone();
    wordspec_combos(
        spec.wordsize,
        spec.input_endian,
        spec.output_endian,
        spec.width.unwrap(),
        extended_search,
    )
    .into_iter()
    .flat_map(move |wordspec| {
        let rev = match spec.width {
            None => Err(CheckReverserError::MissingParameter("width")),
            Some(width) => {
                let chk_words: Vec<_> = chk_bytes
                    .iter()
                    .map(|(f, c)| {
                        (
                            wordspec.iter_words(f),
                            bytes_to_int(c, wordspec.output_endian),
                        )
                    })
                    .collect();
                let revspec = RevSpec {
                    width,
                    init: spec.init,
                    module: spec.module,
                    wordspec,
                };
                reverse(revspec, chk_words, verbosity).map(|x| x.iter())
            }
        };
        unresult_iter(rev)
    })
}

struct RevSpec {
    width: usize,
    init: Option<u64>,
    module: Option<u64>,
    wordspec: WordSpec,
}

struct RevResult {
    modlist: Vec<u128>,
    init: i128,
    width: usize,
    wordspec: WordSpec,
}

impl RevResult {
    // iterate over all possible modules and calculate the corresponding init values
    fn iter(self) -> impl Iterator<Item = ModSum<u64>> {
        let Self {
            modlist,
            init,
            width,
            wordspec,
        } = self;
        modlist.into_iter().map(move |module| {
            let init_negative = init < 0;
            let mut init = init.unsigned_abs() % module;
            if init_negative {
                init = module - init;
            }
            ModSum::with_options()
                .width(width)
                .module(module as u64)
                .init(init as u64)
                .inendian(wordspec.input_endian)
                .outendian(wordspec.output_endian)
                .wordsize(wordspec.wordsize)
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
    spec: RevSpec,
    chk_bytes: Vec<(impl Iterator<Item = u64>, u128)>,
    verbosity: u64,
) -> Result<RevResult, CheckReverserError> {
    let log = |s| {
        if verbosity > 0 {
            eprintln!("<modsum> {}", s);
        }
    };
    let width = spec.width;
    let mut sums = Vec::<i128>::new();
    let max_sum = 1u128 << width;
    let mut min_sum = 0;
    log("summing files up");
    for (f, chk) in chk_bytes {
        min_sum = min_sum.max(chk);
        // here we calculate (init mod m)
        sums.push(f.map(i128::from).sum::<i128>() - chk as i128);
    }
    let original_mod = spec
        .module
        .map(|x| if x == 0 { 1u128 << width } else { x as u128 });
    let mut module = original_mod.unwrap_or(0);
    log("removing inits");
    // here we find module by gcd'ing between the differences (init - init == 0 mod m)
    let init = find_largest_mod(&sums, spec.init.map(i128::from), &mut module);
    if module == 0 {
        return Err(CheckReverserError::UnsuitableFiles(
            "too short or too similar",
        ));
    }
    log("finding all possible factors");
    // find all possible divisors
    let modlist = match original_mod {
        Some(x) => {
            if x == module && x > min_sum && x <= max_sum {
                vec![module]
            } else {
                Vec::new()
            }
        }
        None => divisors_range(module, min_sum + 1, max_sum),
    };
    Ok(RevResult {
        modlist,
        init,
        width,
        wordspec: spec.wordspec,
    })
}

pub(crate) fn find_largest_mod(sums: &[i128], maybe_init: Option<i128>, module: &mut u128) -> i128 {
    match maybe_init {
        Some(i) => {
            // if we already have init, we can just subtract that from the sum and get a multiple of m
            for s in sums {
                *module = gcd(*module, (s + i).unsigned_abs());
            }
            i
        }
        None => {
            // otherwise their difference will do, but we do get one gcd less
            for (s1, s2) in sums.iter().zip(sums.iter().skip(1)) {
                *module = gcd(*module, (s1 - s2).unsigned_abs());
            }
            -sums[0]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::tests::ReverseFileSet;
    use crate::endian::Endian;
    use quickcheck::{Arbitrary, Gen, TestResult};
    impl Arbitrary for ModSumBuilder<u64> {
        fn arbitrary(g: &mut Gen) -> Self {
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
            let wordspec = WordSpec::arbitrary(g);
            let max_word_width = ((width as usize + 7) / 8).next_power_of_two() * 8;
            new_modsum.wordsize(max_word_width.min(wordspec.wordsize));
            new_modsum.inendian(wordspec.input_endian);
            new_modsum.outendian(wordspec.output_endian);
            new_modsum
        }
    }

    #[quickcheck]
    fn qc_modsum_rev(
        files: ReverseFileSet,
        modsum_build: ModSumBuilder<u64>,
        known: (bool, bool),
        wordspec_known: (bool, bool, bool),
    ) -> TestResult {
        let modsum = modsum_build.build().unwrap();
        let mut naive = ModSum::<u64>::with_options();
        naive.width(modsum_build.width.unwrap());
        if known.0 {
            naive.module(modsum_build.module.unwrap());
        }
        if known.1 {
            naive.init(modsum_build.init.unwrap());
        }
        if wordspec_known.0 {
            naive.wordsize(modsum_build.wordsize.unwrap());
        }
        if wordspec_known.1 {
            naive.inendian(modsum_build.input_endian.unwrap());
        }
        if wordspec_known.2 {
            naive.outendian(modsum_build.output_endian.unwrap());
        }
        let chk_files: Vec<_> = files.with_checksums(&modsum);
        let reverser = reverse_modsum(&naive, &chk_files, 0, false);
        files.check_matching(&modsum, reverser)
    }
    #[test]
    fn error1() {
        let modsum = ModSum::with_options()
            .width(38)
            .module(10)
            .init(1)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![],
            vec![0],
            vec![30, 98, 74, 46, 90, 70, 18, 37, 44, 53, 53, 20, 47, 39],
        ]);
        let chk_files: Vec<_> = f.with_checksums(&modsum);
        let mut naive = ModSum::<u64>::with_options();
        naive.width(38);
        let m: Result<Vec<_>, _> = reverse_modsum(&naive, &chk_files, 0, false).collect();
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
        let f = ReverseFileSet(vec![
            vec![61, 25, 35, 56, 90, 96, 75],
            vec![8, 94, 62, 74],
            vec![82, 11, 99, 46],
        ]);
        let chk_files: Vec<_> = f.with_checksums(&modsum);
        let mut naive = ModSum::<u64>::with_options();
        naive.width(38);
        let m: Result<Vec<_>, _> = reverse_modsum(&naive, &chk_files, 0, false).collect();
        if let Ok(x) = m {
            assert!(x.contains(&modsum))
        }
    }
    #[test]
    fn error_preset_module() {
        let modsum = ModSum::with_options()
            .width(45)
            .module(75u64)
            .init(38)
            .inendian(Endian::Big)
            .outendian(Endian::Big)
            .wordsize(64)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![
                33, 34, 85, 19, 55, 8, 87, 34, 35, 87, 39, 6, 16, 86, 80, 55, 69, 46, 64, 64, 14,
                17, 25, 59,
            ],
            vec![93, 77, 50, 93, 18, 85, 12, 23, 32, 3, 7, 76, 90, 45, 70, 65],
            vec![93, 26, 87, 27, 97, 36, 78, 48],
        ]);
        let chk_files: Vec<_> = f.with_checksums(&modsum);
        let mut naive = ModSum::<u64>::with_options();
        naive.width(45).module(75);
        let m = reverse_modsum(&naive, &chk_files, 0, false);
        assert!(!f.check_matching(&modsum, m).is_failure())
    }
    #[test]
    fn error3() {
        // caused by bug in factoring algorithm
        let modsum = ModSum::with_options()
            .width(34)
            .module(0x15758e195u64)
            .init(0xd31ee539)
            .inendian(Endian::Big)
            .outendian(Endian::Little)
            .wordsize(32)
            .build()
            .unwrap();
        let f = ReverseFileSet(vec![
            vec![
                13, 172, 74, 40, 206, 163, 7, 169, 20, 194, 253, 171, 168, 190, 255, 187, 150, 56,
                44, 212, 115, 70, 66, 86, 97, 111, 139, 202, 115, 189, 255, 117, 112, 225, 215,
                168, 211, 64, 1, 26, 127, 1, 71, 249, 71, 212, 144, 47, 253, 140, 57, 42, 232, 170,
                62, 240,
            ],
            vec![
                122, 13, 224, 25, 74, 129, 163, 253, 0, 233, 255, 250, 216, 209, 105, 175, 148, 98,
                154, 210, 9, 216, 253, 18, 0, 56, 26, 85, 104, 61, 0, 19, 156, 103, 255, 6, 122,
                230, 106, 5,
            ],
            vec![
                187, 202, 75, 213, 99, 51, 90, 0, 219, 82, 79, 0, 144, 98, 34, 80, 90, 1, 189, 10,
                137, 199, 176, 174,
            ],
        ]);
        let chk_files: Vec<_> = f.with_checksums(&modsum);
        let mut naive = ModSum::<u64>::with_options();
        naive.width(34);
        let m = reverse_modsum(&naive, &chk_files, 0, false);
        assert!(!f.check_matching(&modsum, m).is_failure())
    }
}

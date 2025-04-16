use crate::{
    checksum::CheckReverserError,
    endian::{WordSpec, wordspec_combos},
};

use super::{PolyHash, PolyHashBuilder};

pub fn reverse_polyhash<'a>(
    spec: &PolyHashBuilder<u64>,
    chk_bytes: &[(&'a [u8], Vec<u8>)],
    _verbosity: u64,
    extended_search: bool,
) -> impl Iterator<Item = Result<PolyHash<u64>, CheckReverserError>> + use<'a> {
    let width = spec.width.unwrap();
    let chk_bytes = chk_bytes.to_vec();
    let combos = wordspec_combos(
        spec.wordsize,
        spec.input_endian,
        spec.output_endian,
        width,
        extended_search,
    );
    combos
        .into_iter()
        .flat_map(move |wordspec| reverse(width, &chk_bytes, wordspec))
}

fn reverse<'a>(
    width: usize,
    chk_bytes: &[(&'a [u8], Vec<u8>)],
    wordspec: WordSpec,
) -> impl Iterator<Item = Result<PolyHash<u64>, CheckReverserError>> + use<'a> {
    let revspec = RevSpec {
        polys: chk_bytes
            .iter()
            .map(|x| poly_from_data(width, &wordspec, x))
            .collect(),
        wordspec,
        width,
    };
    find_solutions(revspec.clone()).map(move |solution| {
        Ok(PolyHash::with_options()
            .factor(solution.factor)
            .width(solution.width)
            .inendian(revspec.wordspec.input_endian)
            .outendian(revspec.wordspec.output_endian)
            .wordsize(revspec.wordspec.wordsize)
            .build()
            .unwrap())
    })
}

fn poly_from_data(
    width: usize,
    wordspec: &WordSpec,
    chk_bytes: &(&[u8], Vec<u8>),
) -> WordPolynomial {
    let chk = wordspec.bytes_to_output::<u64>(&chk_bytes.1);
    let mut poly = WordPolynomial {
        coefficients: wordspec.iter_words(chk_bytes.0).rev().collect(),
    };
    poly.shorten(width);
    let sum = WordPolynomial {
        coefficients: vec![chk],
    };
    poly = poly - sum;
    poly
}

#[derive(Clone)]
struct RevSpec {
    wordspec: WordSpec,
    polys: Vec<WordPolynomial>,
    width: usize,
}

struct PartialSolution {
    width: usize,
    factor: u64,
}

fn find_solutions(spec: RevSpec) -> impl Iterator<Item = PartialSolution> {
    let mut partial_solutions = vec![];
    partial_solutions.extend(initial_solution(&spec.polys));

    std::iter::from_fn(move || {
        while let Some(partial_solution) = partial_solutions.pop() {
            if partial_solution.width == spec.width {
                if partial_solution.factor == 1 {
                    continue;
                }
                return Some(partial_solution);
            }

            partial_solutions.extend(lift_solution(&spec.polys, partial_solution));
        }
        None
    })
}

fn initial_solution(polys: &[WordPolynomial]) -> Option<PartialSolution> {
    polys
        .iter()
        .all(|p| p.eval(1) & 1 == 0)
        .then_some(PartialSolution {
            width: 1,
            factor: 1,
        })
}

fn lift_solution(polys: &[WordPolynomial], subsolution: PartialSolution) -> Vec<PartialSolution> {
    let step = 1u64 << subsolution.width;
    let mask = (step << 1).wrapping_sub(1);
    let mut ret = vec![];

    for factor in [subsolution.factor, subsolution.factor + step] {
        if polys.iter().all(|p| p.eval(factor) & mask == 0) {
            ret.push(PartialSolution {
                width: subsolution.width + 1,
                factor,
            });
        }
    }

    ret
}

#[derive(Clone, Debug)]
struct WordPolynomial {
    coefficients: Vec<u64>,
}

fn add_swap(x: &mut u64, a: u64) -> u64 {
    let old = *x;
    *x = x.wrapping_add(a);
    old
}

impl WordPolynomial {
    // multiplies with (x + factor)
    fn linear_mul(&mut self, factor: u64) {
        self.coefficients.insert(0, 0);
        let Some((&mut mut cur, others)) = self.coefficients.split_last_mut() else {
            return;
        };
        for coeff in others.iter_mut().rev() {
            cur = add_swap(coeff, factor.wrapping_mul(cur));
        }
    }

    fn one() -> Self {
        Self {
            coefficients: vec![1],
        }
    }

    fn zero_poly(width: usize) -> Self {
        let mut cur = Self::one();
        let mut i = 1;
        let mut zero_bits = 0;
        while zero_bits < width {
            cur.linear_mul(i);
            zero_bits += i.trailing_zeros() as usize;
            i += 1;
        }
        cur
    }

    fn eval(&self, x: u64) -> u64 {
        let mut result = 0u64;
        for coeff in self.coefficients.iter().rev() {
            result = result.wrapping_mul(x).wrapping_add(*coeff);
        }
        result
    }

    // returns the degree of self, or None if self is zero
    fn deg(&self) -> Option<usize> {
        self.coefficients.len().checked_sub(
            1 + self
                .coefficients
                .iter()
                .copied()
                .rev()
                .take_while(|x| *x == 0)
                .count(),
        )
    }

    fn reduce(&mut self, other: &Self) {
        let Some(selfdeg) = self.deg() else {
            return;
        };
        let Some(otherdeg) = other.deg() else {
            return;
        };
        if selfdeg < otherdeg {
            return;
        }

        let diff = selfdeg - otherdeg;

        let leading_other = other.coefficients[otherdeg];
        debug_assert_eq!(leading_other, 1, "`other` must be monic");
        for i in (0..=diff).rev() {
            let leading_self = self.coefficients[otherdeg + i];

            for (self_coeff, other_coeff) in self.coefficients[i..]
                .iter_mut()
                .zip(other.coefficients.iter())
            {
                *self_coeff = self_coeff.wrapping_sub(other_coeff.wrapping_mul(leading_self));
            }
        }
        self.coefficients.truncate(otherdeg);
    }

    fn shorten(&mut self, width: usize) {
        let zero = Self::zero_poly(width);
        self.reduce(&zero);
        self.coefficients.shrink_to_fit();
    }
}

impl std::ops::Add for WordPolynomial {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        if self.coefficients.len() < rhs.coefficients.len() {
            std::mem::swap(&mut self, &mut rhs);
        }

        for (a, b) in self.coefficients.iter_mut().zip(rhs.coefficients.iter()) {
            *a = a.wrapping_add(*b);
        }
        self
    }
}

impl std::ops::Sub for WordPolynomial {
    type Output = Self;

    fn sub(mut self, mut rhs: Self) -> Self::Output {
        if self.coefficients.len() < rhs.coefficients.len() {
            std::mem::swap(&mut self, &mut rhs);
        }

        for (a, b) in self.coefficients.iter_mut().zip(rhs.coefficients.iter()) {
            *a = a.wrapping_sub(*b);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::{Arbitrary, Gen, TestResult};

    use crate::{checksum::tests::ReverseFileSet, endian::Endian};

    use super::*;

    impl Arbitrary for WordPolynomial {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let coeffs = Vec::<u64>::arbitrary(g);
            Self {
                coefficients: coeffs,
            }
        }
    }

    fn mask(x: u64, width: u8) -> u64 {
        if width >= 64 { x } else { x % (1 << width) }
    }

    impl Arbitrary for PolyHashBuilder<u64> {
        fn arbitrary(g: &mut Gen) -> Self {
            let mut new_polyhash = PolyHash::with_options();
            let width = u8::arbitrary(g) % 63 + 2;
            new_polyhash.width(width as usize);
            let mut factor = mask(u64::arbitrary(g) << 1 | 1, width);
            if factor == 1 {
                factor = 3;
            }
            new_polyhash.factor(factor);
            let wordspec = WordSpec::arbitrary(g);
            let max_word_width = ((width as usize).div_ceil(8)).next_power_of_two() * 8;
            new_polyhash.wordsize(max_word_width.min(wordspec.wordsize));
            new_polyhash.inendian(wordspec.input_endian);
            new_polyhash.outendian(wordspec.output_endian);
            new_polyhash
        }
    }

    fn run_polyhash_rev(
        files: ReverseFileSet,
        polyhash_build: PolyHashBuilder<u64>,
        known: (bool,),
        wordspec_known: (bool, bool, bool),
    ) -> TestResult {
        let polyhash = polyhash_build.build().unwrap();
        let mut naive = PolyHash::<u64>::with_options();
        naive.width(polyhash_build.width.unwrap());
        if known.0 {
            naive.factor(polyhash_build.factor.unwrap());
        }
        if wordspec_known.0 {
            naive.wordsize(polyhash_build.wordsize.unwrap());
        }
        if wordspec_known.1 {
            naive.inendian(polyhash_build.input_endian.unwrap());
        }
        if wordspec_known.2 {
            naive.outendian(polyhash_build.output_endian.unwrap());
        }
        let chk_files: Vec<_> = files.with_checksums(&polyhash);
        let reverser = reverse_polyhash(&naive, &chk_files, 0, false);
        files.check_matching(&polyhash, reverser)
    }

    #[quickcheck]
    fn qc_polyhash_rev(
        files: ReverseFileSet,
        polyhash_build: PolyHashBuilder<u64>,
        known: (bool,),
        wordspec_known: (bool, bool, bool),
    ) -> TestResult {
        run_polyhash_rev(files, polyhash_build, known, wordspec_known)
    }

    #[test]
    fn error1() {
        let files = ReverseFileSet(vec![vec![0, 18, 232, 236, 87, 255, 203, 100], vec![]]);
        let mut polyhash_build = PolyHash::with_options();
        polyhash_build
            .width(7)
            .factor(47)
            .inendian(Endian::Little)
            .outendian(Endian::Big)
            .wordsize(8);
        assert!(
            !run_polyhash_rev(files, polyhash_build, (false,), (false, false, false)).is_failure()
        );
    }

    #[test]
    fn eval() {
        let poly = WordPolynomial {
            coefficients: vec![
                0x538107cc2cc3d3cd,
                0x53ad12105a37ea91,
                0xf834621210ff5545,
                0x4c0300d048fedcef,
                0x86f111dc9eb71158,
            ],
        };
        let res = poly.eval(0xa0f4946c408448b2);
        assert_eq!(res, 0x5d4cc07b361e7f2b);
    }

    #[test]
    fn zero_polynomials() {
        for width in 0..=16 {
            let zero = WordPolynomial::zero_poly(width);
            let mask = (1 << width) - 1;
            for i in 0..=mask {
                let res = zero.eval(i);
                assert_eq!(res & mask, 0);
            }
        }
    }

    #[test]
    fn rem() {
        let mut p = WordPolynomial {
            coefficients: vec![
                0x2af98107c994f2dc,
                0x130fc109ec31962,
                0x6acc95f45db55397,
                0xb928859e5971a22,
                0xd26e5b00b83f82ce,
            ],
        };
        let q = WordPolynomial {
            coefficients: vec![0xd1b714812043a8af, 0x7c52dfef8e18dbd9, 0x1],
        };
        p.reduce(&q);
        assert_eq!(p.coefficients, [0x759503c87242760d, 0x22f475b3698da76d]);
    }

    #[quickcheck]
    fn shorten_preserves_map(p: WordPolynomial, point: u16) {
        let mut shortened = p.clone();
        shortened.shorten(16);
        assert!(shortened.deg() <= Some(17));
        let a = p.eval(point as u64);
        let b = shortened.eval(point as u64);
        assert_eq!(a & 0xffff, b & 0xffff);
    }
}

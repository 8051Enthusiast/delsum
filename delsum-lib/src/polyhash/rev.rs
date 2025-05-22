use rayon::iter::ParallelIterator;
use rayon::prelude::*;

use crate::{
    checksum::{CheckReverserError, Checksum, filter_opt_err},
    endian::{WordSpec, wordspec_combos},
    utils::unresult_iter,
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
        spec.signedness,
        width,
        extended_search,
    );
    let factor = spec.factor;
    let init = spec.init;
    let addout = spec.addout;
    filter_opt_err(combos.into_iter().flat_map(move |wordspec| {
        unresult_iter(reverse(width, &chk_bytes, factor, init, addout, wordspec))
    }))
}

pub fn reverse_polyhash_para<'a>(
    spec: &PolyHashBuilder<u64>,
    chk_bytes: &[(&'a [u8], Vec<u8>)],
    _verbosity: u64,
    extended_search: bool,
) -> impl ParallelIterator<Item = Result<PolyHash<u64>, CheckReverserError>> + use<'a> {
    let width = spec.width.unwrap();
    let chk_bytes = chk_bytes.to_vec();
    let combos = wordspec_combos(
        spec.wordsize,
        spec.input_endian,
        spec.output_endian,
        spec.signedness,
        width,
        extended_search,
    );
    let factor = spec.factor;
    let init = spec.init;
    let addout = spec.addout;
    combos.into_par_iter().flat_map(move |wordspec| {
        filter_opt_err(unresult_iter(reverse(
            width, &chk_bytes, factor, init, addout, wordspec,
        )))
        .par_bridge()
    })
}

// The way the init/addout parameters are cancelled out in the case of polyhash
// is similar to how they are cancelled out in the other modules.
// However, here we simply lift the solution to higher moduli 2^i iteratively
// since we do not have to find the modulus itself (it is already given as 2^width).
//
// This also allows us to calculate our polynomials modulo so-called zero-polynomials
// that are zero on every point. This makes our polynomials radically shorter (think degree
// 32 or 64), without changing the result.
fn reverse<'a>(
    width: usize,
    chk_bytes: &[(&'a [u8], Vec<u8>)],
    factor: Option<u64>,
    init: Option<u64>,
    addout: Option<u64>,
    wordspec: WordSpec,
) -> Result<impl Iterator<Item = PolyHash<u64>> + use<'a>, Option<CheckReverserError>> {
    let Some(mut polys): Option<Vec<_>> = chk_bytes
        .iter()
        .map(|x| poly_from_data(width, &wordspec, x))
        .collect()
    else {
        return Err(None);
    };

    polys.sort();
    polys.dedup();

    if let Some(value) = check_params(chk_bytes, factor, init, addout) {
        return Err(Some(value));
    }

    let (filtered, addout_info) = filter_addouts(polys, addout);
    let (filtered, init_info) = filter_inits(filtered, init, width);

    let revspec = RevSpec {
        polys: filtered,
        width,
        init: init_info,
        addout: addout_info,
        factor,
    };

    Ok(find_solutions(revspec, wordspec))
}

fn complete_solution(
    revspec: &RevSpec,
    solution: PartialSolution,
    wordspec: WordSpec,
) -> Option<PolyHash<u64>> {
    let addout = match &revspec.addout {
        ParamSource::Given(addout) => *addout,
        ParamSource::Files(files) => find_addout(files, solution.factor, solution.init)?,
    };
    Some(
        PolyHash::with_options()
            .factor(solution.factor)
            .init(solution.init)
            .addout(addout)
            .width(solution.width)
            .inendian(wordspec.input_endian)
            .outendian(wordspec.output_endian)
            .wordsize(wordspec.wordsize)
            .signedness(wordspec.signedness)
            .build()
            .unwrap(),
    )
}

fn find_addout(file_polys: &[FilePoly], factor: u64, init: u64) -> Option<u64> {
    let (first, rest) = file_polys.split_first()?;
    let addout = find_single_addout(first, factor, init);
    rest.iter()
        .all(|p| find_single_addout(p, factor, init) == addout)
        .then_some(addout)
}

fn find_single_addout(p: &FilePoly, factor: u64, init: u64) -> u64 {
    let mask = mask_val(p.width as u8);
    p.eval_with_init(factor, init).wrapping_neg() & mask
}

fn check_params(
    chk_bytes: &[(&[u8], Vec<u8>)],
    factor: Option<u64>,
    init: Option<u64>,
    addout: Option<u64>,
) -> Option<CheckReverserError> {
    if 3 > chk_bytes.len()
        + init.is_some() as usize
        + factor.is_some() as usize
        + addout.is_some() as usize
    {
        return Some(CheckReverserError::MissingParameter(
            "need at least 3 files + parameters (init, factor, addout)",
        ));
    }

    if init.is_none()
        && chk_bytes.iter().map(|x| x.0.len()).max() == chk_bytes.iter().map(|x| x.0.len()).min()
    {
        return Some(CheckReverserError::UnsuitableFiles(
            "need at least one file with different length (or set init to 0)",
        ));
    }
    None
}

fn poly_from_data(
    width: usize,
    wordspec: &WordSpec,
    chk_bytes: &(&[u8], Vec<u8>),
) -> Option<FilePoly> {
    let chk = Checksum::from_bytes(&chk_bytes.1, wordspec.output_endian, width)?;
    let mut poly = WordPolynomial {
        coefficients: wordspec
            .iter_words(chk_bytes.0)
            .rev()
            .map(|x| {
                if x.negative {
                    x.value.wrapping_neg()
                } else {
                    x.value
                }
            })
            .collect(),
    };
    let size = poly.coefficients.len();
    poly.shorten(width);

    let sum = WordPolynomial {
        coefficients: vec![chk],
    };
    poly = poly - sum;

    Some(FilePoly {
        poly,
        init: InitPlace::Single(size),
        width,
    })
}

#[derive(Clone)]
struct RevSpec {
    polys: Vec<WordPolynomial>,
    width: usize,
    factor: Option<u64>,
    init: ParamSource,
    addout: ParamSource,
}

fn filter_addouts(original: Vec<FilePoly>, addout: Option<u64>) -> (Vec<FilePoly>, ParamSource) {
    if let Some(addout) = addout {
        let clean = original
            .into_iter()
            .map(|x| x.remove_addout(addout))
            .collect();
        return (clean, ParamSource::Given(addout));
    }

    let mut result = vec![];
    for x in original.windows(2) {
        let [lhs, rhs] = x else { unreachable!() };
        let (InitPlace::Single(lhs_len), InitPlace::Single(rhs_len)) = (lhs.init, rhs.init) else {
            unreachable!();
        };

        let init = if lhs_len == rhs_len {
            InitPlace::None
        } else {
            InitPlace::Duo(lhs_len, rhs_len)
        };
        let poly = rhs.poly.clone() - lhs.poly.clone();

        result.push(FilePoly {
            poly,
            init,
            width: lhs.width,
        })
    }
    (result, ParamSource::Files(original))
}

fn filter_inits(
    original: Vec<FilePoly>,
    init: Option<u64>,
    width: usize,
) -> (Vec<WordPolynomial>, ParamSource) {
    if let Some(init) = init {
        let clean = original.into_iter().map(|x| x.remove_init(init)).collect();
        return (clean, ParamSource::Given(init));
    }

    let mut init_free = vec![];
    let mut initiferous = vec![];
    for x in original {
        match x.init {
            InitPlace::None => init_free.push(x.poly),
            _ => initiferous.push(x),
        }
    }
    for x in initiferous.windows(2) {
        let [lhs, rhs] = x else { unreachable!() };

        match (lhs.init, rhs.init) {
            (InitPlace::Single(l), InitPlace::Single(r)) => {
                let diff = r - l;
                let mut cancelled = rhs.poly.clone() - (lhs.poly.clone() << diff);
                cancelled.shorten(width);
                init_free.push(cancelled);
            }
            (InitPlace::Duo(ll, lr), InitPlace::Duo(rl, rr)) => {
                let ldiff = lr - ll;
                let rdiff = rr - rl;
                let lhs = (lhs.poly.clone() << rdiff) - lhs.poly.clone();
                let rhs = (rhs.poly.clone() << ldiff) - rhs.poly.clone();
                let diff = rl - ll;
                let mut cancelled = rhs - (lhs << diff);
                cancelled.shorten(width);
                init_free.push(cancelled);
            }
            (InitPlace::None | InitPlace::Duo(..), _)
            | (_, InitPlace::None | InitPlace::Duo(..)) => unreachable!(),
        }
    }

    (init_free, ParamSource::Files(initiferous))
}

#[derive(Debug, Clone)]
enum ParamSource {
    Given(u64),
    Files(Vec<FilePoly>),
}

impl ParamSource {
    fn matches_init(&self, factor: u64, init: u64, mask: u64) -> bool {
        match self {
            ParamSource::Given(given_init) => init & mask == given_init & mask,
            ParamSource::Files(items) => items.iter().all(|p| {
                p.poly
                    .eval(factor)
                    .wrapping_add(p.init.init_value(factor, init))
                    & mask
                    == 0
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FilePoly {
    init: InitPlace,
    poly: WordPolynomial,
    width: usize,
}

impl FilePoly {
    fn remove_init(self, init: u64) -> WordPolynomial {
        match self.init {
            InitPlace::None => self.poly,
            InitPlace::Single(length) => {
                let mut result = self.poly + (WordPolynomial::constant(init) << length);
                result.shorten(self.width);
                result
            }
            InitPlace::Duo(llength, rlength) => {
                let mut result = self.poly + (WordPolynomial::constant(init) << rlength)
                    - (WordPolynomial::constant(init) << llength);
                result.shorten(self.width);
                result
            }
        }
    }

    fn remove_addout(self, addout: u64) -> FilePoly {
        let poly = self.poly + WordPolynomial::constant(addout);
        FilePoly { poly, ..self }
    }

    fn eval_with_init(&self, factor: u64, init: u64) -> u64 {
        self.poly
            .eval(factor)
            .wrapping_add(self.init.init_value(factor, init))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum InitPlace {
    None,
    Single(usize),
    Duo(usize, usize),
}

impl InitPlace {
    fn init_value(self, factor: u64, init: u64) -> u64 {
        match self {
            InitPlace::None => 0,
            InitPlace::Single(pos) => pow(factor, pos).wrapping_mul(init),
            InitPlace::Duo(posl, posr) => pow(factor, posr)
                .wrapping_sub(pow(factor, posl))
                .wrapping_mul(init),
        }
    }
}

#[derive(Debug)]
struct PartialSolution {
    width: usize,
    factor: u64,
    init: u64,
}

impl PartialSolution {
    fn matches_spec(&self, spec: &RevSpec) -> bool {
        let mask = mask_val(self.width as u8);
        if let Some(factor) = spec.factor {
            if factor & mask != self.factor & mask {
                return false;
            }
        }
        if is_poly_solution(&spec.polys, self.factor, mask)
            && spec.init.matches_init(self.factor, self.init, mask)
        {
            true
        } else {
            false
        }
    }
}

fn find_solutions(spec: RevSpec, wordspec: WordSpec) -> impl Iterator<Item = PolyHash<u64>> {
    let mut partial_solutions = vec![];
    partial_solutions.extend(
        INITIAL_SOLUTIONS
            .into_iter()
            .filter(|x| x.matches_spec(&spec)),
    );

    std::iter::from_fn(move || {
        while let Some(partial_solution) = partial_solutions.pop() {
            if partial_solution.width == spec.width {
                if partial_solution.factor != 1 {
                    if let Some(solution) = complete_solution(&spec, partial_solution, wordspec) {
                        return Some(solution);
                    }
                }
            } else {
                partial_solutions.extend(lift_solution(&spec, partial_solution));
            }
        }
        None
    })
}

const INITIAL_SOLUTIONS: [PartialSolution; 2] = [
    PartialSolution {
        width: 1,
        factor: 1,
        init: 0,
    },
    PartialSolution {
        width: 1,
        factor: 1,
        init: 1,
    },
];

fn lift_solution(
    spec: &RevSpec,
    subsolution: PartialSolution,
) -> impl Iterator<Item = PartialSolution> {
    let PartialSolution {
        width,
        factor,
        init,
    } = subsolution;
    let step = 1u64 << subsolution.width;
    [
        (factor, init),
        (factor + step, init),
        (factor, init + step),
        (factor + step, init + step),
    ]
    .into_iter()
    .map(move |(factor, init)| PartialSolution {
        width: width + 1,
        factor,
        init,
    })
    .filter(|subsolution| subsolution.matches_spec(spec))
}

fn is_poly_solution(polys: &[WordPolynomial], factor: u64, mask: u64) -> bool {
    polys.iter().all(|p| p.eval(factor) & mask == 0)
}

fn pow(mut base: u64, mut exp: usize) -> u64 {
    if exp == 0 {
        return 1;
    }
    let mut acc: u64 = 1;
    loop {
        if (exp & 1) == 1 {
            acc = acc.wrapping_mul(base);
            if exp == 1 {
                return acc;
            }
        }
        exp /= 2;
        base = base.wrapping_mul(base);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
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

    fn constant(word: u64) -> WordPolynomial {
        Self {
            coefficients: vec![word],
        }
    }

    fn one() -> Self {
        Self::constant(1)
    }

    // the zero polynomial of width `width` is just the
    // rising factorial of the width.
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

    // naive polynomial remainder
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

    // calculates self %= zero_poly to make it shorter
    // without changing self as a mapping
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
        let swapped = if self.coefficients.len() < rhs.coefficients.len() {
            std::mem::swap(&mut self, &mut rhs);
            true
        } else {
            false
        };

        for (a, b) in self.coefficients.iter_mut().zip(rhs.coefficients.iter()) {
            *a = a.wrapping_sub(*b);
        }
        if swapped {
            self.coefficients
                .iter_mut()
                .for_each(|x| *x = x.wrapping_neg());
        }
        self
    }
}

impl std::ops::Shl<usize> for WordPolynomial {
    type Output = Self;

    fn shl(mut self, rhs: usize) -> Self::Output {
        if rhs == 0 {
            return self;
        }

        let mut coeffs = vec![0; rhs];
        coeffs.append(&mut self.coefficients);
        self.coefficients = coeffs;
        self
    }
}

fn mask_val(width: u8) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::{Arbitrary, Gen, TestResult};

    use crate::{
        checksum::tests::ReverseFileSet,
        endian::{Endian, Signedness},
    };

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
        x & mask_val(width)
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
            let init = mask(u64::arbitrary(g), width);
            new_polyhash.init(init);
            let addout = mask(u64::arbitrary(g), width);
            new_polyhash.addout(addout);
            let wordspec = WordSpec::arbitrary(g);
            let max_word_width = ((width as usize).div_ceil(8)).next_power_of_two() * 8;
            new_polyhash.wordsize(max_word_width.min(wordspec.wordsize));
            new_polyhash.inendian(wordspec.input_endian);
            new_polyhash.outendian(wordspec.output_endian);
            new_polyhash.signedness(wordspec.signedness);
            new_polyhash
        }
    }

    fn run_polyhash_rev(
        files: ReverseFileSet,
        polyhash_build: PolyHashBuilder<u64>,
        known: (bool, bool, bool),
        wordspec_known: (bool, bool, bool, bool),
    ) -> TestResult {
        let polyhash = polyhash_build.build().unwrap();
        let mut naive = PolyHash::<u64>::with_options();
        naive.width(polyhash_build.width.unwrap());
        if known.0 {
            naive.factor(polyhash_build.factor.unwrap());
        }
        if known.1 {
            naive.init(polyhash_build.init.unwrap());
        }
        if known.2 {
            naive.addout(polyhash_build.addout.unwrap());
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
        if wordspec_known.3 {
            naive.signedness(polyhash_build.signedness.unwrap());
        }
        let chk_files: Vec<_> = files.with_checksums(&polyhash);
        let reverser = reverse_polyhash(&naive, &chk_files, 0, false);
        files.check_matching(&polyhash, reverser)
    }

    #[quickcheck]
    fn qc_polyhash_rev(
        files: ReverseFileSet,
        polyhash_build: PolyHashBuilder<u64>,
        known: (bool, bool, bool),
        wordspec_known: (bool, bool, bool, bool),
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
            .addout(0)
            .inendian(Endian::Little)
            .outendian(Endian::Big)
            .signedness(Signedness::Unsigned)
            .wordsize(8);
        let res = run_polyhash_rev(
            files,
            polyhash_build,
            (false, false, true),
            (false, false, false, false),
        );
        assert!(!res.is_failure());
    }

    #[test]
    fn error2() {
        let files = ReverseFileSet(vec![
            vec![
                11, 64, 11, 236, 49, 191, 139, 194, 99, 52, 58, 24, 92, 159, 147, 137, 143, 208,
                88, 235, 210, 48, 91, 21, 245, 211, 0, 255, 29, 255, 156, 185, 137, 81, 79, 166,
                174, 248, 0, 16, 18, 137, 123, 92, 244, 197, 194, 101, 85, 255, 92, 113, 6, 133,
                145, 204, 220, 117, 8, 255, 86, 113, 251, 255, 229, 0, 128, 161, 152, 198, 51, 0,
                83, 232, 192, 117, 136, 24, 107, 214, 160, 1, 155, 255, 125, 5, 112, 35,
            ],
            vec![
                171, 11, 130, 90, 14, 244, 199, 255, 1, 117, 108, 139, 151, 228, 94, 125, 40, 104,
                134, 120, 155, 251, 94, 152, 156, 55, 133, 162, 110, 12, 26, 34, 250, 33, 107, 165,
                121, 249, 183, 48,
            ],
            vec![
                139, 155, 73, 251, 44, 63, 174, 79, 2, 12, 255, 63, 34, 165, 126, 37, 102, 227, 71,
                164, 182, 0, 130, 165,
            ],
        ]);

        let mut polyhash_build = PolyHash::with_options();
        polyhash_build
            .width(60)
            .factor(177519018992307695)
            .inendian(Endian::Little)
            .outendian(Endian::Little)
            .signedness(Signedness::Unsigned)
            .wordsize(16);
        assert!(
            !run_polyhash_rev(
                files,
                polyhash_build,
                (false, false, false),
                (false, false, false, false)
            )
            .is_failure()
        );
    }

    #[test]
    fn error3() {
        let files = ReverseFileSet(vec![
            vec![
                87, 65, 72, 201, 246, 255, 75, 207, 1, 15, 110, 87, 135, 244, 208, 46, 77, 222,
                112, 151, 158, 26, 209, 154, 137, 3, 210, 234, 124, 187, 113, 2, 103, 48, 237, 66,
                97, 20, 189, 182,
            ],
            vec![
                140, 255, 249, 103, 77, 57, 255, 193, 218, 115, 17, 99, 89, 1, 166, 43, 10, 151,
                56, 72, 149, 255, 142, 86, 254, 132, 168, 162, 2, 255, 10, 127,
            ],
            vec![],
        ]);

        let mut polyhash_build = PolyHash::with_options();
        polyhash_build
            .width(42)
            .factor(0x3fa691cbcb1)
            .init(0x2b3ac03d788)
            .inendian(Endian::Little)
            .outendian(Endian::Big)
            .signedness(Signedness::Unsigned)
            .wordsize(64);
        assert!(
            !run_polyhash_rev(
                files,
                polyhash_build,
                (true, false, false),
                (true, true, true, true)
            )
            .is_failure()
        );
    }

    #[test]
    fn error4() {
        let files = ReverseFileSet(vec![vec![49, 102, 242, 157, 81, 134, 181, 69], vec![]]);

        let mut polyhash_build = PolyHash::with_options();
        polyhash_build
            .width(41)
            .factor(0xb614134799)
            .init(0x14f8d8416e7)
            .inendian(Endian::Little)
            .outendian(Endian::Little)
            .signedness(Signedness::Unsigned)
            .wordsize(64);
        assert!(
            !run_polyhash_rev(
                files,
                polyhash_build,
                (false, true, false),
                (true, true, true, true)
            )
            .is_failure()
        );
    }

    #[test]
    fn error5() {
        let files = ReverseFileSet(vec![vec![1, 0], vec![]]);

        let mut polyhash_build = PolyHash::with_options();
        polyhash_build
            .width(16)
            .factor(0x103)
            .addout(0)
            .init(1)
            .inendian(Endian::Big)
            .outendian(Endian::Little)
            .signedness(Signedness::Unsigned)
            .wordsize(16);
        assert!(
            !run_polyhash_rev(
                files,
                polyhash_build,
                (true, true, true),
                (true, false, true, true)
            )
            .is_failure()
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

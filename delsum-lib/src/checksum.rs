use crate::endian::WordSpec;
use crate::utils::SignedInclRange;
use crate::utils::UnsignedInclRange;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt::Debug;

/// A basic trait for a checksum where
/// * init gives an initial state
/// * dig_word processes a single word
/// * finalize is applied to get the final sum after all words are processed.
///
/// They should be implemented in a way such that the digest default implementation
/// corresponds to calculating the checksum.
///
/// Unlike LinearCheck, it is not really required to be linear yet, but in
/// context of this application, there is really no use only implementing this.
pub trait Digest {
    /// A type that holds the checksum.
    ///
    /// Note that in this application, a separate state type that holds the interal state
    /// and gets converted to a Sum by finalize
    /// is not really feasable because of the operations LinearCheck would need to do
    /// both on Sums and interal States, so a single Sum type must be enough.
    type Sum: Clone + Eq + Ord + Debug + Send + Sync + Checksum;
    /// Gets an initial sum before the words are processed through the sum.
    ///
    /// For instance in the case of crc, the sum type is some integer and the returned value from
    /// this function could be 0xffffffff (e.g. in the bzip2 crc).
    fn init(&self) -> Self::Sum;
    /// Processes a single word from the text.
    ///
    /// For a crc, this corresponds to shifting, adding the word and reducing.
    fn dig_word(&self, sum: Self::Sum, word: u64) -> Self::Sum;
    /// After all words are read, this function is called to do some finalization.
    ///
    /// In the case of crc, this corresponds to adding a constant at the end
    /// (and maybe also adding some 0s to the end of the text).
    fn finalize(&self, sum: Self::Sum) -> Self::Sum;
    /// Takes the sum and turns it into an array of bytes (may depend on configured endian)
    fn to_bytes(&self, s: Self::Sum) -> Vec<u8>;
    /// Iterate over the words of a file so that digest calculates the checksum
    fn wordspec(&self) -> WordSpec;
    /// Takes a reader and calculates the checksums of all words therein.
    fn digest(&self, buf: &[u8]) -> Result<Self::Sum, std::io::Error> {
        let wordspec = self.wordspec();
        if buf.len() % wordspec.word_bytes() != 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        }
        let sum = wordspec
            .iter_words(buf)
            .fold(self.init(), |c, s| self.dig_word(c, s));
        Ok(self.finalize(sum))
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Relativity {
    Start,
    End,
}

impl From<SignedInclRange> for Relativity {
    fn from(r: SignedInclRange) -> Self {
        if r.start() < 0 {
            Relativity::End
        } else {
            Relativity::Start
        }
    }
}

/// A checksum that also has some notion of linearity.
///
/// What does linearity mean here? In a mathematically pure world, it would mean
/// that you could add the texts in some way (XOR for crc) and that would be the
/// same as adding (XORing) both checksums.
/// However, we live in a world that needs practical considerations, so it's not as clean.
/// Mostly, this is skewed by `init` and `finalize`.
///
/// This trait adds another type, the `Shift` type.
/// This acts, when applied to an unfinalized sum in the `shift` function, as if appending
/// `n` 0s to the summed text. For example, in a Fletcher sum, this would simply be an integer
/// containing `n` and applying the shift corresponds to adding the first sum `n` times to the second one, possible in
/// constant time. However, in the crc case, this is not possible in constant time just using
/// the integer containing `n`. In this case, the value of of `x^{8n}` reduced by the generator is stored
/// and the shift is applied using polynomial multiplication modulo the generator.
///
/// The assumptions are here (the `self`s are omitted for clarity):
/// * `add(a,b)` forms an abeliean group with `negate(a)` as inverse (hereafter, the sum value 0 will be equal to `add(init(), negate(init()))`)
/// * `shift(s, shift_n(1)) == dig(s, 0u8)`
/// * `shift(s, shift_n(1))` is bijective in the set of all valid `Sum` values
/// * `shift(shift(s, shift_n(a)), shift_n(b)) == shift(s, shift_n(a+b))`
/// * `add(dig_word(s, 0), dig_word(r, 0)) == dig_word(add(s, r), 0)`
/// * `dig_word(s, k) == dig_word(0, k) + dig_word(s, 0)` (consequently, `dig_word(0, 0) == 0`)
/// * for all sums `s`, `add(finalize(s), negate(s))` is constant (finalize adds a constant value to the sum)
/// * all methods without default implementations (including those from `Digest`) should run in constant time (assuming constant `Shift`, `Sum` types)
///
/// Basically, it is a graded ring or something idk.
pub trait LinearCheck: Digest + Send + Sync {
    /// The Shift type (see trait documentation for more).
    type Shift: Clone;
    /// The initial shift corresponding to the identity shift of 0 (see trait documentation for more).
    fn init_shift(&self) -> Self::Shift;
    /// Increments the shift by one (see trait documentation for more).
    fn inc_shift(&self, shift: Self::Shift) -> Self::Shift;
    /// Applies a shift to a sum (see trait documentation for more).
    fn shift(&self, sum: Self::Sum, shift: &Self::Shift) -> Self::Sum;
    /// Adds two sums together (see trait documentation for more).
    fn add(&self, sum_a: Self::Sum, sum_b: &Self::Sum) -> Self::Sum;
    /// Gets inverse in the abelian group of `add` (see trait documentation for more).
    fn negate(&self, sum: Self::Sum) -> Self::Sum;
    /// Acts as if applying `dig_word(s, 0)` `n` times to to `s` (see trait documentation for more).
    ///
    /// Please implement more efficient (equivalent) implementation for each type if possible.
    fn shift_n(&self, n: usize) -> Self::Shift {
        let mut shift = self.init_shift();
        for _ in 0..n {
            shift = self.inc_shift(shift);
        }
        shift
    }

    /// Given some bytes and a target sum, determines all segments in the bytes that have that
    /// particular checksum.
    ///
    /// Each element of the return value contains a tuple consisting of an array of possible segment starts
    /// and an array of possible segment ends. If there are multiple starts or ends, each possible combination
    /// has the target checksum.
    ///
    /// This function has a high space usage per byte: for `n` bytes, it uses a total space of `n*(8 + 2*sizeof(Sum))` bytes.
    /// The time is bounded by the runtime of the sort algorithm, which is around `n*log(n)`.
    /// If Hashtables were used, it could be done in linear time, but they take too much space.
    fn find_segments(
        &self,
        bytes: &[Vec<u8>],
        sum: &[Self::Sum],
        rel: Relativity,
    ) -> Vec<RangePair> {
        if bytes.is_empty() {
            return Vec::new();
        }
        let min_len = bytes.iter().map(|x| x.len()).min().unwrap();
        let start_range = SignedInclRange::new(0, (min_len - 1) as isize);
        let end_range = match rel {
            Relativity::Start => SignedInclRange::new(0, (min_len - 1) as isize),
            Relativity::End => SignedInclRange::new(-(min_len as isize), -1),
        };
        let (start_range, end_range) = match (start_range, end_range) {
            (Some(s), Some(e)) => (s, e),
            (None, _) | (_, None) => return Vec::new(),
        };
        if u32::try_from(bytes[0].len()).is_err() {
            // only support 32-bit length files for now, since a usize for every byte would take a lot of space
            panic!("File must be under 4GiB!");
        }
        self.find_segments_range(bytes, sum, start_range, end_range)
    }

    fn find_segments_range(
        &self,
        bytes: &[Vec<u8>],
        sum: &[Self::Sum],
        start_range: SignedInclRange,
        end_range: SignedInclRange,
    ) -> Vec<RangePair> {
        let mut ret = Vec::new();
        let min_len = bytes.iter().map(|x| x.len()).min().unwrap();
        let step = self.wordspec().word_bytes();

        if Relativity::from(start_range) != Relativity::from(end_range)
            && bytes
                .windows(2)
                .map(|x| x[0].len() % step != x[1].len() % step)
                .any(bool::from)
        {
            // in case one of the ranges is relative to the start and the other relative to the end,
            // and there are two files which have a length difference that is not a multiple of the step
            // lengths, then any checksum on the first file that is a multiple of `step` in length would have
            // a corresponding checksum over a length that is not a multiple of `step`, therefore we return
            // here early since this case isn't actually caught by the general code and would result
            // in unexpected results
            return Vec::new();
        }

        // limit start and end range to actual offsets lying within the smallest file
        let (start_range, end_range) = match (start_range.limit(min_len), end_range.limit(min_len))
        {
            (None, _) | (_, None) => return Vec::new(),
            (Some(start), Some(end)) => (start, end),
        };
        for offset in 0..step {
            let current_start_range =
                match start_range.set_start(start_range.start() + offset as isize) {
                    Some(x) => x,
                    None => break,
                };
            ret.append(
                &mut find_segments_aligned(self, bytes, sum, current_start_range, end_range)
                    .unwrap_or_else(Vec::new),
            );
        }
        ret.sort_unstable();
        ret
    }
}

fn find_segments_aligned<S: LinearCheck + ?Sized>(
    summer: &S,
    bytes: &[Vec<u8>],
    sum: &[<S as Digest>::Sum],
    start_range: SignedInclRange,
    end_range: SignedInclRange,
) -> Option<Vec<RangePair>> {
    let min_len = bytes.iter().map(|x| x.len()).min().unwrap();
    let (start_range, end_range) = normalize_range(
        start_range,
        end_range,
        summer.wordspec().word_bytes(),
        min_len,
    )?;
    #[cfg(feature = "parallel")]
    let (start_presums, end_presums) = bytes
        .par_iter()
        .zip(sum.par_iter())
        .map(|(b, s)| {
            presums(
                summer,
                b,
                s,
                // since they are already normalized, this should work
                start_range.to_unsigned(b.len()).unwrap(),
                end_range.to_unsigned(b.len()).unwrap(),
            )
        })
        .unzip();
    #[cfg(not(feature = "parallel"))]
    let (start_presums, end_presums) = bytes
        .iter()
        .zip(sum.iter())
        .map(|(b, s)| {
            presums(
                summer,
                b,
                &s,
                // since they are already normalized, this should work
                start_range.to_unsigned(b.len()).unwrap(),
                end_range.to_unsigned(b.len()).unwrap(),
            )
        })
        .unzip();
    let start_preset = PresumSet::new(start_presums);
    let end_preset = PresumSet::new(end_presums);
    let mut ret_vec = Vec::new();

    let step = summer.wordspec().word_bytes();
    let least_start_range_start = start_range.to_unsigned(min_len)?.start();
    let least_end_range_start = end_range.to_unsigned(min_len)?.start();
    for (a, b) in start_preset.equal_pairs(&end_preset) {
        let starts: Vec<_> = a
            .iter()
            .map(|x| usize::try_from(*x).unwrap() * step + least_start_range_start)
            .collect();
        let ends: Vec<_> = b
            .iter()
            .map(|x| usize::try_from(*x).unwrap() * step + least_end_range_start)
            .collect();
        let min_start = *starts.iter().min().unwrap_or(&min_len);
        let max_end = *ends.iter().max().unwrap_or(&0);
        let rel_ends: Vec<_> = ends
            .into_iter()
            .filter(|x| x > &min_start)
            .map(|x| match end_range.into() {
                Relativity::Start => isize::try_from(x).unwrap(),
                Relativity::End => -isize::try_from(min_len - x).unwrap(),
            })
            .collect();
        let rel_starts = starts
            .into_iter()
            .filter(|x| x < &max_end)
            .map(|x| match start_range.into() {
                Relativity::Start => isize::try_from(x).unwrap(),
                Relativity::End => -isize::try_from(min_len - x).unwrap(),
            })
            .collect();
        if !rel_ends.is_empty() {
            ret_vec.push((rel_starts, rel_ends));
        }
    }
    Some(ret_vec)
}

// this takes care of shortening the ranges so that the least presums are calculated,
// this is done before calling presums because presums does not know the lengths of the
// other files and we might get different lengths for different files
fn normalize_range(
    mut start_range: SignedInclRange,
    mut end_range: SignedInclRange,
    step: usize,
    min_len: usize,
) -> Option<(SignedInclRange, SignedInclRange)> {
    let mut start = start_range.to_unsigned(min_len)?;
    let mut end = end_range.to_unsigned(min_len)?;
    end = end.set_end(end.end().max(start.start() + step - 1))?;

    // the "middle" part must be in the total range
    end = end.set_start(end.start().clamp(start.start() + step - 1, end.end()))?;

    // align them to be a multiple of step away from start.start
    end = end
        .set_end(start.start() + step - 1 + (end.end() - start.start() - step + 1) / step * step)?
        .set_start(start.start() + step - 1 + (end.start() - start.start()) / step * step)?;
    // clamp and align the end of the start range too
    start = start.set_end(start.end().clamp(start.start(), end.end()))?;
    start = start.set_end(start.start() + (start.end() - start.start()) / step * step)?;

    let to_rel = |x: SignedInclRange| {
        if x.start() >= 0 {
            Relativity::Start
        } else {
            Relativity::End
        }
    };
    start_range = start.to_signed(to_rel(start_range), to_rel(start_range), min_len)?;
    end_range = end.to_signed(to_rel(end_range), to_rel(end_range), min_len)?;
    Some((start_range, end_range))
}

fn presums<S: LinearCheck + ?Sized>(
    summer: &S,
    bytes: &[u8],
    sum: &S::Sum,
    start_range: UnsignedInclRange,
    end_range: UnsignedInclRange,
) -> (Vec<S::Sum>, Vec<S::Sum>) {
    if start_range.start() > start_range.end() || end_range.start() > end_range.end() {
        return (Vec::new(), Vec::new());
    }
    if start_range.start() >= bytes.len() {
        return (Vec::new(), Vec::new());
    }
    // we calculate two presum arrays, one for the starting values and one for the end values
    let step = summer.wordspec().word_bytes();
    let mut state = summer.init();
    let mut start_presums = Vec::with_capacity(start_range.len() / step);
    let mut end_presums = Vec::with_capacity(end_range.len() / step);
    let neg_init = summer.negate(summer.init());
    let iter_range = start_range.start()..=end_range.end();
    for (i, c) in summer
        .wordspec()
        .iter_words(&bytes[iter_range.clone()])
        .enumerate()
        .map(|(i, c)| (i * step + start_range.start(), c))
    {
        if start_range.contains(i) {
            // from the startsums, we substract the init value of the checksum
            start_presums.push(summer.add(state.clone(), &neg_init));
        }
        state = summer.dig_word(state, c);
        if end_range.contains(i + step - 1) {
            // from the endsums, we finalize them and subtract the given final sum
            let endstate = summer.add(summer.finalize(state.clone()), &summer.negate(sum.clone()));
            end_presums.push(endstate);
        }
    }
    let mut start_index = start_presums.len();
    let mut end_index = end_presums.len();
    // we then shift checksums to length of file
    let mut shift = summer.init_shift();
    for i in iter_range
        .rev()
        .filter(|i| (i - start_range.start()) % step == 0)
    {
        if end_range.contains(i + step - 1) {
            end_index -= 1;
            end_presums[end_index] = summer.shift(end_presums[end_index].clone(), &shift)
        }
        shift = summer.inc_shift(shift);
        if start_range.contains(i) {
            start_index -= 1;
            start_presums[start_index] = summer.shift(start_presums[start_index].clone(), &shift)
        }
    }
    assert_eq!(start_index, 0);
    assert_eq!(end_index, 0);
    // This has the effect that, when substracting the n'th startsum from the m'th endsum, we get the checksum
    // from n to m, minus the final sum (all shifted by (len-m)), which is 0 exactly when the checksum from n to m is equal to
    // the final sum, which means that start_presums[n] = end_presums[m]
    //
    // we then sort an array of indices so equal elements are adjacent, allowing us to easily get the equal elements
    // Anyway, here's some cryptic stuff i made up and have to put at least *somewhere* so i don't forget it
    // 		        ([0..m] + f - s)*x^(k-m) - ([0..n] - i)*x^(k-n)
    // (4)	        = ([0..m] + f - s)*x^(k-m) - ([0..n] - i)*x^(m-n)*x^(k-m)
    // (1) (5)		= ([0..m] + f - s - [0..n]*x^(n-m) + i*x^(m-n))*x^(k-m)
    // (2) (6)		= ([0..n]*x^(m-n) + [n..m] + f - s - [0 ..n]*x^(n-m) + i*x^(m-n))*x^(k-m)
    // (1)		    = ([n..m] + f - s + i*x^(m-n))*x^(k-m)
    // (6)		    = (i*[n..m] + f - s)*x^(k-m)
    // (7)		    = (finalize(i*[n..m]) - s)*x^(k-m)
    // therefore
    //                  (finalize(i*[n..m]) - s)*x^(k-m) == 0
    // (2) (3) (6)  <=> finalize(i*[n..m]) - s           == 0
    // (1)          <=> finalize(i*[n..m])               == s
    (start_presums, end_presums)
}

pub type RangePair = (Vec<isize>, Vec<isize>);

/// A struct for helping to sort and get duplicates of arrays of arrays.
#[derive(Debug)]
struct PresumSet<Sum: Clone + Eq + Ord + Debug> {
    idx: Vec<u32>,
    presum: Vec<Vec<Sum>>,
}

impl<Sum: Clone + Eq + Ord + Debug + Send + Sync> PresumSet<Sum> {
    /// Gets a new PresumSet. Gets sorted on construction.
    fn new(presum: Vec<Vec<Sum>>) -> Self {
        let firstlen = presum[0].len();
        // check that all sum arrays are of the same length
        for x in presum.iter() {
            assert_eq!(firstlen, x.len());
        }
        // vector of all indices
        let mut idxvec: Vec<_> = (0..firstlen as u32).collect();
        // get a permutation vector representing the sort of the presum arrays first by value and then by index

        #[cfg(feature = "parallel")]
        idxvec.par_sort_unstable_by(|a, b| Self::cmp_idx(&presum, *a, &presum, *b).then(a.cmp(b)));
        #[cfg(not(feature = "parallel"))]
        idxvec.sort_unstable_by(|a, b| Self::cmp_idx(&presum, *a, &presum, *b).then(a.cmp(&b)));
        Self {
            idx: idxvec,
            presum,
        }
    }
    /// Compares all elements of the first vector at an index to the ones of the second vector lexicographically (assuming same length).
    fn cmp_idx(presum_a: &[Vec<Sum>], a: u32, presum_b: &[Vec<Sum>], b: u32) -> Ordering {
        for (x, y) in presum_a.iter().zip(presum_b.iter()) {
            let cmp = x[a as usize].cmp(&y[b as usize]);
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    }
    /// Finds groups of indices equal elements in the first set and the second set and
    /// returns them for each equal array.
    fn equal_pairs(&self, other: &Self) -> Vec<(Vec<u32>, Vec<u32>)> {
        let mut ret = Vec::new();
        let mut a_idx = 0;
        let mut b_idx = 0;
        while a_idx < self.idx.len() && b_idx < other.idx.len() {
            let apos = self.idx[a_idx];
            let bpos = other.idx[b_idx];
            match Self::cmp_idx(&self.presum, apos, &other.presum, bpos) {
                Ordering::Less => {
                    a_idx += 1;
                }
                Ordering::Greater => {
                    b_idx += 1;
                }
                Ordering::Equal => {
                    let mut n_a = 0;
                    // gets all runs of equal elements in a and b array
                    for x in &self.idx[a_idx..] {
                        if Self::cmp_idx(&self.presum, apos, &self.presum, *x) == Ordering::Equal {
                            n_a += 1;
                        } else {
                            break;
                        }
                    }
                    let mut n_b = 0;
                    for x in &other.idx[b_idx..] {
                        if Self::cmp_idx(&other.presum, bpos, &other.presum, *x) == Ordering::Equal
                        {
                            n_b += 1;
                        } else {
                            break;
                        }
                    }
                    let mut a_vec = Vec::new();
                    a_vec.extend_from_slice(&self.idx[a_idx..a_idx + n_a]);
                    let mut b_vec = Vec::new();
                    b_vec.extend_from_slice(&other.idx[b_idx..b_idx + n_b]);
                    ret.push((a_vec, b_vec));
                    // puts indexes beyond equal elements
                    a_idx += n_a;
                    b_idx += n_b;
                }
            }
        }
        // sort it, for good measure
        ret.sort_unstable();
        ret
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum CheckBuilderErr {
    /// The checksum given on construction does not match
    /// the checksum of "123456789"
    CheckFail,
    /// A mandatory parameter is missing
    MissingParameter(&'static str),
    /// A value of a parameter is out of range
    ValueOutOfRange(&'static str),
    /// The given string given to the from_str function
    /// could not be interpreted correctly,
    ///
    /// The String indicates the key with the malformant.
    MalformedString(String),
    /// A key given to the from_str function is not known
    UnknownKey(String),
}

impl std::fmt::Display for CheckBuilderErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use CheckBuilderErr::*;
        match self {
            CheckFail => write!(f, "Failed checksum test"),
            MissingParameter(para) => write!(f, "Missing parameter '{}'", para),
            ValueOutOfRange(key) => write!(f, "Value for parameter '{}' invalid", key),
            MalformedString(key) => {
                if key.is_empty() {
                    write!(f, "Malformed input string")
                } else {
                    write!(f, "Malformed input string at {}", key)
                }
            }
            UnknownKey(key) => write!(f, "Unknown key '{}'", key),
        }
    }
}

impl std::error::Error for CheckBuilderErr {}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum CheckReverserError {
    MissingParameter(&'static str),
    UnsuitableFiles(&'static str),
    ChecksumFileMismatch,
}

impl std::fmt::Display for CheckReverserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use CheckReverserError::*;
        match self {
            MissingParameter(s) => write!(f, "Missing Parameters: {}", s),
            UnsuitableFiles(s) => write!(
                f,
                "Could not reverse because \
                files are unsuitable: {}",
                s
            ),
            ChecksumFileMismatch => write!(
                f,
                "Number of files does not \
                match number of checksums"
            ),
        }
    }
}
impl std::error::Error for CheckReverserError {}

/// Trait for checksums
pub trait Checksum {
    fn to_width_str(&self, width: usize) -> String;
}

// default implementation for normal numbers
impl<T: crate::bitnum::BitNum> Checksum for T {
    fn to_width_str(&self, width: usize) -> String {
        if width == 0 {
            return String::new();
        }
        let w = (width - 1) / 4 + 1;
        format!("{:0width$x}", self, width = w)
    }
}

#[allow(dead_code)]
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::checksum::Relativity;
    use quickcheck::{Arbitrary, Gen, TestResult};
    use rand::Rng;
    static EXAMPLE_TEXT: &str = r#"Als Gregor Samsa eines Morgens aus unruhigen Träumen erwachte, fand er sich in
seinem Bett zu einem ungeheueren Ungeziefer verwandelt. Er lag auf seinem
panzerartig harten Rücken und sah, wenn er den Kopf ein wenig hob, seinen
gewölbten, braunen, von bogenförmigen Versteifungen geteilten Bauch, auf
dessen Höhe sich die Bettdecke, zum gänzlichen Niedergleiten bereit, kaum
noch erhalten konnte. Seine vielen, im Vergleich zu seinem sonstigen Umfang
kläglich dünnen Beine flimmerten ihm hilflos vor den Augen.

»Was ist mit mir geschehen?« dachte er. Es war kein Traum, sein Zimmer, ein
richtiges, nur etwas zu kleines Menschenzimmer, lag ruhig zwischen den vier
wohlbekannten Wänden, über dem Tisch, auf dem eine auseinandergepackte
Musterkollektion von Tuchwaren ausgebreitet war – Samsa war Reisender –,
hing das Bild, das er vor kurzem aus einer illustrierten Zeitschrift
ausgeschnitten und in einem hübschen, vergoldeten Rahmen untergebracht hatte.
Es stellte eine Dame dar, die, mit einem Pelzhut und einer Pelzboa versehen,
aufrecht dasaß und einen schweren Pelzmuff, in dem ihr ganzer Unterarm
verschwunden war, dem Beschauer entgegenhob.

Gregors Blick richtete sich dann zum Fenster, und das trübe Wetter – man
hörte Regentropfen auf das Fensterblech aufschlagen – machte ihn ganz
melancholisch. »Wie wäre es, wenn ich noch ein wenig weiterschliefe und alle
Narrheiten vergäße,« dachte er, aber das war gänzlich undurchführbar, denn
er war gewöhnt, auf der rechten Seite zu schlafen, konnte sich aber in seinem
gegenwärtigen Zustand nicht in diese Lage bringen. Mit welcher Kraft er sich
auch auf die rechte Seite warf, immer wieder schaukelte er in die Rückenlage
zurück. Er versuchte es wohl hundertmal, schloß die Augen, um die zappelnden
Beine nicht sehen zu müssen und ließ erst ab, als er in der Seite einen noch
nie gefühlten, leichten, dumpfen Schmerz zu fühlen begann.
"#;
    pub fn test_shifts<T: LinearCheck>(chk: &T) {
        let test_sum = chk
            .digest(&b"T\x00\x00\x00E\x00\x00\x00S\x00\x00\x00\x00T"[..])
            .unwrap();
        let shift3 = chk.shift_n(3);
        let shift4 = chk.inc_shift(shift3.clone());
        let mut new_sum = chk.init();
        new_sum = chk.dig_word(new_sum, b'T' as u64);
        new_sum = chk.shift(new_sum, &shift3);
        new_sum = chk.dig_word(new_sum, b'E' as u64);
        new_sum = chk.shift(new_sum, &shift3);
        new_sum = chk.dig_word(new_sum, b'S' as u64);
        new_sum = chk.shift(new_sum, &shift4);
        new_sum = chk.dig_word(new_sum, b'T' as u64);
        assert_eq!(test_sum, chk.finalize(new_sum));
    }
    pub fn test_find<L: LinearCheck>(chk: &L) {
        let sum_1_9 = chk.digest(&b"123456789"[..]).unwrap();
        let sum_9_1 = chk.digest(&b"987654321"[..]).unwrap();
        let sum_1_9_1 = chk.digest(&b"12345678987654321"[..]).unwrap();
        assert_eq!(
            chk.find_segments(
                &[Vec::from("a123456789X1235H123456789Y")],
                &[sum_1_9.clone()],
                Relativity::Start
            ),
            vec![(vec![1], vec![9]), (vec![16], vec![24])]
        );
        assert_eq!(
            chk.find_segments(
                &[
                    Vec::from("XX98765432123456789XXX"),
                    Vec::from("XX12345678987654321XX")
                ],
                &[sum_1_9.clone(), sum_9_1.clone()],
                Relativity::Start
            ),
            vec![(vec![10], vec![18])]
        );
        assert_eq!(
            chk.find_segments(
                &[
                    Vec::from("XXX12345678987654321AndSoOn"),
                    Vec::from("ABC123456789.super."),
                    Vec::from("Za!987654321ergrfrf")
                ],
                &[sum_1_9_1, sum_1_9, sum_9_1],
                Relativity::End
            ),
            vec![(vec![3], vec![-8])]
        )
    }
    pub fn check_example<D: Digest>(chk: &D, sum: D::Sum) {
        assert_eq!(chk.digest(EXAMPLE_TEXT.as_bytes()).unwrap(), sum)
    }
    // this was written before including quickcheck, hence this manual property testing implementation
    pub fn test_prop<L: LinearCheck>(chk: &L) {
        let mut test_values = Vec::new();
        test_values.push(chk.init());
        let e = &chk.add(chk.negate(chk.init()), &chk.init());
        test_values.push(e.clone());
        let mut rng = rand::thread_rng();
        let mut s = chk.init();
        while test_values.len() < 100 {
            s = chk.dig_word(s, rng.gen());
            if rng.gen_bool(0.01) {
                test_values.push(s.clone());
            }
        }
        for a in test_values.iter() {
            check_neutral(chk, e, a);
            check_invert(chk, e, a);
            check_shift1(chk, a);
            check_shiftn(chk, a);
            check_bil(chk, e, a);
            check_fin(chk, e, a);
            for b in test_values.iter() {
                check_commut(chk, a, b);
                check_dist(chk, a, b);
                for c in test_values.iter() {
                    check_assoc(chk, a, b, c);
                }
            }
        }
    }
    fn check_assoc<L: LinearCheck>(chk: &L, a: &L::Sum, b: &L::Sum, c: &L::Sum) {
        assert_eq!(
            chk.add(chk.add(a.clone(), b), c),
            chk.add(a.clone(), &chk.add(b.clone(), c)),
            "Associativity Fail: ({:x?} + {:x?}) + {:x?} != {:x?} + ({:x?} + {:x?})",
            a,
            b,
            c,
            a,
            b,
            c
        );
    }
    fn check_neutral<L: LinearCheck>(chk: &L, e: &L::Sum, a: &L::Sum) {
        assert_eq!(
            chk.add(a.clone(), e),
            a.clone(),
            "Neutral Element Fail: {:x?} + {:x?} != {:x?}",
            a,
            e,
            a
        );
    }
    fn check_commut<L: LinearCheck>(chk: &L, a: &L::Sum, b: &L::Sum) {
        assert_eq!(
            chk.add(b.clone(), a),
            chk.add(a.clone(), b),
            "Commutativity Fail: {:x?} + {:x?} != {:x?} + {:x?}",
            b,
            a,
            a,
            b
        );
    }
    fn check_invert<L: LinearCheck>(chk: &L, e: &L::Sum, a: &L::Sum) {
        assert_eq!(
            chk.add(chk.negate(a.clone()), a),
            e.clone(),
            "Inversion Fail: -{:x?} + {:x?} != {:x?}",
            a,
            a,
            e
        );
    }
    fn check_shift1<L: LinearCheck>(chk: &L, a: &L::Sum) {
        assert_eq!(
            chk.shift(a.clone(), &chk.shift_n(1)),
            chk.dig_word(a.clone(), 0u64),
            "Shift1 Fail: shift({:x?}, shift_n1(1)) != dig_word({:x?}, 0u8)",
            a,
            a
        );
    }
    fn check_shiftn<L: LinearCheck>(chk: &L, a: &L::Sum) {
        for x in &[1, 5, 16, 1094, 5412] {
            let shifted = chk.shift(a.clone(), &chk.shift_n(*x));
            for y in &[4, 526, 0, 41, 4321] {
                assert_eq!(
                    chk.shift(shifted.clone(), &chk.shift_n(*y)),
                    chk.shift(a.clone(), &chk.shift_n(x+y)),
                    "Shiftn Fail: shift(shift({:x?}, shift_n({:?})), shift_n({:?})) != shift({:x?}, shift_n({} + {}))", a, x, y, a, x, y
                );
            }
        }
    }
    fn check_dist<L: LinearCheck>(chk: &L, a: &L::Sum, b: &L::Sum) {
        assert_eq!(
            chk.add(chk.dig_word(a.clone(), 0u64), &chk.dig_word(b.clone(), 0u64)),
            chk.dig_word(chk.add(a.clone(), b), 0u64),
            "Distributivity Fail: dig_word({:x?}, 0u8) + dig_word({:x?}, 0u8) != dig_word({:x?} + {:x?}, 0u8)", a, b, a, b
        );
    }
    fn check_bil<L: LinearCheck>(chk: &L, e: &L::Sum, a: &L::Sum) {
        for k in 0u64..=255 {
            assert_eq!(
                chk.dig_word(a.clone(), k),
                chk.add(chk.dig_word(a.clone(), 0u64), &chk.dig_word(e.clone(), k)),
                "Bilinearity Fail: dig_word({:x?}, {:#x}) != dig_word({:x?}, 0u8) + dig_word(0, {:#x}u8)", a, k, a ,k
            )
        }
    }
    fn check_fin<L: LinearCheck>(chk: &L, e: &L::Sum, a: &L::Sum) {
        assert_eq!(
            chk.add(chk.finalize(a.clone()), &chk.negate(a.clone())),
            chk.finalize(e.clone()),
            "Finalize Linearity Fail: finalize({:x?}) - {:x?} != {:x?}",
            a,
            a,
            &chk.finalize(e.clone())
        )
    }
    /// For generating files for tests so that there are at least 3 with one having a different length
    /// and the individual lengths are multiples of 8 so that power-of-two wordsizes can be tested
    #[derive(Clone, PartialEq, Eq, Debug)]
    pub struct ReverseFileSet(pub Vec<Vec<u8>>);
    impl Arbitrary for ReverseFileSet {
        fn arbitrary(g: &mut Gen) -> Self {
            let new_size = |q: &mut Gen| {
                let s = q.size() / 8;
                8 * (usize::arbitrary(q) % s)
            };
            let n_files = (usize::arbitrary(g) % g.size()) + 3;
            let mut lengths = Vec::new();
            for _ in 0..n_files {
                lengths.push(new_size(g));
            }
            if lengths.iter().all(|x| *x == lengths[0]) {
                lengths[0] += 8;
            }
            let mut ret = Vec::with_capacity(n_files);
            for new_len in lengths {
                let mut cur_file = Vec::with_capacity(new_len);
                for _ in 0..new_len {
                    cur_file.push(u8::arbitrary(g));
                }
                ret.push(cur_file);
            }
            ret.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)).reverse());
            ReverseFileSet(ret)
        }
        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            let vec = self.0.clone();
            Box::new((1..=(vec.len() - 3)).map(move |x| ReverseFileSet(Vec::from(&vec[x..]))))
        }
    }
    impl ReverseFileSet {
        pub fn with_checksums<T: Digest>(&self, dig: &T) -> Vec<(&[u8], Vec<u8>)> {
            self.0
                .iter()
                .map(|f| {
                    let checksum = dig.to_bytes(dig.digest(f.as_slice()).unwrap());
                    (f.as_slice(), checksum)
                })
                .collect()
        }
        pub fn check_matching<T, I>(&self, reference: &T, result_iter: I) -> TestResult
        where
            T: Digest + Eq + std::fmt::Display,
            I: Iterator<Item = Result<T, CheckReverserError>>,
            T::Sum: std::fmt::LowerHex,
        {
            let chk_files = self.with_checksums(reference);
            let mut has_appeared = false;
            for (count, modsum_loop) in result_iter.enumerate() {
                if count > 10000 {
                    return TestResult::discard();
                }
                let modsum_loop = match modsum_loop {
                    Err(_) => return TestResult::discard(),
                    Ok(x) => x,
                };
                if &modsum_loop == reference {
                    has_appeared = true;
                }
                for (file, original_check) in &chk_files {
                    let checksum = modsum_loop.to_bytes(modsum_loop.digest(file).unwrap());
                    if &checksum != original_check {
                        eprint!("expected checksum: ");
                        for x in original_check {
                            eprint!("{:02x}", x);
                        }
                        eprintln!();
                        eprint!("actual checksum: ");
                        for x in checksum {
                            eprint!("{:02x}", x);
                        }
                        eprintln!();
                        eprintln!("checksum: {}", modsum_loop);
                        eprintln!("original checksum: {}", reference);
                        return TestResult::failed();
                    }
                }
            }
            if !has_appeared {
                eprintln!("{} has not appeared!", reference);
                return TestResult::failed();
            }
            TestResult::passed()
        }
    }
}

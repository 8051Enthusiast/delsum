pub mod crc;
pub mod fletcher;
pub mod modsum;

use rayon::prelude::*;
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::io::Read;

/// A basic trait for a checksum where
/// * init gives an initial state
/// * dig_byte processes a single byte
/// * finalize is applied to get the final sum after all bytes are processed.
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
    type Sum: Clone + Eq + Ord + std::fmt::Debug + Send + Sync;
    /// Gets an initial sum before the bytes are processed through the sum.
    ///
    /// For instance in the case of crc, the sum type is some integer and the returned value from
    /// this function could be 0xffffffff (e.g. in the bzip2 crc).
    fn init(&self) -> Self::Sum;
    /// Processes a single byte from the text.
    ///
    /// For a crc, this corresponds to shifting, adding the byte and reducing.
    fn dig_byte(&self, sum: Self::Sum, byte: u8) -> Self::Sum;
    /// After all bytes are read, this function is called to do some finalization.
    ///
    /// In the case of crc, this corresponds to adding a constant at the end
    /// (and maybe also adding some 0s to the end of the text).
    fn finalize(&self, sum: Self::Sum) -> Self::Sum;
    /// Takes a reader and calculates the checksums of all bytes therein.
    fn digest<R: Read>(&self, buf: R) -> Result<Self::Sum, std::io::Error> {
        let sum = buf.bytes().try_fold(self.init(), |partsum, newbyte| {
            newbyte.map(|x| self.dig_byte(partsum, x))
        })?;
        Ok(self.finalize(sum))
    }
}

#[derive(Copy, Clone)]
pub enum Relativity {
    Start,
    End,
}
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum RelativeIndex {
    FromStart(usize),
    FromEnd(usize),
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
/// * `add(dig_byte(s, 0u8), dig_byte(r, 0u8)) == dig_byte(add(s, r), 0u8)`
/// * `dig_byte(s, k) == dig_byte(0, k) + dig_byte(s, 0u8)` (consequently, `dig_byte(0, 0u8) == 0`)
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
    /// Acts as if applying `dig_byte(s, 0)` `n` times to to `s` (see trait documentation for more).
    ///
    /// Please implement more efficient (equivalent) implementation for each type if possible.
    fn shift_n(&self, n: usize) -> Self::Shift {
        let mut shift = self.init_shift();
        for _ in 0..n {
            shift = self.inc_shift(shift);
        }
        shift
    }
    #[doc(hidden)]
    fn presums(
        &self,
        bytes: &[u8],
        sum: &Self::Sum,
        start_range: std::ops::Range<usize>,
        end_range: std::ops::Range<usize>,
    ) -> (Vec<Self::Sum>, Vec<Self::Sum>) {
        // we calculate two presum arrays, one for the starting values and one for the end values
        let mut state = self.init();
        let mut start_presums = Vec::with_capacity(start_range.len());
        let mut end_presums = Vec::with_capacity(end_range.len());
        let neg_init = self.negate(self.init());
        for (i, c) in bytes.iter().enumerate() {
            if start_range.contains(&i) {
                // from the startsums, we substract the init value of the checksum
                start_presums.push(self.add(state.clone(), &neg_init));
            }
            state = self.dig_byte(state, *c);
            if end_range.contains(&i) {
                // from the endsums, we finalize them and subtract the given final sum
                let endstate = self.add(self.finalize(state.clone()), &self.negate(sum.clone()));
                end_presums.push(endstate);
            }
        }
        // we then shift checksums to length of file
        let mut shift = self.init_shift();
        for i in (0..bytes.len()).rev() {
            if end_range.contains(&i) {
                end_presums[i - end_range.start] =
                    self.shift(end_presums[i - end_range.start].clone(), &shift)
            }
            shift = self.inc_shift(shift);
            if start_range.contains(&i) {
                start_presums[i - start_range.start] =
                    self.shift(start_presums[i - start_range.start].clone(), &shift)
            }
        }
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
    fn find_segments(&self, bytes: &[Vec<u8>], sum: &[Self::Sum], rel: Relativity) -> RangePairs {
        if bytes.is_empty() {
            return Vec::new();
        }
        if u32::try_from(bytes[0].len()).is_err() {
            // only support 32-bit length files for now, since a usize for every byte would take a lot of space
            panic!("File must be under 4GiB!");
        }
        let min_len = bytes.iter().map(|x| x.len()).min().unwrap();
        let end_range = |b: &[u8]| match rel {
            Relativity::Start => 0..min_len,
            Relativity::End => (b.len() - min_len)..b.len(),
        };
        let (start_presums, end_presums) = bytes
            .par_iter()
            .zip(sum.par_iter())
            .map(|(b, s)| self.presums(b, &s, 0..min_len, end_range(&b)))
            .unzip();

        let start_preset = PresumSet::new(start_presums);
        let end_preset = PresumSet::new(end_presums);

        let mut ret_vec = Vec::new();
        for (a, b) in start_preset.equal_pairs(&end_preset) {
            let starts: Vec<_> = a.iter().map(|x| usize::try_from(*x).unwrap()).collect();
            let ends: Vec<_> = b.iter().map(|x| usize::try_from(*x).unwrap() + 1).collect();
            let min_start = *starts.iter().min().unwrap_or(&min_len);
            let max_end = *ends.iter().max().unwrap_or(&0);
            let rel_ends: Vec<_> = ends
                .into_iter()
                .filter(|x| x > &min_start)
                .map(|x| match rel {
                    Relativity::Start => RelativeIndex::FromStart(x),
                    Relativity::End => RelativeIndex::FromEnd(min_len - x),
                })
                .collect();
            let rel_starts = starts.into_iter().filter(|x| x < &max_end).collect();
            if !rel_ends.is_empty() {
                ret_vec.push((rel_starts, rel_ends));
            }
        }
        ret_vec
    }
}

pub type RangePairs = Vec<(Vec<usize>, Vec<RelativeIndex>)>;

/// A struct for helping to sort and get duplicates of arrays of arrays.
#[derive(Debug)]
struct PresumSet<Sum: Clone + Eq + Ord + std::fmt::Debug> {
    idx: Vec<u32>,
    presum: Vec<Vec<Sum>>,
}

impl<Sum: Clone + Eq + Ord + std::fmt::Debug + Send + Sync> PresumSet<Sum> {
    /// Gets a new PresumSet. Gets sorted on construction.
    fn new(presum: Vec<Vec<Sum>>) -> Self {
        let firstlen = presum[0].len();
        // check that all sum arrays are of the same length
        for x in presum.iter() {
            assert_eq!(firstlen, x.len());
        }
        // vector of all indices
        let mut idxvec: Vec<_> = (0..=(firstlen - 1) as u32).collect();
        // get a permutation vector representing the sort of the presum arrays first by value and then by index
        idxvec.par_sort_unstable_by(|a, b| Self::cmp_idx(&presum, *a, &presum, *b).then(a.cmp(&b)));
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
        while a_idx < self.idx.len() && b_idx < self.idx.len() {
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

#[derive(Clone, PartialEq, Eq, Debug)]
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

#[derive(Clone, PartialEq, Eq, Debug)]
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
                "Could not reverse because\
                files are unsuitable: {}",
                s
            ),
            ChecksumFileMismatch => write!(
                f,
                "Number of files does not\
                match number of checksums"
            ),
        }
    }
}
impl std::error::Error for CheckReverserError {}

/// Turns Result<Iterator, Error> into Iterator<Result<Iterator::Item, Error>> so that
/// on Err, only the single error is iterated, and else the items of the iterator
fn unresult_iter<I, E>(x: Result<I, E>) -> impl Iterator<Item = Result<I::Item, E>>
where
    I: std::iter::Iterator,
    E: std::error::Error,
{
    let (i, e) = match x {
        Ok(i) => (Some(i.map(Ok)), None),
        Err(e) => (None, Some(std::iter::once(Err(e)))),
    };
    e.into_iter().flatten().chain(i.into_iter().flatten())
}

#[allow(dead_code)]
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::checksum::{RelativeIndex, Relativity};
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
        new_sum = chk.dig_byte(new_sum, b'T');
        new_sum = chk.shift(new_sum, &shift3);
        new_sum = chk.dig_byte(new_sum, b'E');
        new_sum = chk.shift(new_sum, &shift3);
        new_sum = chk.dig_byte(new_sum, b'S');
        new_sum = chk.shift(new_sum, &shift4);
        new_sum = chk.dig_byte(new_sum, b'T');
        assert_eq!(test_sum, chk.finalize(new_sum));
    }
    pub fn test_find<L: LinearCheck>(chk: &L) {
        let sum_1_9 = chk.digest(&b"123456789"[..]).unwrap();
        let sum_9_1 = chk.digest(&b"987654321"[..]).unwrap();
        let sum_1_9_1 = chk.digest(&b"12345678987654321"[..]).unwrap();
        assert_eq!(
            chk.find_segments(
                &[Vec::from(&"a123456789X1235H123456789Y"[..])],
                &[sum_1_9.clone()],
                Relativity::Start
            ),
            vec![
                (vec![1], vec![RelativeIndex::FromStart(10)]),
                (vec![16], vec![RelativeIndex::FromStart(25)])
            ]
        );
        assert_eq!(
            chk.find_segments(
                &[
                    Vec::from(&"XX98765432123456789XXX"[..]),
                    Vec::from(&"XX12345678987654321XX"[..])
                ],
                &[sum_1_9.clone(), sum_9_1.clone()],
                Relativity::Start
            ),
            vec![(vec![10], vec![RelativeIndex::FromStart(19)])]
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
            vec![(vec![3], vec![RelativeIndex::FromEnd(7)])]
        )
    }
    pub fn check_example<D: Digest>(chk: &D, sum: D::Sum) {
        assert_eq!(chk.digest(EXAMPLE_TEXT.as_bytes()).unwrap(), sum)
    }

    pub fn test_prop<L: LinearCheck>(chk: &L) {
        let mut test_values = Vec::new();
        test_values.push(chk.init());
        let e = &chk.add(chk.negate(chk.init()), &chk.init());
        test_values.push(e.clone());
        let mut rng = rand::thread_rng();
        let mut s = chk.init();
        while test_values.len() < 100 {
            s = chk.dig_byte(s, rng.gen());
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
            chk.dig_byte(a.clone(), 0u8),
            "Shift1 Fail: shift({:x?}, shift_n1(1)) != dig_byte({:x?}, 0u8)",
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
            chk.add(chk.dig_byte(a.clone(), 0u8), &chk.dig_byte(b.clone(), 0u8)),
            chk.dig_byte(chk.add(a.clone(), b), 0u8),
            "Distributivity Fail: dig_byte({:x?}, 0u8) + dig_byte({:x?}, 0u8) != dig_byte({:x?} + {:x?}, 0u8)", a, b, a, b
        );
    }
    fn check_bil<L: LinearCheck>(chk: &L, e: &L::Sum, a: &L::Sum) {
        for k in 0u8..=255 {
            assert_eq!(
                chk.dig_byte(a.clone(), k),
                chk.add(chk.dig_byte(a.clone(), 0u8), &chk.dig_byte(e.clone(), k)),
                "Bilinearity Fail: dig_byte({:x?}, {:#x}) != dig_byte({:x?}, 0u8) + dig_byte(0, {:#x}u8)", a, k, a ,k
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
}

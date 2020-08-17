pub mod crc;

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
    type Sum: Clone + Eq + Ord + std::fmt::Debug;
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
/// * `add(shift(s, shift_n(n)), shift(r, shift_n(n))) == shift(add(s, r), n)`
/// * `dig_byte(s, k) == dig_byte(0, k) + dig_byte(s, 0u8)` (consequently, `dig_byte(0, 0u8) == 0`)
/// * for all sums `s`, `add(finalize(s), negate(s))` is constant (finalize adds a constant value to the sum)
/// * all methods without default implementations (including those from `Digest`) should run in constant time (assuming constant `Shift`, `Sum` types)
///
/// Basically, it is a graded ring or something idk.
pub trait LinearCheck: Digest {
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
    /// Returns an array of unfinalized checksums up to each byte
    /// start: index of first byte to calculate checksums to (not including byte itself)
    /// end: index of last byte to calculate checksums to (including byte itself)
    fn presums(&self, bytes: &[u8], start: usize, end: usize) -> Vec<Self::Sum> {
        if start >= end || end > bytes.len() {
            panic!("Oh no you didn't!");
        }
        // get checksum state before first byte
        let start_state = bytes[..start]
            .iter()
            .fold(self.init(), |s, b| self.dig_byte(s, *b));
        // calculate checksums for each value in range and collect
        // them into a vector
        std::iter::once(start_state.clone())
            .chain(bytes[start..end].iter().scan(start_state, |s, b| {
                *s = self.dig_byte(s.clone(), *b);
                Some(s.clone())
            }))
            .collect()
    }
    #[doc(hidden)]
    fn normalize_presums(&self, presums: &mut Vec<Self::Sum>, pre_shift: Self::Shift) {
        let mut shift = pre_shift;
        for x in presums.iter_mut().rev() {
            *x = self.shift(x.clone(), &shift);
            shift = self.inc_shift(shift);
        }
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
    fn find_checksum_segments(&self, bytes: &[u8], sum: Self::Sum) -> Vec<(Vec<u32>, Vec<u32>)> {
        // we calculate two presum arrays, one for the starting values and one for the end values
        let mut start_presums = self.presums(bytes, 0, bytes.len());
        let mut end_presums = start_presums.clone();
        let neg_init = self.negate(self.init());
        // from the startsums, we substract the init value of the checksum and then shift the sums to the length of the file
        for x in start_presums.iter_mut() {
            *x = self.add(x.clone(), &neg_init);
        }
        self.normalize_presums(&mut start_presums, self.init_shift());
        // from the endsums, we finalize them and subtract the given final sum, and shift the sums to the length of the file
        for x in end_presums.iter_mut() {
            *x = self.add(self.finalize(x.clone()), &self.negate(sum.clone()));
        }
        self.normalize_presums(&mut end_presums, self.init_shift());
        // This has the effect that, when substracting the n'th startsum from the m'th endsum, we get the checksum
        // from n to m, minus the final sum (all shifted by (len-m)), which is 0 exactly when the checksum from n to m is equal to
        // the final sum, which means that start_presums[n] = end_presums[m]
        //
        // we then sort an array of indices so equal elements are adjacent, allowing us to easily get the equal elements
        // Anyway, here's some cryptic stuff i made up and have to put at least *somewhere* so i don't forget it
        // 		        ([0..m] + f - s)*x^(k-m) - ([0..n] - init)*x^(k-n)
        // (4)	        = ([0..m] + f - s)*x^(k-m) - ([0..n] - init)*x^(m-n)*x^(k-m)
        // (1) (5)		= ([0..m] + f - s - [0..n]*x^(n-m) + init*x^(m-n))*x^(k-m)
        // (2) (6)		= ([0..n]*x^(m-n) + [n..m] + f - s - [0 ..n]*x^(n-m) + init*x^(m-n))*x^(k-m)
        // (1)		    = ([n..m] + f - s + init*x^(m-n))*x^(k-m)
        // (6)		    = (init*[n..m] + f - s)*x^(k-m)
        // (7)		    = (finalize(init*[n..m]) - s)*x^(k-m)
        // therefore
        //                  (finalize(init*[n..m]) - s)*x^(k-m) == 0
        // (3)          <=> finalize(init*[n..m]) - s           == 0
        // (1)          <=> finalize(init*[n..m])               == s


        if u32::try_from(start_presums.len()).is_err() {
            // only support 32-bit length files for now, since a usize for every byte would take a lot of space
            panic!("File must be under 4GiB!");
        }
        let start_preset = PresumSet::new(vec![start_presums]);
        let end_preset = PresumSet::new(vec![end_presums]);
        start_preset.equal_pairs(&end_preset)
    }
}

/// A struct for helping to sort and get duplicates of arrays of arrays.
#[derive(Debug)]
struct PresumSet<Sum: Clone + Eq + Ord + std::fmt::Debug> {
    idx: Vec<u32>,
    presum: Vec<Vec<Sum>>,
}

impl<Sum: Clone + Eq + Ord + std::fmt::Debug> PresumSet<Sum> {
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
        while a_idx < self.idx.len() && b_idx < self.idx.len() {
            let apos = self.idx[a_idx];
            let bpos = other.idx[b_idx];
            match Self::cmp_idx(&self.presum, apos, &other.presum, bpos) {
                Ordering::Less => {
                    a_idx += 1;
                },
                Ordering::Greater => {
                    b_idx += 1;
                },
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

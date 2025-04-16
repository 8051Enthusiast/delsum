mod bitnum;
pub mod checksum;
mod divisors;
mod keyval;
pub mod utils;

pub mod crc;
pub(crate) mod endian;
pub mod fletcher;
pub mod modsum;
pub mod polyhash;

use bitnum::BitNum;
use checksum::{CheckBuilderErr, CheckReverserError, Checksum, const_sum};
use checksum::{Digest, LinearCheck, RangePair};
use crc::{CRC, CrcBuilder, reverse_crc};
use fletcher::{Fletcher, FletcherBuilder, reverse_fletcher};
use modsum::{ModSum, ModSumBuilder, reverse_modsum};
use num_traits::Zero;
use polyhash::{PolyHash, PolyHashBuilder, reverse_polyhash};
use std::cmp::Ordering;
use std::str::FromStr;
use utils::SignedInclRange;
#[cfg(feature = "parallel")]
use {crc::reverse_crc_para, fletcher::reverse_fletcher_para, rayon::prelude::*};
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

/// For figuring out what type of integer to use, we need to parse the width from the
/// model string, but to parse the model string, we need to know the integer type,
/// so it is done here separately.
/// We also need the prefix to find out what algorithm to use
fn find_prefix_width(s: &str) -> Result<(&str, usize, &str), CheckBuilderErr> {
    let stripped = s.trim_start();
    // it is done like this to ensure that no non-whitespace (blackspace?) is left at the end of the prefix
    let pref = stripped.split_whitespace().next();
    let (prefix, rest) = match PREFIXES.iter().find(|x| Some(**x) == pref) {
        Some(p) => (*p, &stripped[p.len()..]),
        None => return Err(CheckBuilderErr::MalformedString("algorithm".to_owned())),
    };
    for x in keyval::KeyValIter::new(rest) {
        match x {
            Err(k) => return Err(CheckBuilderErr::MalformedString(k)),
            Ok((k, v)) => {
                if &k == "width" {
                    return v
                        .parse()
                        .map_err(|_| CheckBuilderErr::MalformedString(k))
                        .map(|width| (prefix, width, rest));
                }
            }
        }
    }
    Err(CheckBuilderErr::MissingParameter("width"))
}

// modifies the end_range so that there is
// checksum_size bytes padding to the end
// in all files
fn cutoff_checksum_length(
    end_range: SignedInclRange,
    bytes: &[Vec<u8>],
    checksum_size: usize,
) -> Option<SignedInclRange> {
    let end = end_range.end();
    if end < 0 {
        let new_end = (-1 - checksum_size as isize).min(end);
        end_range.set_end(new_end)
    } else {
        let min_len = bytes.iter().map(|x| x.len()).min()? as isize;
        let new_end = (min_len - checksum_size as isize - 1).min(end);
        end_range.set_end(new_end)
    }
}

#[derive(Clone, Copy)]
pub enum SegmentChecksum<'a> {
    FromEnd(usize),
    Constant(&'a [Vec<u8>]),
}

/// A helper function for calling the find_segments function with strings arguments
fn find_segment_str<L>(
    spec: &str,
    bytes: &[Vec<u8>],
    sum: SegmentChecksum,
    start_range: SignedInclRange,
    end_range: SignedInclRange,
) -> Result<Vec<RangePair>, CheckBuilderErr>
where
    L: LinearCheck + FromStr<Err = CheckBuilderErr>,
    L::Sum: BitNum,
{
    let spec = L::from_str(spec)?;
    match sum {
        SegmentChecksum::Constant(sum_bytes) => {
            let sum_array: Vec<_> = sum_bytes
                .iter()
                .map(|x| const_sum(spec.wordspec().bytes_to_output(x)))
                .collect();
            Ok(spec.find_segments_range(bytes, &sum_array, start_range, end_range))
        }
        SegmentChecksum::FromEnd(n) => {
            let endian = spec.wordspec().output_endian;
            let width = spec.to_bytes(<L::Sum as Zero>::zero()).len();
            let checksum_length = width + n;
            let Some(end_range) = cutoff_checksum_length(end_range, bytes, checksum_length) else {
                return Ok(Vec::new());
            };
            let sum_array: Vec<_> = bytes
                .iter()
                .map(move |_| {
                    move |bytes: &[u8], addr: usize| {
                        let start = addr + n;
                        let end = addr + checksum_length;
                        Checksum::from_bytes(&bytes[start..end], endian)
                    }
                })
                .collect();
            Ok(spec.find_segments_range(bytes, &sum_array, start_range, end_range))
        }
    }
}

/// The available checksum types
static PREFIXES: &[&str] = &["fletcher", "crc", "modsum", "polyhash"];

/// A stringy function for determining which segments of a file have a given checksum.
///
/// It is given
/// * a string that models a checksum algorithm
/// * a vector of bytes slices (each slice containing the bytes of a file)
/// * a comma-separated string (without whitespace) containing target checksums for each file
/// * a parameter indicating whether the ends of the segments are relative to the start or the end of the file
///
/// # The Model String
/// A model string is generally of the form
/// ```text
/// [algorithm] width=[number] {more parameters}
/// ```
/// The `algorithm` parameter is either `fletcher`, `crc` or `modsum`.
/// Parameters depend solely on what kind of algorithm is used and more information is available
/// at the respective Builders.
pub fn find_checksum_segments(
    strspec: &str,
    bytes: &[Vec<u8>],
    sum: SegmentChecksum,
    start_range: SignedInclRange,
    end_range: SignedInclRange,
) -> Result<Vec<RangePair>, CheckBuilderErr> {
    let (prefix, width, rest) = find_prefix_width(strspec)?;
    match (width, prefix) {
        (1..=32, "crc") => find_segment_str::<CRC<u32>>(rest, bytes, sum, start_range, end_range),
        (33..=64, "crc") => find_segment_str::<CRC<u64>>(rest, bytes, sum, start_range, end_range),
        (65..=128, "crc") => {
            find_segment_str::<CRC<u128>>(rest, bytes, sum, start_range, end_range)
        }
        (1..=32, "modsum") => {
            find_segment_str::<ModSum<u32>>(rest, bytes, sum, start_range, end_range)
        }
        (33..=64, "modsum") => {
            find_segment_str::<ModSum<u64>>(rest, bytes, sum, start_range, end_range)
        }
        (1..=32, "fletcher") => {
            find_segment_str::<Fletcher<u16>>(rest, bytes, sum, start_range, end_range)
        }
        (33..=64, "fletcher") => {
            find_segment_str::<Fletcher<u32>>(rest, bytes, sum, start_range, end_range)
        }
        (65..=128, "fletcher") => {
            find_segment_str::<Fletcher<u64>>(rest, bytes, sum, start_range, end_range)
        }
        (1..=32, "polyhash") => {
            find_segment_str::<PolyHash<u32>>(rest, bytes, sum, start_range, end_range)
        }
        (33..=64, "polyhash") => {
            find_segment_str::<PolyHash<u64>>(rest, bytes, sum, start_range, end_range)
        }
        _ => Err(CheckBuilderErr::ValueOutOfRange("width")),
    }
}

fn get_checksums<A>(
    strspec: &str,
    files: &[&[u8]],
    width: usize,
) -> Result<Vec<Vec<u8>>, CheckBuilderErr>
where
    A: Digest + FromStr<Err = CheckBuilderErr>,
    A::Sum: crate::bitnum::BitNum,
{
    let algo = A::from_str(strspec)?;
    let mut sums = Vec::new();
    for file in files {
        sums.push(
            algo.wordspec()
                .output_to_bytes(algo.digest(file).unwrap(), width),
        );
    }
    Ok(sums)
}

pub fn find_checksum(strspec: &str, bytes: &[&[u8]]) -> Result<Vec<Vec<u8>>, CheckBuilderErr> {
    let (prefix, width, rest) = find_prefix_width(strspec)?;
    match (width, prefix) {
        (1..=64, "crc") => get_checksums::<CRC<u64>>(rest, bytes, width),
        (65..=128, "crc") => get_checksums::<CRC<u128>>(rest, bytes, width),
        (1..=64, "modsum") => get_checksums::<ModSum<u64>>(rest, bytes, width),
        (2..=64, "polyhash") => get_checksums::<PolyHash<u64>>(rest, bytes, width),
        (1..=128, "fletcher") => get_checksums::<Fletcher<u64>>(rest, bytes, width),
        _ => Err(CheckBuilderErr::ValueOutOfRange("width")),
    }
}

enum BuilderEnum {
    Crc(CrcBuilder<u128>),
    ModSum(ModSumBuilder<u64>),
    Fletcher(FletcherBuilder<u64>),
    PolyHash(PolyHashBuilder<u64>),
}

pub struct AlgorithmFinder<'a> {
    pairs: Vec<(&'a [u8], Vec<u8>)>,
    spec: BuilderEnum,
    verbosity: u64,
    extended_search: bool,
}

impl<'a> AlgorithmFinder<'a> {
    fn iter_solutions<T, S: ToString, E, I: Iterator<Item = Result<S, E>>>(
        &self,
        x: &T,
        reverser: fn(&T, &[(&'a [u8], Vec<u8>)], u64, bool) -> I,
    ) -> impl Iterator<Item = Result<String, E>> + use<T, S, E, I> {
        reverser(x, &self.pairs, self.verbosity, self.extended_search)
            .map(|x| x.map(|y| y.to_string()))
    }

    #[cfg(feature = "parallel")]
    fn par_iter_solutions<T, S, E: Send + Sync, I: ParallelIterator<Item = Result<S, E>>>(
        &self,
        x: &T,
        reverser: fn(&T, &[(&'a [u8], Vec<u8>)], u64, bool) -> I,
    ) -> impl ParallelIterator<Item = Result<String, E>> + use<T, S, E, I>
    where
        S: ToString,
    {
        reverser(x, &self.pairs, self.verbosity, self.extended_search)
            .map(|x| x.map(|y| y.to_string()))
    }

    pub fn find_all(&self) -> impl Iterator<Item = Result<String, CheckReverserError>> + use<'a> {
        let maybe_crc = if let BuilderEnum::Crc(crc) = &self.spec {
            Some(self.iter_solutions(crc, reverse_crc))
        } else {
            None
        };
        let maybe_modsum = if let BuilderEnum::ModSum(modsum) = &self.spec {
            Some(self.iter_solutions(modsum, reverse_modsum))
        } else {
            None
        };
        let maybe_fletcher = if let BuilderEnum::Fletcher(fletcher) = &self.spec {
            Some(self.iter_solutions(fletcher, reverse_fletcher))
        } else {
            None
        };
        let maybe_polyhash = if let BuilderEnum::PolyHash(polyhash) = &self.spec {
            Some(self.iter_solutions(polyhash, reverse_polyhash))
        } else {
            None
        };

        maybe_crc
            .into_iter()
            .flatten()
            .chain(maybe_modsum.into_iter().flatten())
            .chain(maybe_fletcher.into_iter().flatten())
            .chain(maybe_polyhash.into_iter().flatten())
    }

    #[cfg(feature = "parallel")]
    pub fn find_all_para(
        &self,
    ) -> impl ParallelIterator<Item = Result<String, CheckReverserError>> + use<'a> {
        let maybe_crc = if let BuilderEnum::Crc(crc) = &self.spec {
            Some(self.par_iter_solutions(crc, reverse_crc_para))
        } else {
            None
        };
        let maybe_modsum = if let BuilderEnum::ModSum(modsum) = &self.spec {
            Some(self.iter_solutions(modsum, reverse_modsum).par_bridge())
        } else {
            None
        };
        let maybe_fletcher = if let BuilderEnum::Fletcher(fletcher) = &self.spec {
            Some(self.par_iter_solutions(fletcher, reverse_fletcher_para))
        } else {
            None
        };
        let maybe_polyhash = if let BuilderEnum::PolyHash(polyhash) = &self.spec {
            Some(self.iter_solutions(polyhash, reverse_polyhash).par_bridge())
        } else {
            None
        };

        maybe_crc
            .into_par_iter()
            .flatten()
            .chain(maybe_modsum.into_par_iter().flatten())
            .chain(maybe_fletcher.into_par_iter().flatten())
            .chain(maybe_polyhash.into_par_iter().flatten())
    }
}

pub fn find_algorithm<'a>(
    strspec: &str,
    bytes: &[&'a [u8]],
    sums: &[Vec<u8>],
    verbosity: u64,
    extended_search: bool,
) -> Result<AlgorithmFinder<'a>, CheckBuilderErr> {
    let (prefix, _, rest) = find_prefix_width(strspec)?;
    let prefix = prefix.to_ascii_lowercase();
    let spec = match prefix.as_str() {
        "crc" => BuilderEnum::Crc(CrcBuilder::<u128>::from_str(rest)?),
        "modsum" => BuilderEnum::ModSum(ModSumBuilder::<u64>::from_str(rest)?),
        "fletcher" => BuilderEnum::Fletcher(FletcherBuilder::<u64>::from_str(rest)?),
        "polyhash" => BuilderEnum::PolyHash(PolyHashBuilder::<u64>::from_str(rest)?),
        _ => unimplemented!(),
    };
    match sums.len().cmp(&bytes.len()) {
        Ordering::Greater => {
            return Err(CheckBuilderErr::MissingParameter(
                "not enough files for checksums given",
            ));
        }
        Ordering::Less => {
            return Err(CheckBuilderErr::MissingParameter(
                "not enough checksums for files given",
            ));
        }
        Ordering::Equal => (),
    };
    let pairs: Vec<_> = bytes.iter().cloned().zip(sums.iter().cloned()).collect();
    Ok(AlgorithmFinder {
        pairs,
        spec,
        verbosity,
        extended_search,
    })
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn multibyte_part_range() {
        assert_eq!(
            find_checksum_segments(
                "modsum width=16 wordsize=24 module=0",
                &[vec![0u8; 15]],
                SegmentChecksum::Constant(&[vec![0, 0]]),
                SignedInclRange::new(0, 5).unwrap(),
                SignedInclRange::new(-5, -2).unwrap()
            ),
            Ok(vec![
                (vec![0, 3], vec![-4]),
                (vec![1, 4], vec![-3]),
                (vec![2, 5], vec![-5, -2])
            ])
        );
        assert_eq!(
            find_checksum_segments(
                "modsum width=16 wordsize=16 module=0 wordsize=16",
                &[vec![0u8; 15], vec![0u8; 12], vec![0u8; 9]],
                SegmentChecksum::Constant(&[vec![0, 0], vec![0, 0], vec![0, 0]]),
                SignedInclRange::new(0, 8).unwrap(),
                SignedInclRange::new(-9, -1).unwrap(),
            ),
            Ok(vec![])
        );
        assert_eq!(
            find_checksum_segments(
                "modsum width=16 wordsize=16 module=0 wordsize=24",
                &[vec![0u8; 15], vec![0u8; 12], vec![0u8; 9]],
                SegmentChecksum::Constant(&[vec![0, 0], vec![0, 0], vec![0, 0]]),
                SignedInclRange::new(0, 8).unwrap(),
                SignedInclRange::new(-9, -1).unwrap(),
            ),
            Ok(vec![
                (vec![0, 3, 6], vec![-7, -4, -1]),
                (vec![1, 4], vec![-6, -3]),
                (vec![2, 5], vec![-5, -2]),
            ])
        );
        assert_eq!(
            find_checksum_segments(
                "crc width=16 poly=1 wordsize=16 out_endian=little",
                &[
                    vec![0x6d, 0x79, 0x72, 0x3f, 0x00, 0x5d],
                    vec![0x75, 0x2d, 0xf4, 0xd4, 0xf5, 0xcf, 0xd8, 0x35]
                ],
                SegmentChecksum::Constant(&[vec![0x72, 0x3f], vec![0x01, 0x1b]]),
                SignedInclRange::new(0, 5).unwrap(),
                SignedInclRange::new(-7, -1).unwrap(),
            ),
            Ok(vec![(vec![2], vec![-3])])
        );
    }
    #[test]
    fn png_checksums() {
        let png = vec![
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x37, 0x6e, 0xf9, 0x24, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x01, 0x63, 0x60, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0x73, 0x75, 0x01, 0x18, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        assert_eq!(
            find_checksum_segments(
                "crc width=32 poly=0x04c11db7 init=0xffffffff refin=true refout=true xorout=0xffffffff out_endian=big",
                &[png.clone()],
                SegmentChecksum::FromEnd(0),
                SignedInclRange::new(0, png.len() as _).unwrap(),
                SignedInclRange::new(0, png.len() as _).unwrap(),
            ),
            Ok(vec![
                (vec![12], vec![28]),
                (vec![37], vec![50]),
                (vec![59], vec![62])
            ])
        );
    }
}

mod bitnum;
pub mod checksum;
mod keyval;
use bitnum::BitNum;
use checksum::CheckBuilderErr;
use checksum::{
    crc::{CRCBuilder, CRC},
    fletcher::Fletcher,
    modsum::ModSum,
    LinearCheck, RangePairs, Relativity,
};
use rayon::prelude::*;
use std::str::FromStr;
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
                    return usize::from_str_radix(&v, 10)
                        .map_err(|_| CheckBuilderErr::MalformedString(k))
                        .map(|width| (prefix, width, rest));
                }
            }
        }
    }
    Err(CheckBuilderErr::MissingParameter("width"))
}

/// A helper function for calling the find_segments function with strings arguments
fn find_segment_str<L>(
    spec: &str,
    bytes: &[Vec<u8>],
    sum: &str,
    rel: Relativity,
) -> Result<RangePairs, CheckBuilderErr>
where
    L: LinearCheck + FromStr<Err = CheckBuilderErr>,
    L::Sum: BitNum,
{
    let sum_array = sum
        .split(|x| x == ',')
        .map(L::Sum::from_hex)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| CheckBuilderErr::MalformedString(String::default()))?;
    Ok(L::from_str(spec)?.find_segments(bytes, &sum_array, rel))
}

/// The available checksum types
static PREFIXES: &[&str] = &["fletcher", "crc", "modsum"];

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
    sum: &str,
    rel: Relativity,
) -> Result<RangePairs, CheckBuilderErr> {
    let (prefix, width, rest) = find_prefix_width(strspec)?;
    match (width, prefix) {
        (1..=8, "crc") => find_segment_str::<CRC<u8>>(rest, bytes, sum, rel),
        (9..=16, "crc") => find_segment_str::<CRC<u16>>(rest, bytes, sum, rel),
        (17..=32, "crc") => find_segment_str::<CRC<u32>>(rest, bytes, sum, rel),
        (33..=64, "crc") => find_segment_str::<CRC<u64>>(rest, bytes, sum, rel),
        (65..=128, "crc") => find_segment_str::<CRC<u128>>(rest, bytes, sum, rel),
        (1..=8, "modsum") => find_segment_str::<ModSum<u8>>(rest, bytes, sum, rel),
        (9..=16, "modsum") => find_segment_str::<ModSum<u16>>(rest, bytes, sum, rel),
        (17..=32, "modsum") => find_segment_str::<ModSum<u32>>(rest, bytes, sum, rel),
        (33..=64, "modsum") => find_segment_str::<ModSum<u64>>(rest, bytes, sum, rel),
        (1..=8, "fletcher") => find_segment_str::<Fletcher<u8>>(rest, bytes, sum, rel),
        (9..=16, "fletcher") => find_segment_str::<Fletcher<u16>>(rest, bytes, sum, rel),
        (17..=32, "fletcher") => find_segment_str::<Fletcher<u32>>(rest, bytes, sum, rel),
        (33..=64, "fletcher") => find_segment_str::<Fletcher<u64>>(rest, bytes, sum, rel),
        (65..=128, "fletcher") => find_segment_str::<Fletcher<u128>>(rest, bytes, sum, rel),
        _ => Err(CheckBuilderErr::ValueOutOfRange("width")),
    }
}

enum BuilderEnum {
    CRC(CRCBuilder<u128>),
}

pub struct AlgorithmFinder<'a> {
    pairs: Vec<(&'a [u8], u128)>,
    spec: BuilderEnum,
    verbosity: u64,
}

impl<'a> AlgorithmFinder<'a> {
    pub fn find_all<'b>(&'b self) -> impl Iterator<Item = String> + 'b {
        match &self.spec {
            BuilderEnum::CRC(crc) => {
                checksum::crc::rev::reverse_crc(crc, self.pairs.as_slice(), self.verbosity)
                    .map(|x| x.to_string())
            }
        }
    }
    pub fn find_all_para<'b>(&'b self) -> impl ParallelIterator<Item = String> + 'b {
        match &self.spec {
            BuilderEnum::CRC(crc) => {
                checksum::crc::rev::reverse_crc_para(crc, self.pairs.as_slice(), self.verbosity)
                    .map(|x| x.to_string())
            }
        }
    }
}

pub fn find_algorithm<'a>(
    strspec: &str,
    bytes: &'a [&[u8]],
    sum: &str,
    verbosity: u64,
) -> Result<AlgorithmFinder<'a>, CheckBuilderErr> {
    let (prefix, _, rest) = find_prefix_width(strspec)?;
    if prefix != "crc" {
        unimplemented!()
    }
    let spec = CRCBuilder::<u128>::from_str(rest)?;
    let sums = sum
        .split(|x| x == ',')
        .map(u128::from_hex)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| CheckBuilderErr::MalformedString(String::default()))?;
    if sums.len() != bytes.len() {
        panic!("Help how do I error handle this?")
    }
    let pairs: Vec<_> = bytes.into_iter().cloned().zip(sums.into_iter()).collect();
    Ok(AlgorithmFinder {
        spec: BuilderEnum::CRC(spec),
        pairs,
        verbosity,
    })
}

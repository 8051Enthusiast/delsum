mod bitnum;
pub mod checksum;
mod keyval;
use bitnum::BitNum;
use checksum::CheckBuilderErr;
use checksum::{crc::CRC, fletcher::Fletcher, modsum::ModSum, LinearCheck, RangePairs, Relativity};
use std::str::FromStr;

/// For figuring out what type of integer to use, we need to parse the width from the
/// model string, but to parse the model string, we need to know the integer type,
/// so it is done here separately.
fn find_width(s: &str) -> Result<usize, CheckBuilderErr> {
    for x in keyval::KeyValIter::new(s) {
        match x {
            Err(k) => return Err(CheckBuilderErr::MalformedString(k)),
            Ok((k, v)) => {
                if &k == "width" {
                    return usize::from_str_radix(&v, 10)
                        .map_err(|_| CheckBuilderErr::MalformedString(k));
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
    let mut sum_array = Vec::new();
    for s in sum.split(|x| x == ',') {
        sum_array.push(
            L::Sum::from_dec_or_hex(s)
                .map_err(|_| CheckBuilderErr::MalformedString(String::default()))?,
        );
    }
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
    let stripped = strspec.trim_start();
    // it is done like this to ensure that no non-whitespace (blackspace?) is left at the end of the prefix
    let pref = stripped.split_whitespace().next();
    let (prefix, rest) = match PREFIXES.iter().find(|x| Some(**x) == pref) {
        Some(p) => (*p, &stripped[p.len()..]),
        None => return Err(CheckBuilderErr::MalformedString(String::default())),
    };
    let width = find_width(rest)?;
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

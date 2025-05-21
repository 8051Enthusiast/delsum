wit_bindgen::generate!({
    world: "delsum",
});

struct Delsum;

use std::fmt::Write;

use delsum_lib::{
    checksum::{CheckBuilderErr, CheckReverserError},
    utils::SignedInclRange,
    DelsumError, SegmentChecksum,
};
use exports::delsum::web::checksums::*;

impl From<DelsumError> for ChecksumError {
    fn from(err: DelsumError) -> Self {
        match err {
            DelsumError::ModelError(check_builder_err) => {
                ChecksumError::Model(check_builder_err.to_string())
            }
            DelsumError::ChecksumCountMismatch(msg) => ChecksumError::Other(msg.to_string()),
        }
    }
}

impl From<CheckReverserError> for ChecksumError {
    fn from(value: CheckReverserError) -> Self {
        match value {
            CheckReverserError::MissingParameter(err) => ChecksumError::Model(err.to_string()),
            _ => ChecksumError::Other(value.to_string()),
        }
    }
}

impl From<CheckBuilderErr> for ChecksumError {
    fn from(value: CheckBuilderErr) -> Self {
        ChecksumError::Model(value.to_string())
    }
}

impl Guest for Delsum {
    fn reverse(
        files: Vec<ChecksummedFile>,
        model: String,
        trailing_check: bool,
    ) -> Result<Vec<String>, ChecksumError> {
        let bytes = files.iter().map(|x| x.file.as_slice()).collect::<Vec<_>>();
        let sums: Vec<Vec<u8>>;
        let segment_checksums = if trailing_check {
            SegmentChecksum::FromEnd(0)
        } else {
            sums = files.iter().map(|x| x.checksum.clone()).collect::<Vec<_>>();
            SegmentChecksum::Constant(&sums)
        };
        let result = delsum_lib::find_algorithm(&model, &bytes, segment_checksums, 0, false);
        let matches = match result {
            Ok(m) => m,
            Err(err) => return Err(err.into()),
        };
        matches
            .find_all()
            .map(|x| x.map_err(|e| e.into()))
            .collect()
    }

    fn part(
        files: Vec<ChecksummedFile>,
        model: String,
        trailing_check: bool,
        end_relative: bool,
    ) -> Result<Vec<ChecksumRanges>, ChecksumError> {
        if files.is_empty() {
            return Ok(vec![]);
        }
        let sums: Vec<Vec<u8>>;
        let segment_checksums = if trailing_check {
            SegmentChecksum::FromEnd(0)
        } else {
            sums = files.iter().map(|x| x.checksum.clone()).collect::<Vec<_>>();
            SegmentChecksum::Constant(&sums)
        };
        let bytes = files.into_iter().map(|x| x.file).collect::<Vec<_>>();
        let min_len = bytes.iter().map(|x| x.len()).min().unwrap() as isize;
        if min_len == 0 {
            return Ok(vec![]);
        }

        let start_range = SignedInclRange::new(0isize, min_len - 1).unwrap();
        let end_range = if end_relative {
            SignedInclRange::new(-min_len, -1).unwrap()
        } else {
            start_range
        };

        match delsum_lib::find_checksum_segments(
            &model,
            &bytes,
            segment_checksums,
            start_range,
            end_range,
        ) {
            Ok(res) => Ok(res
                .into_iter()
                .map(|(start, end)| ChecksumRanges {
                    start: start.into_iter().map(|x| x as i32).collect(),
                    end: end.into_iter().map(|x| x as i32).collect(),
                })
                .collect()),
            Err(err) => Err(err.into()),
        }
    }

    fn check(file: Vec<u8>, model: String) -> Result<String, ChecksumError> {
        let checksum = &delsum_lib::find_checksum(&model, &[file.as_slice()])?[0];
        let mut res = String::with_capacity(checksum.len() * 2);
        for byte in checksum {
            write!(&mut res, "{:02x}", byte).unwrap();
        }
        Ok(res)
    }
}

export!(Delsum);

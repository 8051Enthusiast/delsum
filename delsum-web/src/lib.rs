wit_bindgen::generate!({
    world: "delsum",
});

struct Delsum;

use delsum_lib::{utils::SignedInclRange, SegmentChecksum};
use exports::delsum::web::checksums::*;

impl Guest for Delsum {
    fn reverse(files: Vec<ChecksummedFile>, model: String) -> Result<Vec<String>, String> {
        let bytes = files.iter().map(|x| x.file.as_slice()).collect::<Vec<_>>();
        let sums = files.iter().map(|x| x.checksum.clone()).collect::<Vec<_>>();
        let result =
            delsum_lib::find_algorithm(&model, &bytes, SegmentChecksum::Constant(&sums), 0, false);
        let matches = match result {
            Ok(m) => m,
            Err(err) => return Err(err.to_string()),
        };
        matches
            .find_all()
            .map(|x| x.map_err(|e| e.to_string()))
            .collect()
    }

    fn part(files: Vec<ChecksummedFile>, model: String) -> Result<Vec<ChecksumRanges>, String> {
        if files.is_empty() {
            return Ok(vec![]);
        }
        let sums = files.iter().map(|x| x.checksum.clone()).collect::<Vec<_>>();
        let bytes = files.into_iter().map(|x| x.file).collect::<Vec<_>>();
        let min_len = bytes.iter().map(|x| x.len()).min().unwrap() as isize;
        if min_len == 0 {
            return Ok(vec![]);
        }

        match delsum_lib::find_checksum_segments(
            &model,
            &bytes,
            SegmentChecksum::Constant(&sums),
            SignedInclRange::new(0isize, min_len - 1).unwrap(),
            SignedInclRange::new(0isize, min_len - 1).unwrap(),
        ) {
            Ok(res) => Ok(res
                .into_iter()
                .map(|(start, end)| ChecksumRanges {
                    start: start.into_iter().map(|x| x as i32).collect(),
                    end: end.into_iter().map(|x| x as i32).collect(),
                })
                .collect()),
            Err(err) => Err(err.to_string()),
        }
    }
}

export!(Delsum);

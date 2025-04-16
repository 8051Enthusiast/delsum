wit_bindgen::generate!({
    // the name of the world in the `*.wit` input file
    world: "delsum",
});

struct Delsum;

impl exports::delsum::web::checksums::Guest for Delsum {
    fn reverse(
        files: Vec<exports::delsum::web::checksums::ChecksummedFile>,
        model: String,
    ) -> Result<String, String> {
        let bytes = files.iter().map(|x| x.file.as_slice()).collect::<Vec<_>>();
        let sums = files.iter().map(|x| x.checksum.clone()).collect::<Vec<_>>();
        let result = delsum_lib::find_algorithm(&model, &bytes, &sums, 0, false);
        let matches = match result {
            Ok(m) => m,
            Err(err) => return Err(err.to_string()),
        };
        matches
            .find_all()
            .map(|x| x.map_err(|e| e.to_string()))
            .collect()
    }
}

export!(Delsum);

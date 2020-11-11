use clap::{App, Arg, ArgGroup};
use delsum::checksum::{RelativeIndex, Relativity};
use delsum::{find_algorithm, find_checksum_segments};
//use rayon::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::process::exit;

fn main() {
    let matches = App::new("delsum")
        .version("0.1.0")
        .author("8051Enthusiast <8051enthusiast@protonmail.com>")
        .about("Finds segments with given checksums inside files")
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .multiple(true)
                .help("put some info on what the program is currently doing to stderr")
        )
        .arg(
            Arg::with_name("model")
                .short("m")
                .long("model")
                .value_name("MODEL STRING")
                .help("use the checksum algorithm given by the model string")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("modelfile")
                .short("M")
                .long("modelfile")
                .value_name("FILE")
                .help("read model strings line-by-line from given file"),
        )
        .group(
            ArgGroup::with_name("models")
                .arg("model")
                .arg("modelfile")
                .required(true),
        )
        .arg(
            Arg::with_name("start")
                .help("sets the end of the checksum segments to be relative to the start of the file (default)")
                .long("start")
                .short("s")
        )
        .arg(
            Arg::with_name("end")
                .help("sets the end of the checksum segments to be relative to the end of the file")
                .long("end")
                .short("e")
        )
        .group(
            ArgGroup::with_name("relativity")
                .arg("start")
                .arg("end")
        )
        .arg(
            Arg::with_name("parallel")
                .help("try doing more parallelism, in turn using more memory")
                .long("parallel")
                .short("p")
        )
        .arg(
            Arg::with_name("checksums")
                .help("a comma-separated list of checksums to match")
                .short("c")
                .long("checksum")
                .value_name("CHECKSUMS")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("reverse")
                .help("reverse")
                .short("r")
                .long("reverse")
        )
        .arg(
            Arg::with_name("files")
                .help("the files to find checksummed segments of")
                .index(1)
                .min_values(1),
        )
        .get_matches();
    let verbosity = matches.occurrences_of("verbose");
    let files = matches.values_of_os("files").unwrap();
    let mut bytes = Vec::new();
    for file in files {
        let mut current_bytes = Vec::new();
        File::open(file)
            .unwrap_or_else(|err| {
                eprintln!("Could not open file '{}': {}", file.to_string_lossy(), err);
                exit(1);
            })
            .read_to_end(&mut current_bytes)
            .unwrap_or_else(|err| {
                eprintln!("Could not read file '{}': {}", file.to_string_lossy(), err);
                exit(1);
            });
        bytes.push(current_bytes);
    }
    let models = matches.value_of_os("modelfile").map_or_else(
        || vec![matches.value_of("model").map(String::from).unwrap()],
        |file| {
            let mut s = String::new();
            File::open(file)
                .unwrap_or_else(|err| {
                    eprintln!("Could not open file '{}': {}", file.to_string_lossy(), err);
                    exit(1);
                })
                .read_to_string(&mut s)
                .unwrap_or_else(|err| {
                    eprintln!("Could not read file '{}': {}", file.to_string_lossy(), err);
                    exit(1);
                });
            s.lines()
                .filter(|x| !x.is_empty() && !x.starts_with('#'))
                .map(String::from)
                .collect()
        },
    );
    let checksums = matches.value_of("checksums").unwrap();
    let rel = if matches.is_present("end") {
        Relativity::End
    } else {
        Relativity::Start
    };
    let parallel = matches.is_present("parallel");
    let reverse = matches.is_present("reverse");
    let byte_slices: Vec<_> = bytes.iter().map(Vec::<u8>::as_slice).collect();
    let subsum_print = |model| {
        let segs = find_checksum_segments(model, &bytes, checksums, rel).unwrap_or_else(|err| {
            eprintln!("Could not process model '{}': {}", model, err);
            exit(1);
        });
        if !segs.is_empty() {
            println!("{}:", model);
            for (a, b) in segs {
                let a_list = a
                    .iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<_>>()
                    .join(",");
                let b_list = b
                    .iter()
                    .map(|x| match x {
                        RelativeIndex::FromStart(n) => format!("{}", n),
                        RelativeIndex::FromEnd(n) => format!("-{}", n),
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                println!("\t{}:{}", a_list, b_list);
            }
        }
    };
    let algorithms = |model: &str| {
        find_algorithm(model, &byte_slices, checksums, verbosity).unwrap_or_else(|err| {
            eprintln!("Could not process model '{}': {}", model, err);
            exit(1);
        })
    };
    match (reverse, parallel) {
        (true, true) => {
            models.par_iter().for_each(|x|
                algorithms(x).find_all_para().for_each(|algo| {
                    match algo {
                        Ok(a) => println!("{}", a),
                        Err(e) => eprintln!("Error on {}: {}", x, e),
                    }
                })
            );
        }
        (true, false) => {
            models.iter().for_each(|x|
                algorithms(x).find_all().for_each(|algo| {
                    match algo {
                        Ok(a) => println!("{}", a),
                        Err(e) => eprintln!("Error on {}: {}", x, e),
                    }
                })
            );
        }
        (false, true) => {
            models.par_iter().map(|x| x.as_str()).for_each(subsum_print);
        }
        (false, false) => {
            models.iter().map(|x| x.as_str()).for_each(subsum_print);
        }
    }
}

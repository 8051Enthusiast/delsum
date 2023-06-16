use delsum_lib::utils::{read_signed_maybe_hex, SignedInclRange};
use delsum_lib::{find_algorithm, find_checksum, find_checksum_segments};
use hex::{FromHex, FromHexError, ToHex};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::process::exit;
use std::sync::Mutex;
use structopt::StructOpt;

fn main() {
    let opt = Opt::from_args();
    match opt {
        Opt::Part(p) => part(&p),
        Opt::Reverse(r) => reverse(&r),
        Opt::Check(c) => check(&c),
    }
}

fn reverse(opts: &Reverse) {
    let files = read_files(&opts.files);
    let start = opts.start.unwrap_or(0);
    let end = opts.end.unwrap_or(-1);
    let ranged_files: Vec<_> = apply_range_to_file(&files, start, end);
    let models = read_models(&opts.model, &opts.model_file);
    let checksums = match read_checksums(&opts.checksums) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("Could not read checksums: {}", e);
            exit(1);
        }
    };
    let algorithms = |model: &str| {
        find_algorithm(
            &model,
            &ranged_files,
            &checksums,
            opts.verbose,
            opts.extended_search,
        )
        .unwrap_or_else(|err| {
            eprintln!("Could not process model '{}': {}", model, err);
            exit(1);
        })
    };
    let error_set = Mutex::new(HashSet::new());
    #[cfg(feature = "parallel")]
    let parallel = opts.parallel;
    #[cfg(not(feature = "parallel"))]
    let parallel = false;

    match parallel {
        true => {
            #[cfg(feature = "parallel")]
            models.par_iter().for_each(|x| {
                algorithms(x).find_all_para().for_each(|algo| match algo {
                    Ok(a) => println!("{}", a),
                    Err(e) => {
                        let do_print = error_set
                            .lock()
                            .map(|mut set| set.insert((x.clone(), e.clone())))
                            .unwrap_or(true);
                        if do_print {
                            eprintln!("Error on {}: {}", x, e)
                        }
                    }
                })
            });
        }
        false => {
            models.iter().for_each(|x| {
                algorithms(x).find_all().for_each(|algo| match algo {
                    Ok(a) => println!("{}", a),
                    Err(e) => {
                        let do_print = error_set
                            .lock()
                            .map(|mut set| set.insert((x.clone(), e.clone())))
                            .unwrap_or(true);
                        if do_print {
                            eprintln!("Error on {}: {}", x, e)
                        }
                    }
                })
            });
        }
    }
}

fn part(opts: &Part) {
    let files = read_files(&opts.files);
    let models = read_models(&opts.model, &opts.model_file);
    let min_len = files.iter().map(|x| x.len()).min().unwrap();
    let end_range = match (opts.start, opts.end, opts.end_range) {
        (true, false, None) => SignedInclRange::new(0, min_len as isize - 1),
        (false, _, None) => SignedInclRange::new(-(min_len as isize), -1),
        (false, false, Some(r)) => Some(r),
        _ => {
            eprintln!("Error: arguments must contain at most one of -e, -E, -s");
            exit(1);
        }
    };
    let start_range = opts
        .start_range
        .map(Some)
        .unwrap_or_else(|| SignedInclRange::new(0, min_len as isize - 1));
    let checksums = match read_checksums(&opts.checksums) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("Could not read checksums: {}", e);
            exit(1);
        }
    };
    if min_len < 1 {
        eprintln!("Warning: file of zero size, no ranges fonud");
        return;
    }
    let start_range = start_range.unwrap();
    let end_range = end_range.unwrap();
    #[cfg(feature = "parallel")]
    let parallel = opts.parallel;
    #[cfg(not(feature = "parallel"))]
    let parallel = false;
    let subsum_print = |model| {
        let segs = find_checksum_segments(model, &files, &checksums, start_range, end_range)
            .unwrap_or_else(|err| {
                eprintln!("Could not process model '{}': {}", model, err);
                exit(1);
            });
        if !segs.is_empty() {
            let mut list = String::new();
            list.push_str(&format!("{}:\n", model));
            for (a, b) in segs {
                let a_list = a
                    .iter()
                    .map(|x| match x >= &0 {
                        true => format!("0x{:x}", x),
                        false => format!("-0x{:x}", -x),
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                let b_list = b
                    .iter()
                    .map(|x| match x >= &0 {
                        true => format!("0x{:x}", x),
                        false => format!("-0x{:x}", -x),
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                list.push_str(&format!("\t{}:{}\n", a_list, b_list));
            }
            print!("{}", list);
        }
    };
    match parallel {
        true => {
            #[cfg(feature = "parallel")]
            models.par_iter().map(|x| x.as_str()).for_each(subsum_print);
        }
        false => {
            models.iter().map(|x| x.as_str()).for_each(subsum_print);
        }
    };
}

fn check(opts: &Check) {
    let files = read_files(&opts.files);
    let start = opts.start.unwrap_or(0);
    let end = opts.end.unwrap_or(-1);
    let ranged_files: Vec<_> = apply_range_to_file(&files, start, end);
    let models = read_models(&opts.model, &opts.model_file);
    let is_single = models.len() <= 1;
    #[cfg(feature = "parallel")]
    let parallel = opts.parallel;
    #[cfg(not(feature = "parallel"))]
    let parallel = false;
    let print_sums = |model| {
        let checksums = find_checksum(model, &ranged_files).unwrap_or_else(|err| {
            eprintln!("Could not process model '{}': {}", model, err);
            exit(1);
        });
        if is_single {
            println!(
                "{}",
                checksums
                    .iter()
                    .map(|x| x.encode_hex())
                    .collect::<Vec<String>>()
                    .join(",")
            )
        } else {
            println!(
                "{}: {}",
                model,
                checksums
                    .iter()
                    .map(|x| x.encode_hex())
                    .collect::<Vec<String>>()
                    .join(",")
            )
        }
    };
    match parallel {
        true => {
            #[cfg(feature = "parallel")]
            models.par_iter().for_each(|x| print_sums(x));
        }
        false => {
            models.iter().for_each(|x| print_sums(x));
        }
    }
}

/// Takes the slices of files corresponding to the indices of start and end
fn apply_range_to_file(files: &[Vec<u8>], start: isize, end: isize) -> Vec<&[u8]> {
    files
        .iter()
        .map(|x| {
            let range = SignedInclRange::new(start, end)
                .and_then(|range| range.to_unsigned(x.len()))
                .unwrap_or_else(|| {
                    eprintln!("Error: Range from {} to {} is too big or the start is after the end", start, end);
                    exit(1)
                });
            &x[range.start()..=range.end()]
        })
        .collect()
}

/// reads a bunch of checksums in hex separated by ','
fn read_checksums(s: &str) -> Result<Vec<Vec<u8>>, FromHexError> {
    s.split(',').map(|x| x.trim()).map(Vec::from_hex).collect()
}

#[derive(Debug, StructOpt)]
enum Opt {
    Part(Part),
    Reverse(Reverse),
    Check(Check),
}

/// With given checksum algorithm and checksums, find parts of the file matching the checksum
#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Part {
    /// Print some messages indicating progress
    #[structopt(short, long, parse(from_occurrences))]
    #[allow(unused)]
    verbose: u64,
    /// Sets the end of the checksum segments to be relative to the start of the file
    #[structopt(short, long)]
    start: bool,
    /// Sets the end of the checksum segments to be relative to the end of the file (default)
    #[structopt(short, long)]
    end: bool,
    /// The inclusive range of numbers where a checksum may start in format [number]:[number] where [number]
    /// is a signed hexadecimal and negative numbers indicate offsets relative from the end
    #[structopt(short = "S", long)]
    start_range: Option<delsum_lib::utils::SignedInclRange>,
    /// The inclusive range of numbers where a checksum may end in format [number]:[number] where [number]
    /// is a signed hexadecimal and negative numbers indicate offsets relative from the end
    #[structopt(short = "E", long)]
    end_range: Option<delsum_lib::utils::SignedInclRange>,
    /// Do more parallelism, in turn using more memory
    #[structopt(short, long)]
    parallel: bool,
    /// Use the checksum algorithm given by the model string
    #[structopt(short, long)]
    model: Option<String>,
    /// Read model strings line-by-line from given file
    #[structopt(short = "M", long)]
    model_file: Option<OsString>,
    /// A comma separated list of checksums, each corresponding to a file
    #[structopt(short, long)]
    checksums: String,
    /// The files of which to find checksummed parts
    files: Vec<OsString>,
}

/// From given files and checksums, find out the checksum algorithms
#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Reverse {
    /// Print some messages indicating progress
    #[structopt(short, long, parse(from_occurrences))]
    verbose: u64,
    /// Do more parallelism, in turn using more memory
    #[structopt(short, long)]
    parallel: bool,
    /// Use the checksum algorithm given by the model string
    #[structopt(short, long)]
    model: Option<String>,
    /// The hexadecimal offset of the first byte to be checksummed (can be negative to indicate offset from end)
    #[structopt(short = "S", long, parse(try_from_str = read_signed_maybe_hex))]
    start: Option<isize>,
    /// The hexadecimal offset of the last byte to be checksummed (can be negative to indicate offset from end)
    #[structopt(short = "E", long, parse(try_from_str = read_signed_maybe_hex))]
    end: Option<isize>,
    /// Extend the search to parameter combinations that are unlikely
    #[structopt(short, long)]
    extended_search: bool,
    /// Read model strings line-by-line from given file
    #[structopt(short = "M", long)]
    model_file: Option<OsString>,
    /// A comma separated list of checksums, each corresponding to a file
    #[structopt(short, long)]
    checksums: String,
    /// The files of which to find checksummed parts
    files: Vec<OsString>,
}

/// From given files and algorithms, find out the checksums
#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Check {
    /// Print some messages indicating progress
    #[structopt(short, long, parse(from_occurrences))]
    #[allow(unused)]
    verbose: u64,
    /// Do more parallelism, in turn using more memory
    #[structopt(short, long)]
    parallel: bool,
    /// Use the checksum algorithm given by the model string
    #[structopt(short, long)]
    model: Option<String>,
    /// The hexadecimal offset of the first byte to be checksummed (can be negative to indicate offset from end)
    #[structopt(short = "S", long, parse(try_from_str = read_signed_maybe_hex))]
    start: Option<isize>,
    /// The hexadecimal offset of the last byte to be checksummed (can be negative to indicate offset from end)
    #[structopt(short = "E", long, parse(try_from_str = read_signed_maybe_hex))]
    end: Option<isize>,
    /// Read model strings line-by-line from given file
    #[structopt(short = "M", long)]
    model_file: Option<OsString>,
    /// The files of which to find checksummed parts
    files: Vec<OsString>,
}
fn read_models(model: &Option<String>, model_file: &Option<OsString>) -> Vec<String> {
    model_file.clone().map_or_else(
        || {
            let model = model.clone().unwrap_or_else(|| {
                eprintln!("Need at least one of -m/-M",);
                exit(1);
            });
            vec![model]
        },
        |file| {
            let mut s = String::new();
            File::open(&file)
                .unwrap_or_else(|err| {
                    eprintln!("Could not open file '{}': {}", &file.to_string_lossy(), err);
                    exit(1);
                })
                .read_to_string(&mut s)
                .unwrap_or_else(|err| {
                    eprintln!("Could not read file '{}': {}", &file.to_string_lossy(), err);
                    exit(1);
                });
            s.lines()
                .filter(|x| !x.is_empty() && !x.starts_with('#'))
                .map(String::from)
                .collect()
        },
    )
}

fn read_files(files: &[OsString]) -> Vec<Vec<u8>> {
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
    bytes
}

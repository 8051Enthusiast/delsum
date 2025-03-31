use clap::{Parser, Subcommand};
use delsum_lib::checksum::CheckReverserError;
use delsum_lib::utils::{read_signed_maybe_hex, SignedInclRange};
use delsum_lib::{find_algorithm, find_checksum, find_checksum_segments, SegmentChecksum};
use hex::{FromHex, FromHexError, ToHex};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::process::exit;
use std::sync::Mutex;

fn main() {
    let cli = Cli::parse();
    let opt = cli.command;
    match opt {
        Command::Part(p) => part(&p),
        Command::Reverse(r) => reverse(&r),
        Command::Check(c) => check(&c),
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
            model,
            &ranged_files,
            &checksums,
            opts.verbose as u64,
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
    let handle_errors = |model: &str, algo: Result<String, CheckReverserError>| match algo {
        Ok(a) => Some(a),
        Err(e) => {
            let do_print = error_set
                .lock()
                .map(|mut set| set.insert((model.to_string(), e.clone())))
                .unwrap_or(true);
            if do_print {
                eprintln!("Error on {}: {}", model, e)
            }
            None
        }
    };

    if opts.json {
        let algos = match parallel {
            #[cfg(feature = "parallel")]
            true => models
                .par_iter()
                .map(|model| (algorithms(model), model))
                .flat_map(|(r, model)| r.find_all_para().flat_map(move |a| handle_errors(model, a)))
                .collect::<Vec<_>>(),
            #[cfg(not(feature = "parallel"))]
            true => unreachable!(),
            false => models
                .iter()
                .map(|model| (algorithms(model), model))
                .flat_map(|(r, model)| r.find_all().flat_map(move |a| handle_errors(model, a)))
                .collect::<Vec<_>>(),
        };
        println!("{}", serde_json::to_string_pretty(&algos).unwrap());
    } else {
        match parallel {
            true => {
                #[cfg(feature = "parallel")]
                models.into_par_iter().for_each(|x| {
                    algorithms(&x)
                        .find_all_para()
                        .flat_map(|r| handle_errors(&x, r))
                        .for_each(|algo| println!("{}", algo));
                });
            }
            false => {
                models.into_iter().for_each(|x| {
                    algorithms(&x)
                        .find_all()
                        .flat_map(|r| handle_errors(&x, r))
                        .for_each(|algo| println!("{}", algo))
                });
            }
        }
    }
    if error_set.lock().unwrap().len() > 0 {
        exit(1);
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
    let checksums = if let Some(checksums) = &opts.checksums {
        match read_checksums(checksums) {
            Ok(x) => Some(x),
            Err(e) => {
                eprintln!("Could not read checksums: {}", e);
                exit(1);
            }
        }
    } else {
        None
    };
    if checksums.is_some() && opts.trailing.is_some() {
        eprintln!("Error: need exactly one of either -c or -t");
        exit(1);
    }
    let Some(checksums) = checksums
        .as_ref()
        .map(|x| SegmentChecksum::Constant(x))
        .or_else(|| Some(SegmentChecksum::FromEnd(opts.trailing?)))
    else {
        eprintln!("Error: need either -c or -t");
        exit(1);
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
        let segs = find_checksum_segments(model, &files, checksums, start_range, end_range)
            .unwrap_or_else(|err| {
                eprintln!("Could not process model '{}': {}", model, err);
                exit(1);
            });
        print_parts(segs, model);
    };
    let json_format = |model: &String| {
        let segs = find_checksum_segments(model, &files, checksums, start_range, end_range)
            .unwrap_or_else(|err| {
                eprintln!("Could not process model '{}': {}", model, err);
                exit(1);
            })
            .into_iter()
            .map(|(start, end)| PartExtends { start, end })
            .collect::<Vec<_>>();
        (model.clone(), segs)
    };
    if opts.json {
        let segs = match parallel {
            #[cfg(feature = "parallel")]
            true => models
                .par_iter()
                .map(json_format)
                .collect::<HashMap<_, _>>(),
            #[cfg(not(feature = "parallel"))]
            true => unreachable!(),
            false => models.iter().map(json_format).collect::<HashMap<_, _>>(),
        };
        println!("{}", serde_json::to_string_pretty(&segs).unwrap());
    } else {
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
}

fn print_parts(segs: Vec<(Vec<isize>, Vec<isize>)>, model: &str) {
    if !segs.is_empty() {
        let mut list = String::new();
        list.push_str(&format!("{}:\n", model));
        for (a, b) in segs {
            let num = |x: &isize| match *x >= 0 {
                true => format!("0x{:x}", x),
                false => format!("-0x{:x}", -x),
            };
            let a_list = a.iter().map(num).collect::<Vec<_>>().join(",");
            let b_list = b.iter().map(num).collect::<Vec<_>>().join(",");
            list.push_str(&format!("\t{}:{}\n", a_list, b_list));
        }
        print!("{}", list);
    }
}

#[derive(Serialize, PartialEq, Eq)]
struct PartExtends {
    start: Vec<isize>,
    end: Vec<isize>,
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
    let json_format = |model: &String| {
        let checksums = find_checksum(model, &ranged_files).unwrap_or_else(|err| {
            eprintln!("Could not process model '{}': {}", model, err);
            exit(1);
        });
        (
            model.clone(),
            checksums
                .iter()
                .map(|x| x.encode_hex())
                .collect::<Vec<String>>(),
        )
    };
    if opts.json {
        let checksums = match parallel {
            #[cfg(feature = "parallel")]
            true => models
                .par_iter()
                .map(json_format)
                .collect::<HashMap<_, _>>(),
            #[cfg(not(feature = "parallel"))]
            true => unreachable!(),
            false => models.iter().map(json_format).collect::<HashMap<_, _>>(),
        };
        println!("{}", serde_json::to_string_pretty(&checksums).unwrap());
    } else {
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
}

/// Takes the slices of files corresponding to the indices of start and end
fn apply_range_to_file(files: &[Vec<u8>], start: isize, end: isize) -> Vec<&[u8]> {
    files
        .iter()
        .map(|x| {
            let range = SignedInclRange::new(start, end)
                .and_then(|range| range.to_unsigned(x.len()))
                .unwrap_or_else(|| {
                    eprintln!(
                        "Error: Range from {} to {} is too big or the start is after the end",
                        start, end
                    );
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

#[derive(Debug, Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Part(Part),
    Reverse(Reverse),
    Check(Check),
}

/// With given checksum algorithm and checksums, find parts of the file matching the checksum
#[derive(Debug, Parser)]
struct Part {
    /// Print some messages indicating progress
    #[arg(short, long, action = clap::ArgAction::Count)]
    #[allow(unused)]
    verbose: u8,
    /// Output the results in JSON format
    #[arg(short, long)]
    json: bool,
    /// Sets the end of the checksum segments to be relative to the start of the file
    #[arg(short, long)]
    start: bool,
    /// Sets the end of the checksum segments to be relative to the end of the file (default)
    #[arg(short, long)]
    end: bool,
    /// The inclusive range of numbers where a checksum may start in format [number]:[number] where [number]
    /// is a signed hexadecimal and negative numbers indicate offsets relative from the end
    #[arg(short = 'S', long)]
    start_range: Option<delsum_lib::utils::SignedInclRange>,
    /// The inclusive range of numbers where a checksum may end in format [number]:[number] where [number]
    /// is a signed hexadecimal and negative numbers indicate offsets relative from the end
    #[arg(short = 'E', long)]
    end_range: Option<delsum_lib::utils::SignedInclRange>,
    /// Do more parallelism, in turn using more memory
    #[arg(short, long)]
    parallel: bool,
    /// Use the checksum algorithm given by the model string
    #[arg(short, long)]
    model: Option<String>,
    /// Read model strings line-by-line from given file
    #[arg(short = 'M', long)]
    model_file: Option<OsString>,
    /// A comma separated list of checksums, each corresponding to a file
    #[arg(short, long)]
    checksums: Option<String>,
    /// Instead of a constant list of checksums, use the bytes right after
    /// each checksummed region as the checksums, with n bytes of padding
    /// between the end of the checksummed region and the location of the
    /// checksum
    #[arg(short, long)]
    trailing: Option<usize>,
    /// The files of which to find checksummed parts
    files: Vec<OsString>,
}

/// From given files and checksums, find out the checksum algorithms
#[derive(Debug, Parser)]
struct Reverse {
    /// Print some messages indicating progress
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
    /// Output the results in JSON format
    #[arg(short, long)]
    json: bool,
    /// Do more parallelism, in turn using more memory
    #[arg(short, long)]
    parallel: bool,
    /// Use the checksum algorithm given by the model string
    #[arg(short, long)]
    model: Option<String>,
    /// The hexadecimal offset of the first byte to be checksummed (can be negative to indicate offset from end)
    #[arg(short = 'S', long, value_parser = read_signed_maybe_hex)]
    start: Option<isize>,
    /// The hexadecimal offset of the last byte to be checksummed (can be negative to indicate offset from end)
    #[arg(short = 'E', long, value_parser = read_signed_maybe_hex)]
    end: Option<isize>,
    /// Extend the search to parameter combinations that are unlikely
    #[arg(short, long)]
    extended_search: bool,
    /// Read model strings line-by-line from given file
    #[arg(short = 'M', long)]
    model_file: Option<OsString>,
    /// A comma separated list of checksums, each corresponding to a file
    #[arg(short, long)]
    checksums: String,
    /// The files of which to find checksummed parts
    files: Vec<OsString>,
}

/// From given files and algorithms, find out the checksums
#[derive(Debug, Parser)]
struct Check {
    /// Print some messages indicating progress
    #[arg(short, long, action = clap::ArgAction::Count)]
    #[allow(unused)]
    verbose: u8,
    /// Output the results in JSON format
    #[arg(short, long)]
    json: bool,
    /// Do more parallelism, in turn using more memory
    #[arg(short, long)]
    parallel: bool,
    /// Use the checksum algorithm given by the model string
    #[arg(short, long)]
    model: Option<String>,
    /// The hexadecimal offset of the first byte to be checksummed (can be negative to indicate offset from end)
    #[arg(short = 'S', long, value_parser = read_signed_maybe_hex)]
    start: Option<isize>,
    /// The hexadecimal offset of the last byte to be checksummed (can be negative to indicate offset from end)
    #[arg(short = 'E', long, value_parser = read_signed_maybe_hex)]
    end: Option<isize>,
    /// Read model strings line-by-line from given file
    #[arg(short = 'M', long)]
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

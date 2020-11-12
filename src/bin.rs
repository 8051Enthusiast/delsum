use delsum_lib::checksum::{RelativeIndex, Relativity};
use delsum_lib::{find_algorithm, find_checksum, find_checksum_segments};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::process::exit;
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
    let models = read_models(&opts.model, &opts.model_file);
    let byte_slices: Vec<_> = files.iter().map(Vec::<u8>::as_slice).collect();
    let algorithms = |model: &str| {
        find_algorithm(&model, &byte_slices, &opts.checksums, opts.verbose).unwrap_or_else(|err| {
            eprintln!("Could not process model '{}': {}", model, err);
            exit(1);
        })
    };
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
                    Err(e) => eprintln!("Error on {}: {}", x, e),
                })
            });
        }
        false => {
            models.iter().for_each(|x| {
                algorithms(x).find_all().for_each(|algo| match algo {
                    Ok(a) => println!("{}", a),
                    Err(e) => eprintln!("Error on {}: {}", x, e),
                })
            });
        }
    }
}

fn part(opts: &Part) {
    let files = read_files(&opts.files);
    let models = read_models(&opts.model, &opts.model_file);
    let rel = if opts.start {
        Relativity::Start
    } else {
        Relativity::End
    };
    #[cfg(feature = "parallel")]
    let parallel = opts.parallel;
    #[cfg(not(feature = "parallel"))]
    let parallel = false;
    let subsum_print = |model| {
        let segs =
            find_checksum_segments(model, &files, &opts.checksums, rel).unwrap_or_else(|err| {
                eprintln!("Could not process model '{}': {}", model, err);
                exit(1);
            });
        if !segs.is_empty() {
            let mut list = String::new();
            list.push_str(&format!("{}:\n", model));
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
    let models = read_models(&opts.model, &opts.model_file);
    let is_single = models.len() <= 1;
    #[cfg(feature = "parallel")]
    let parallel = opts.parallel;
    #[cfg(not(feature = "parallel"))]
    let parallel = false;
    let print_sums = |model| {
        let checksums = find_checksum(model, &files).unwrap_or_else(|err| {
            eprintln!("Could not process model '{}': {}", model, err);
            exit(1);
        });
        if is_single {
            println!("{}", checksums.join(","))
        } else {
            println!("{}: {}", model, checksums.join(","))
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
    verbose: u64,
    /// Sets the end of the checksum segments to be relative to the start of the file
    #[structopt(short, long)]
    start: bool,
    /// Sets the end of the checksum segments to be relative to the end of the file (default)
    #[structopt(short, long)]
    end: bool,
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
    verbose: u64,
    /// Do more parallelism, in turn using more memory
    #[structopt(short, long)]
    parallel: bool,
    /// Use the checksum algorithm given by the model string
    #[structopt(short, long)]
    model: Option<String>,
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

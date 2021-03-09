//! A simple modular sum over bytes (i.e. `bytes.sum() % module`)
//!
//! There are a number of parameters:
//! * width: The number of bits in the sum type, at most 64
//! * module: The sum is taken modulo this number
//! * init: The initial number
//! * check: The checksum of "123456789" (optional, gets checked at construction)
//! * name: An optional name that gets used for display purposes
//!
//! Note that a parameter to add at the end is not needed, since it is equivalent to `init`.
mod rev;
use crate::bitnum::Modnum;
use crate::checksum::{CheckBuilderErr, Digest, LinearCheck};
use crate::endian::{Endian, WordSpec};
use crate::keyval::KeyValIter;
use std::fmt::Display;
use std::str::FromStr;
pub use rev::reverse_modsum;
pub(crate) use rev::find_largest_mod;

/// A builder to set the various parameters for the modsum algorithm.
///
/// Example:
/// ```
/// # use delsum_lib::modsum::ModSum;
/// ModSum::<u8>::with_options()
///     .width(8)
///     .check(0xdd)
///     .build()
///     .is_ok();
/// ```
/// Note that module = 0 is assumed to be 2^width
#[derive(Debug, Clone)]
pub struct ModSumBuilder<S: Modnum> {
    width: Option<usize>,
    module: Option<S>,
    init: Option<S>,
    input_endian: Option<Endian>,
    output_endian: Option<Endian>,
    wordsize: Option<usize>,
    check: Option<S>,
    name: Option<String>,
}

impl<S: Modnum> ModSumBuilder<S> {
    /// The total width, in bits, of the sum. Mandatory.
    ///
    /// Can actually be wider than the sum itself, important is that `2^width >= module`.
    pub fn width(&mut self, w: usize) -> &mut Self {
        self.width = Some(w);
        self
    }
    /// The number by which the remainder is taken. Mandatory.
    ///
    /// If this is 0, it is equivalent to be `2^width`.
    pub fn module(&mut self, m: S) -> &mut Self {
        self.module = Some(m);
        self
    }
    /// The initial value, optional, defaults to 0.
    pub fn init(&mut self, i: S) -> &mut Self {
        self.init = Some(i);
        self
    }
    /// The endian of the words of the input file
    pub fn inendian(&mut self, e: Endian) -> &mut Self {
        self.input_endian = Some(e);
        self
    }
    /// The number of bits in a word of the input file
    pub fn wordsize(&mut self, n: usize) -> &mut Self {
        self.wordsize = Some(n);
        self
    }
    /// The endian of the checksum
    pub fn outendian(&mut self, e: Endian) -> &mut Self {
        self.output_endian = Some(e);
        self
    }
    /// The checksum of "123456789", gets checked on creation.
    pub fn check(&mut self, c: S) -> &mut Self {
        self.check = Some(c);
        self
    }
    /// An optional name that gets used for display purposes.
    pub fn name(&mut self, n: &str) -> &mut Self {
        self.name = Some(String::from(n));
        self
    }
    /// Builds the algorithm, after validating the parameters.
    pub fn build(&self) -> Result<ModSum<S>, CheckBuilderErr> {
        let width = self
            .width
            .ok_or(CheckBuilderErr::MissingParameter("width"))?;
        let mut module = self.module.unwrap_or_else(S::zero);
        if module == S::zero() && width < module.bits() {
            module = S::one() << width
        };
        let mut init = self.init.unwrap_or_else(S::zero);
        if module != S::zero() {
            init = init % module
        };
        let wordsize = self.wordsize.unwrap_or(8);
        if wordsize == 0 || wordsize % 8 != 0 || wordsize > 64 {
            return Err(CheckBuilderErr::ValueOutOfRange("wordsize"));
        }
        let wordspec = WordSpec {
            input_endian: self.input_endian.unwrap_or(Endian::Big),
            wordsize,
            output_endian: self.output_endian.unwrap_or(Endian::Big),
        };
        let s = ModSum {
            width,
            module,
            init,
            wordspec,
            name: self.name.clone(),
        };
        match self.check {
            Some(c) => {
                let mut sum = s.init();
                for &x in b"123456789" {
                    sum = s.dig_word(sum, x as u64);
                }
                s.finalize(sum);
                if sum == c {
                    return Ok(s);
                }
                Err(CheckBuilderErr::CheckFail)
            }
            None => Ok(s),
        }
    }
}

/// A Modsum checksum algorithm.
///
/// Implements LinearCheck so that finding checksummed locations in a file is efficiently possible.
#[derive(Debug, PartialEq, Eq)]
pub struct ModSum<S: Modnum> {
    width: usize,
    module: S,
    init: S,
    wordspec: WordSpec,
    name: Option<String>,
}

impl<S: Modnum> ModSum<S> {
    /// Creates a `ModSumBuilder`, for more information see its documentation.
    pub fn with_options() -> ModSumBuilder<S> {
        ModSumBuilder {
            width: None,
            module: None,
            init: None,
            input_endian: None,
            output_endian: None,
            wordsize: None,
            check: None,
            name: None,
        }
    }
}

impl<Sum: Modnum> Display for ModSum<Sum> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.name {
            Some(n) => write!(f, "{}", n),
            None => {
                write!(
                    f,
                    "modsum width={} module={:#x} init={:#x}",
                    self.width, self.module, self.init
                )?;
                if self.wordspec.word_bytes() != 1 {
                    write!(
                        f,
                        " in_endian={} wordsize={}",
                        self.wordspec.input_endian, self.wordspec.wordsize,
                    )?;
                };
                if self.width > 8 {
                    write!(f, " out_endian={}", self.wordspec.output_endian)?;
                }
                Ok(())
            }
        }
    }
}

impl<Sum: Modnum> FromStr for ModSumBuilder<Sum> {
    /// See FromStr for ModSum<Sum>
    fn from_str(s: &str) -> Result<ModSumBuilder<Sum>, CheckBuilderErr> {
        let mut sum = ModSum::<Sum>::with_options();
        for x in KeyValIter::new(s) {
            let (current_key, current_val) = match x {
                Err(key) => return Err(CheckBuilderErr::MalformedString(key)),
                Ok(s) => s,
            };
            let crc_op = match current_key.as_str() {
                "width" => usize::from_str(&current_val).ok().map(|x| sum.width(x)),
                "module" => Sum::from_hex(&current_val).ok().map(|x| sum.module(x)),
                "init" => Sum::from_hex(&current_val).ok().map(|x| sum.init(x)),
                "in_endian" => Endian::from_str(&current_val).ok().map(|x| sum.inendian(x)),
                "wordsize" => usize::from_str(&current_val).ok().map(|x| sum.wordsize(x)),
                "out_endian" => Endian::from_str(&current_val)
                    .ok()
                    .map(|x| sum.outendian(x)),
                "name" => Some(sum.name(&current_val)),
                _ => return Err(CheckBuilderErr::UnknownKey(current_key)),
            };
            match crc_op {
                Some(c) => sum = c.clone(),
                None => return Err(CheckBuilderErr::MalformedString(current_key)),
            }
        }
        Ok(sum)
    }
    type Err = CheckBuilderErr;
}

impl<Sum: Modnum> FromStr for ModSum<Sum> {
    /// Construct a new modular sum from a string specification.
    ///
    /// Example:
    ///
    /// width=16 module=65535 init=0
    fn from_str(s: &str) -> Result<ModSum<Sum>, CheckBuilderErr> {
        ModSumBuilder::from_str(s)?.build()
    }
    type Err = CheckBuilderErr;
}

impl<S: Modnum> Digest for ModSum<S> {
    type Sum = S;
    fn init(&self) -> Self::Sum {
        self.init
    }
    fn dig_word(&self, sum: Self::Sum, word: u64) -> Self::Sum {
        let modword = S::mod_from(word, &self.module);
        sum.add_mod(&modword, &self.module)
    }
    fn finalize(&self, sum: Self::Sum) -> Self::Sum {
        sum
    }
    fn to_bytes(&self, s: Self::Sum) -> Vec<u8> {
        self.wordspec.output_to_bytes(s, self.width)
    }

    fn wordspec(&self) -> WordSpec {
        self.wordspec
    }
}

impl<S: Modnum> LinearCheck for ModSum<S> {
    type Shift = ();
    // shifts are trivial in this checksum type
    fn init_shift(&self) -> Self::Shift {}
    fn inc_shift(&self, _: Self::Shift) -> Self::Shift {}
    fn shift(&self, sum: Self::Sum, _: &Self::Shift) -> Self::Sum {
        sum
    }
    fn shift_n(&self, _: usize) -> Self::Shift {}
    fn add(&self, sum_a: Self::Sum, sum_b: &Self::Sum) -> Self::Sum {
        sum_a.add_mod(sum_b, &self.module)
    }
    fn negate(&self, sum: Self::Sum) -> Self::Sum {
        if sum == S::zero() {
            sum
        } else if self.module == S::zero() {
            // this is just -sum in the underlying type, but I don't have
            // wrapping sub, so this should work as a substitute
            !sum + S::one()
        } else {
            self.module - sum
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::tests::{test_prop, test_shifts};
    use crate::checksum::{RelativeIndex, Relativity};
    #[test]
    fn modsum_8() {
        let s = ModSum::<u8>::with_options()
            .width(8)
            .check(0xdd)
            .build()
            .unwrap();
        test_shifts(&s);
        test_prop(&s);
        let s = ModSum::<u16>::with_options()
            .width(8)
            .check(0xdd)
            .build()
            .unwrap();
        test_shifts(&s);
        test_prop(&s);
    }
    #[test]
    fn mod_17() {
        let s = ModSum::<u8>::with_options()
            .width(5)
            .module(17)
            .check(1)
            .build()
            .unwrap();
        test_shifts(&s);
        test_prop(&s);
    }
    #[test]
    fn ethsum() {
        let chk = ModSum::<u16>::with_options()
            .width(16)
            .init(0xff00)
            .module(0xffff)
            .check(0xde)
            .build()
            .unwrap();
        test_shifts(&chk);
        test_prop(&chk);
        // 0xff*0x101 = 0xffff
        let many_255: Vec<_> = std::iter::repeat(0xffu8).take(0x101).collect();
        assert_eq!(chk.digest(many_255.as_slice()).unwrap(), 0xff00);
        let x = Vec::from("implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR ");
        let y = Vec::from("This program comes with ABSOLUTELY NO WARRANTY; for details type");
        let merchantibility = chk.digest(b" MERCHANTABILITY".as_ref()).unwrap();
        let ith_absolutely_ = chk.digest(b"with ABSOLUTELY ".as_ref()).unwrap();
        assert_eq!(
            chk.find_segments(
                &[x, y],
                &[merchantibility, ith_absolutely_],
                Relativity::Start
            ),
            vec![(vec![19], vec![RelativeIndex::FromStart(35)])]
        );
    }
}

//! A builder for a Fletcher-like algorithm.
//!
//! The basic functionality of this algorithm is:
//! * there is a sum which is just the bytes summed modulo some number
//! * there is also a second sum which the sum of all of the normal sums (modulo the same number)
//!
//! Note that text word sizes are currently only `u8`.
//!
//! It works roughly like this:
//! ```
//! # fn check(file: &[u8]) -> u32 {
//! # let module = 0xfff1u32;
//! # let init = 1;
//! # let (addout1, addout2) = (0, 0);
//! # let hwidth = 16;
//! let mut sum1 = init;
//! let mut sum2 = 0;
//! for byte in file {
//!     sum1 = (sum1 + *byte as u32) % module;
//!     sum2 = (sum2 + sum1) % module;
//! }
//! return (sum2 + addout2) % module << hwidth | (sum1 + addout1) % module;
//! # }
//! ```
//! Normally, the sum is represented as the cumulative sum bitshifted to be above the regular sum.
//! This representation will be referred to as "compact".
//!
//! These are the parameters:
//! * width: Total number of bits of the checksum (twice the amount of bits of the individual sums)
//! * module: The number by which both sums get reduced
//! * init: The initial value of the regular sum
//! * addout: The value that gets added at the end, compact
//! * swap: Whether to swap the values in the compact representation, i.e. put the regular sum above the cumulative sum
//! * check: The checksum of the bytes "123456789", checked to be correct on build
//! * name: The name to be used when displaying the algorithm (optional)
//!
//! Note that the `init` parameter, unlike the `addout` parameter, is not compact and is only added to the regular sum,
//! as for the cumulative sum, it is equivalent to the addout (so you can just add the cumulative `init` to the cumulative `addout`).

mod rev;
use crate::bitnum::{BitNum, Modnum};
use crate::checksum::{CheckBuilderErr, Digest, LinearCheck};
use crate::endian::{Endian, WordSpec};
use crate::keyval::KeyValIter;
use num_traits::{One, Zero};
pub use rev::reverse_fletcher;
#[cfg(feature = "parallel")]
pub use rev::reverse_fletcher_para;
use std::fmt::Display;
use std::str::FromStr;

/// A builder for a fletcher.
///
/// One can use it for specifying a fletcher algorithm, which can be used for checksumming.
///
/// Example:
/// ```
/// # use delsum_lib::fletcher::Fletcher;
/// let adler32 = Fletcher::<u32>::with_options()
///     .width(32)
///     .init(1)
///     .module(65521)
///     .check(0x091e01de)
///     .name("adler32")
///     .build()
///     .is_ok();
/// ```
#[derive(Clone, Debug)]
pub struct FletcherBuilder<Sum: Modnum> {
    width: Option<usize>,
    module: Option<Sum>,
    init: Option<Sum>,
    addout: Option<Sum::Double>,
    swap: Option<bool>,
    input_endian: Option<Endian>,
    output_endian: Option<Endian>,
    wordsize: Option<usize>,
    check: Option<Sum::Double>,
    name: Option<String>,
}

impl<S: Modnum> FletcherBuilder<S> {
    /// Sets the width of the type (both sums included, must be even, mandatory)
    pub fn width(&mut self, w: usize) -> &mut Self {
        self.width = Some(w);
        self
    }
    /// Sets the module of both sums (mandatory)
    pub fn module(&mut self, m: S) -> &mut Self {
        self.module = Some(m);
        self
    }
    /// Sets the initial value
    ///
    /// Contains one value for the regular sum.
    pub fn init(&mut self, i: S) -> &mut Self {
        self.init = Some(i);
        self
    }
    /// Sets a value that gets added after the checksum is finished
    ///
    /// Contains separate values for both sums, the cumulative one is bitshifted
    pub fn addout(&mut self, o: S::Double) -> &mut Self {
        self.addout = Some(o);
        self
    }
    /// Normally, the cumulative sum is saved on the higher bits and the normal sum in the lower bits.
    /// Setting this option to true swaps the positions.
    pub fn swap(&mut self, s: bool) -> &mut Self {
        self.swap = Some(s);
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
    /// Checks whether c is the same as the checksum of "123456789" on creation
    pub fn check(&mut self, c: S::Double) -> &mut Self {
        self.check = Some(c);
        self
    }
    /// A name to be displayed
    pub fn name(&mut self, n: &str) -> &mut Self {
        self.name = Some(String::from(n));
        self
    }
    /// Returns the Fletcher object after verifying correctness
    pub fn build(&self) -> Result<Fletcher<S>, CheckBuilderErr> {
        let init = self.init.unwrap_or_else(S::zero);
        let addout = self.addout.unwrap_or_else(S::Double::zero);
        // note: we only store the half width because it is more useful to us
        let hwidth = match self.width {
            None => return Err(CheckBuilderErr::MissingParameter("width")),
            Some(w) => {
                if w % 2 != 0 || w > addout.bits() {
                    return Err(CheckBuilderErr::ValueOutOfRange("width"));
                } else {
                    w / 2
                }
            }
        };

        let mask = (S::Double::one() << hwidth) - S::Double::one();
        let module = self.module.unwrap_or_else(S::zero);
        let wordsize = self.wordsize.unwrap_or(8);
        if wordsize == 0 || wordsize % 8 != 0 || wordsize > 64 {
            return Err(CheckBuilderErr::ValueOutOfRange("wordsize"));
        }
        let wordspec = WordSpec {
            input_endian: self.input_endian.unwrap_or(Endian::Big),
            wordsize,
            output_endian: self.output_endian.unwrap_or(Endian::Big),
        };
        let mut fletch = Fletcher {
            hwidth,
            module,
            init,
            addout,
            swap: self.swap.unwrap_or(false),
            wordspec,
            mask,
            name: self.name.clone(),
        };
        let (mut s, mut c) = fletch.from_compact(addout);
        if !module.is_zero() {
            s = s % module;
            c = c % module;
            fletch.init = init % module;
        } else {
            fletch.init = init;
        };
        fletch.addout = fletch.to_compact((s, c));
        match self.check {
            Some(chk) => {
                if fletch.digest(&b"123456789"[..]).unwrap() != chk {
                    println!("{:x?}", fletch.digest(&b"123456789"[..]).unwrap());
                    Err(CheckBuilderErr::CheckFail)
                } else {
                    Ok(fletch)
                }
            }
            None => Ok(fletch),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fletcher<Sum: Modnum> {
    hwidth: usize,
    module: Sum,
    init: Sum,
    addout: Sum::Double,
    swap: bool,
    wordspec: WordSpec,
    mask: Sum::Double,
    name: Option<String>,
}

impl<Sum: Modnum> Display for Fletcher<Sum> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.name {
            Some(n) => write!(f, "{}", n),
            None => {
                write!(
                    f,
                    "fletcher width={} module={:#x} init={:#x} addout={:#x} swap={}",
                    2 * self.hwidth,
                    self.module,
                    self.init,
                    self.addout,
                    self.swap
                )?;
                if self.wordspec.word_bytes() != 1 {
                    write!(
                        f,
                        " in_endian={} wordsize={}",
                        self.wordspec.input_endian, self.wordspec.wordsize
                    )?;
                };
                if self.hwidth * 2 > 8 {
                    write!(f, " out_endian={}", self.wordspec.output_endian)?;
                };
                Ok(())
            }
        }
    }
}

impl<Sum: Modnum> Fletcher<Sum> {
    /// Creates a `FletcherBuilder`, see `FletcherBuilder` documentation for more details.
    pub fn with_options() -> FletcherBuilder<Sum> {
        FletcherBuilder {
            width: None,
            module: None,
            init: None,
            addout: None,
            swap: None,
            input_endian: None,
            output_endian: None,
            wordsize: None,
            check: None,
            name: None,
        }
    }
    fn from_compact(&self, x: Sum::Double) -> (Sum, Sum) {
        let l = Sum::from_double(x & self.mask);
        let h = Sum::from_double((x >> self.hwidth) & self.mask);
        if self.swap {
            (h, l)
        } else {
            (l, h)
        }
    }
    fn to_compact(&self, (s, c): (Sum, Sum)) -> Sum::Double {
        let (l, h) = if self.swap { (c, s) } else { (s, c) };
        (Sum::Double::from(l) & self.mask) ^ (Sum::Double::from(h) & self.mask) << self.hwidth
    }
}

impl<Sum: Modnum> FromStr for FletcherBuilder<Sum> {
    /// See documentation of FromStr on Fletcher<Sum>
    fn from_str(s: &str) -> Result<FletcherBuilder<Sum>, CheckBuilderErr> {
        let mut fletch = Fletcher::<Sum>::with_options();
        for x in KeyValIter::new(s) {
            let (current_key, current_val) = match x {
                Err(key) => return Err(CheckBuilderErr::MalformedString(key)),
                Ok(s) => s,
            };
            let fletch_op = match current_key.as_str() {
                "width" => usize::from_str(&current_val).ok().map(|x| fletch.width(x)),
                "module" => Sum::from_hex(&current_val).ok().map(|x| fletch.module(x)),
                "init" => Sum::from_hex(&current_val).ok().map(|x| fletch.init(x)),
                "addout" => Sum::Double::from_hex(&current_val)
                    .ok()
                    .map(|x| fletch.addout(x)),
                "swap" => bool::from_str(&current_val).ok().map(|x| fletch.swap(x)),
                "in_endian" => Endian::from_str(&current_val)
                    .ok()
                    .map(|x| fletch.inendian(x)),
                "wordsize" => usize::from_str(&current_val)
                    .ok()
                    .map(|x| fletch.wordsize(x)),
                "out_endian" => Endian::from_str(&current_val)
                    .ok()
                    .map(|x| fletch.outendian(x)),
                "check" => Sum::Double::from_hex(&current_val)
                    .ok()
                    .map(|x| fletch.check(x)),
                "name" => Some(fletch.name(&current_val)),
                _ => return Err(CheckBuilderErr::UnknownKey(current_key)),
            };
            match fletch_op {
                Some(f) => fletch = f.clone(),
                None => return Err(CheckBuilderErr::MalformedString(current_key)),
            }
        }
        Ok(fletch)
    }
    type Err = CheckBuilderErr;
}

impl<Sum: Modnum> FromStr for Fletcher<Sum> {
    /// Construct a new fletcher sum algorithm from a string.
    /// Note that all parameters except width are in hexadecimal.
    ///
    /// Example:
    ///
    /// ```
    /// # use delsum_lib::fletcher::Fletcher;
    /// # use std::str::FromStr;
    /// Fletcher::<u32>::from_str("width=32 init=1 module=0xfff1 name=\"adler-32\"").is_ok();
    /// ```
    fn from_str(s: &str) -> Result<Fletcher<Sum>, CheckBuilderErr> {
        FletcherBuilder::<Sum>::from_str(s)?.build()
    }
    type Err = CheckBuilderErr;
}

impl<S: Modnum> Digest for Fletcher<S> {
    type Sum = S::Double;
    fn init(&self) -> Self::Sum {
        self.to_compact((self.init, S::zero()))
    }
    fn dig_word(&self, sum: Self::Sum, word: u64) -> Self::Sum {
        let (mut s, mut c) = self.from_compact(sum);
        let modword = S::mod_from(word, &self.module);
        s = S::add_mod(s, &modword, &self.module);
        c = S::add_mod(c, &s, &self.module);
        self.to_compact((s, c))
    }
    fn finalize(&self, sum: Self::Sum) -> Self::Sum {
        self.add(sum, &self.addout)
    }

    fn to_bytes(&self, s: Self::Sum) -> Vec<u8> {
        self.wordspec.output_to_bytes(s, 2 * self.hwidth)
    }

    fn wordspec(&self) -> WordSpec {
        self.wordspec
    }
}

impl<S: Modnum> LinearCheck for Fletcher<S> {
    type Shift = S;
    fn init_shift(&self) -> Self::Shift {
        S::zero()
    }
    fn inc_shift(&self, shift: Self::Shift) -> Self::Shift {
        S::add_mod(shift, &S::one(), &self.module)
    }
    fn shift(&self, sum: Self::Sum, shift: &Self::Shift) -> Self::Sum {
        let (s, mut c) = self.from_compact(sum);
        let shift_diff = S::mul_mod(s, shift, &self.module);
        c = S::add_mod(c, &shift_diff, &self.module);
        self.to_compact((s, c))
    }
    fn add(&self, sum_a: Self::Sum, sum_b: &Self::Sum) -> Self::Sum {
        let (sa, ca) = self.from_compact(sum_a);
        let (sb, cb) = self.from_compact(*sum_b);
        let sum_s = sa.add_mod(&sb, &self.module);
        let sum_c = ca.add_mod(&cb, &self.module);
        self.to_compact((sum_s, sum_c))
    }
    fn negate(&self, sum: Self::Sum) -> Self::Sum {
        let (s, c) = self.from_compact(sum);
        self.to_compact((s.neg_mod(&self.module), c.neg_mod(&self.module)))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::tests::{check_example, test_find, test_prop, test_shifts};
    use std::str::FromStr;
    #[test]
    fn adler32() {
        let adel = Fletcher::<u16>::with_options()
            .width(32)
            .init(1)
            .module(65521)
            .check(0x091e01de)
            .build()
            .unwrap();
        test_shifts(&adel);
        test_find(&adel);
        test_prop(&adel);
        check_example(&adel, 0x81bfd25f);
        let nobel = Fletcher::with_options()
            .width(32)
            .init(1u32)
            .module(65521)
            .check(0x091e01de)
            .build()
            .unwrap();
        test_shifts(&nobel);
        test_find(&nobel);
        test_prop(&adel);
        check_example(&nobel, 0x81bfd25f);
    }
    #[test]
    fn fletcher16() {
        let f16 = Fletcher::with_options()
            .width(16)
            .module(0xffu8)
            .check(0x1ede)
            .build()
            .unwrap();
        test_shifts(&f16);
        test_find(&f16);
        test_prop(&f16);
        check_example(&f16, 0x7815);
    }
    #[test]
    fn fletcher8() {
        let f8 = Fletcher::<u8>::from_str("width=8 module=f init=0 addout=0 swap=false check=0xc")
            .unwrap();
        test_shifts(&f8);
        test_prop(&f8);
        check_example(&f8, 0x6);
    }
}

use crate::bitnum::BitNum;
use crate::checksum::{CheckBuilderErr, Digest, LinearCheck};
use crate::keyval::KeyValIter;
use std::fmt::Display;
use std::str::FromStr;

#[derive(Clone)]
/// A builder for a Fletcher-like algorithm.
///
/// The basic functionality of this algorithm is:
/// * there is a sum which is just the bytes summed modulo some number
/// * there is also a second sum which the sum of all of the normal sums (modulo the same number)
///
/// Note that text word sizes are currently only `u8`.
///
/// It works roughly like this:
/// ```
/// # fn check(file: &[u8]) -> u32 {
/// # let module = 0xfff1u32;
/// # let (init1, init2) = (1, 0);
/// # let (addout1, addout2) = (0, 0);
/// # let hwidth = 16;
/// let mut sum1 = init1;
/// let mut sum2 = init2;
/// for byte in file {
///     sum1 = (sum1 + *byte as u32) % module;
///     sum2 = (sum2 + sum1) % module;
/// }
/// return (sum2 + addout2) % module << hwidth | (sum1 + addout1) % module;
/// # }
/// ```
/// Normally, the sum is represented as the cumulative sum bitshifted to be above the regular sum.
/// This representation will be referred to as "compact".
///
/// These are the parameters:
/// * width: Total number of bits of the checksum (twice the amount of bits of the individual sums)
/// * module: The number by which both sums get reduced
/// * init: The initial value of the sum, compact
/// * addout: The value that gets added at the end, compact
/// * swap: Whether to swap the values in the compact representation, i.e. put the regular sum above the cumulative sum
/// * check: The checksum of the bytes "123456789", checked to be correct on build
/// * name: The name to be used when displaying the algorithm (optional)
pub struct FletcherBuilder<Sum: BitNum> {
    width: Option<usize>,
    module: Sum,
    init: Sum,
    addout: Sum,
    swap: bool,
    check: Option<Sum>,
    name: Option<String>,
}

impl<S: BitNum> FletcherBuilder<S> {
    /// Sets the width of the type (both sums included, must be even, mandatory)
    pub fn width(&mut self, w: usize) -> &mut Self {
        self.width = Some(w);
        self
    }
    /// Sets the module of both sums (mandatory)
    pub fn module(&mut self, m: S) -> &mut Self {
        self.module = m;
        self
    }
    /// Sets the initial value
    ///
    /// Contains separate values for both sums, the cumulative one is bitshifted
    pub fn init(&mut self, i: S) -> &mut Self {
        self.init = i;
        self
    }
    /// Sets a value that gets added after the checksum is finished
    ///
    /// Contains separate values for both sums, the cumulative one is bitshifted
    pub fn addout(&mut self, o: S) -> &mut Self {
        self.addout = o;
        self
    }
    /// Normally, the cumulative sum is saved on the higher bits and the normal sum in the lower bits.
    /// Setting this option to true swaps the positions.
    pub fn swap(&mut self, s: bool) -> &mut Self {
        self.swap = s;
        self
    }
    /// Checks whether c is the same as the checksum of "123456789" on creation
    pub fn check(&mut self, c: S) -> &mut Self {
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
        // note: we only store the half width because it is more useful to us
        let hwidth = match self.width {
            None => return Err(CheckBuilderErr::MissingParameter("width")),
            Some(w) => {
                if w % 2 != 0 || w > self.init.bits() {
                    return Err(CheckBuilderErr::ValueOutOfRange("width"));
                } else {
                    w / 2
                }
            }
        };

        let mask = (S::one() << hwidth) - S::one();
        let module = if self.module == S::zero() {
            S::one() << hwidth
        } else {
            self.module
        };
        let mut fletch = Fletcher {
            hwidth,
            module,
            init: self.init,
            addout: self.addout,
            swap: self.swap,
            mask,
            name: self.name.clone(),
        };
        let (mut s, mut c) = fletch.from_compact(self.init);
        s = s % module;
        c = c % module;
        fletch.init = fletch.to_compact((s, c));
        let (mut s, mut c) = fletch.from_compact(self.addout);
        s = s % module;
        c = c % module;
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

pub struct Fletcher<Sum: BitNum> {
    hwidth: usize,
    module: Sum,
    init: Sum,
    addout: Sum,
    swap: bool,
    mask: Sum,
    name: Option<String>,
}

impl<Sum: BitNum> Display for Fletcher<Sum> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.name {
            Some(n) => write!(f, "{}", n),
            None => write!(
                f,
                "fletcher width={} module={:#x} init={:#x} addout={:#x} swap={}",
                2 * self.hwidth,
                self.module,
                self.init,
                self.addout,
                self.swap
            ),
        }
    }
}

impl<Sum: BitNum> Fletcher<Sum> {
    /// Creates a `FletcherBuilder`, see `FletcherBuilder` documentation for more details.
    pub fn with_options() -> FletcherBuilder<Sum> {
        FletcherBuilder {
            width: None,
            module: Sum::zero(),
            init: Sum::zero(),
            addout: Sum::zero(),
            swap: false,
            check: None,
            name: None,
        }
    }
    fn from_compact(&self, x: Sum) -> (Sum, Sum) {
        let l = x & self.mask;
        let h = (x >> self.hwidth) & self.mask;
        if self.swap {
            (h, l)
        } else {
            (l, h)
        }
    }
    fn to_compact(&self, (s, c): (Sum, Sum)) -> Sum {
        let (l, h) = if self.swap { (c, s) } else { (s, c) };
        (l & self.mask) ^ (h & self.mask) << self.hwidth
    }
}

impl<Sum: BitNum> FromStr for FletcherBuilder<Sum> {
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
                "addout" => Sum::from_hex(&current_val).ok().map(|x| fletch.addout(x)),
                "swap" => bool::from_str(&current_val).ok().map(|x| fletch.swap(x)),
                "check" => Sum::from_hex(&current_val).ok().map(|x| fletch.check(x)),
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

impl<Sum: BitNum> FromStr for Fletcher<Sum> {
    /// Construct a new fletcher sum algotithm
    ///
    /// Example:
    ///
    /// ```
    /// # use libdelsum::checksum::fletcher::Fletcher;
    /// # use std::str::FromStr;
    /// Fletcher::<u32>::from_str("width=32 init=1 module=0xfff1 name=\"adler-32\"").is_ok();
    /// ```
    fn from_str(s: &str) -> Result<Fletcher<Sum>, CheckBuilderErr> {
        FletcherBuilder::<Sum>::from_str(s)?.build()
    }
    type Err = CheckBuilderErr;
}

impl<S: BitNum> Digest for Fletcher<S> {
    type Sum = S;
    fn init(&self) -> Self::Sum {
        self.init
    }
    fn dig_byte(&self, sum: Self::Sum, byte: u8) -> Self::Sum {
        let (mut s, mut c) = self.from_compact(sum);
        s = (s + S::from(byte) % self.module) % self.module;
        c = (c + s) % self.module;
        self.to_compact((s, c))
    }
    fn finalize(&self, sum: Self::Sum) -> Self::Sum {
        self.add(sum, &self.addout)
    }
}

impl<S: BitNum> LinearCheck for Fletcher<S> {
    type Shift = S;
    fn init_shift(&self) -> Self::Shift {
        S::zero()
    }
    fn inc_shift(&self, shift: Self::Shift) -> Self::Shift {
        (shift + S::one()) % self.module
    }
    fn shift(&self, sum: Self::Sum, shift: &Self::Shift) -> Self::Sum {
        let (s, mut c) = self.from_compact(sum);
        c = (c + s * *shift) % self.module;
        self.to_compact((s, c))
    }
    fn add(&self, sum_a: Self::Sum, sum_b: &Self::Sum) -> Self::Sum {
        let (sa, ca) = self.from_compact(sum_a);
        let (sb, cb) = self.from_compact(*sum_b);
        self.to_compact(((sa + sb) % self.module, (ca + cb) % self.module))
    }
    fn negate(&self, sum: Self::Sum) -> Self::Sum {
        let (mut s, mut c) = self.from_compact(sum);
        if s != S::zero() {
            s = self.module - s;
        }
        if c != S::zero() {
            c = self.module - c;
        }
        self.to_compact((s, c))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::tests::{check_example, test_find, test_prop, test_shifts};
    use std::str::FromStr;
    #[test]
    fn adler32() {
        let adel = Fletcher::<u32>::with_options()
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
            .init(1u64)
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
            .module(0xffu16)
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
        let f8 = Fletcher::<u8>::from_str("width=8 module=15 init=0 addout=0 swap=false check=0xc")
            .unwrap();
        test_shifts(&f8);
        test_prop(&f8);
        check_example(&f8, 0x6);
    }
}

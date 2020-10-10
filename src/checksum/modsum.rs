use crate::bitnum::Modnum;
use crate::checksum::{CheckBuilderErr, Digest, LinearCheck};
use crate::keyval::KeyValIter;
use std::fmt::Display;
use std::str::FromStr;

/// The builder for doing a simple modular sum over bytes (i.e. `bytes.sum() % module`)
///
/// There are a number of parameters:
/// * width: The number of bits in the sum type, at most 64
/// * module: The sum is taken modulo this number
/// * init: The initial number
/// * check: The checksum of "123456789" (optional, gets checked at construction)
/// * name: An optional name that gets used for display purposes
///
/// Note that a parameter to add at the end is not needed, since it is equivalent to `init`.
#[derive(Clone)]
pub struct ModSumBuilder<S: Modnum> {
    width: Option<usize>,
    module: S,
    init: S,
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
        self.module = m;
        self
    }
    /// The initial value, optional, defaults to 0.
    pub fn init(&mut self, i: S) -> &mut Self {
        self.init = i;
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
            .ok_or_else(|| CheckBuilderErr::MissingParameter("width"))?;
        let module = if self.module == S::zero() && width < self.module.bits() {
            S::one() << width
        } else {
            self.module
        };
        let init = if module != S::zero() {
            self.init % module
        } else {
            self.init
        };
        let s = ModSum {
            width,
            module,
            init,
            name: self.name.clone(),
        };
        match self.check {
            Some(c) => {
                // tbh this check is rather useless,
                // but so is this whole type of checksum
                if s.digest(&b"123456789"[..]).unwrap() == c {
                    Ok(s)
                } else {
                    Err(CheckBuilderErr::CheckFail)
                }
            }
            None => Ok(s),
        }
    }
}

pub struct ModSum<S: Modnum> {
    width: usize,
    module: S,
    init: S,
    name: Option<String>,
}

impl<S: Modnum> ModSum<S> {
    /// Creates a `ModSumBuilder`, for more information see its documentation.
    pub fn with_options() -> ModSumBuilder<S> {
        ModSumBuilder {
            width: None,
            module: S::zero(),
            init: S::zero(),
            check: None,
            name: None,
        }
    }
}

impl<Sum: Modnum> Display for ModSum<Sum> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.name {
            Some(n) => write!(f, "{}", n),
            None => write!(
                f,
                "modsum width={} module={:#x} init={:#x}",
                self.width, self.module, self.init
            ),
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
    fn dig_byte(&self, sum: Self::Sum, byte: u8) -> Self::Sum {
        sum.add_mod(&S::from(byte), &self.module)
    }
    fn finalize(&self, sum: Self::Sum) -> Self::Sum {
        sum
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
    fn screw() {
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
    fn this() {
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
    fn checksum_type() {
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
        let x = Vec::from("implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR");
        let y = Vec::from("This program comes with ABSOLUTELY NO WARRANTY; for details typ");
        let merchantibility = chk.digest("MERCHANTABILITY".as_bytes()).unwrap();
        let ith_absolutely_ = chk.digest("ith ABSOLUTELY ".as_bytes()).unwrap();
        assert_eq!(
            chk.find_segments(
                &[x, y],
                &[merchantibility, ith_absolutely_],
                Relativity::Start
            ),
            vec![(vec![20], vec![RelativeIndex::FromStart(35)])]
        );
    }
}

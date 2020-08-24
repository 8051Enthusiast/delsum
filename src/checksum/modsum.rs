use crate::bitnum::Modnum;
use crate::checksum::{CheckBuilderErr, Digest, LinearCheck};
use crate::keyval::KeyValIter;
use std::fmt::Display;
use std::str::FromStr;

#[derive(Clone)]
pub struct ModSumBuilder<S: Modnum> {
    width: Option<usize>,
    modulo: S,
    init: S,
    check: Option<S>,
    name: Option<String>,
}

impl<S: Modnum> ModSumBuilder<S> {
    pub fn width(&mut self, w: usize) -> &mut Self {
        self.width = Some(w);
        self
    }
    pub fn modulo(&mut self, m: S) -> &mut Self {
        self.modulo = m;
        self
    }
    pub fn init(&mut self, i: S) -> &mut Self {
        self.init = i;
        self
    }
    pub fn check(&mut self, c: S) -> &mut Self {
        self.check = Some(c);
        self
    }
    pub fn name(&mut self, n: &str) -> &mut Self {
        self.name = Some(String::from(n));
        self
    }
    pub fn build(&self) -> Result<ModSum<S>, CheckBuilderErr> {
        let width = self
            .width
            .ok_or_else(|| CheckBuilderErr::MissingParameter("width"))?;
        let modulo = if self.modulo == S::zero() && width < self.modulo.bits() {
            S::one() << width
        } else {
            self.modulo
        };
        let init = if modulo != S::zero() {
            self.init % modulo
        } else {
            self.init
        };
        let s = ModSum {
            width,
            modulo,
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
    modulo: S,
    init: S,
    name: Option<String>,
}

impl<S: Modnum> ModSum<S> {
    pub fn with_options() -> ModSumBuilder<S> {
        ModSumBuilder {
            width: None,
            modulo: S::zero(),
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
                "<modsum width={} modulo={:#x} init={:#x}>",
                self.width, self.modulo, self.init
            ),
        }
    }
}

impl<Sum: Modnum> FromStr for ModSum<Sum> {
    /// Construct a new modular sum from a string specification.
    ///
    /// Example:
    ///
    /// width=16 modulo=65535 init=0
    fn from_str(s: &str) -> Result<ModSum<Sum>, CheckBuilderErr> {
        let mut sum = Self::with_options();
        for x in KeyValIter::new(s) {
            let (current_key, current_val) = match x {
                Err(key) => return Err(CheckBuilderErr::MalformedString(key)),
                Ok(s) => s,
            };
            let crc_op = match current_key.as_str() {
                "width" => usize::from_str(&current_val).ok().map(|x| sum.width(x)),
                "modulo" => Sum::from_dec_or_hex(&current_val)
                    .ok()
                    .map(|x| sum.modulo(x)),
                "init" => Sum::from_dec_or_hex(&current_val).ok().map(|x| sum.init(x)),
                "name" => Some(sum.name(&current_val)),
                _ => return Err(CheckBuilderErr::UnknownKey(current_key)),
            };
            match crc_op {
                Some(c) => sum = c.clone(),
                None => return Err(CheckBuilderErr::MalformedString(current_key)),
            }
        }
        sum.build()
    }
    type Err = CheckBuilderErr;
}

impl<S: Modnum> Digest for ModSum<S> {
    type Sum = S;
    fn init(&self) -> Self::Sum {
        self.init
    }
    fn dig_byte(&self, sum: Self::Sum, byte: u8) -> Self::Sum {
        sum.add_mod(&S::from(byte), &self.modulo)
    }
    fn finalize(&self, sum: Self::Sum) -> Self::Sum {
        sum
    }
}

impl<S: Modnum> LinearCheck for ModSum<S> {
    type Shift = ();
    fn init_shift(&self) -> Self::Shift {}
    fn inc_shift(&self, _: Self::Shift) -> Self::Shift {}
    fn shift(&self, sum: Self::Sum, _: &Self::Shift) -> Self::Sum {
        sum
    }
    fn shift_n(&self, _: usize) -> Self::Shift {}
    fn add(&self, sum_a: Self::Sum, sum_b: &Self::Sum) -> Self::Sum {
        sum_a.add_mod(sum_b, &self.modulo)
    }
    fn negate(&self, sum: Self::Sum) -> Self::Sum {
        if sum == S::zero() {
            sum
        } else if self.modulo == S::zero() {
            !sum + S::one()
        } else {
            self.modulo - sum
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::{RelativeIndex, Relativity};
    #[test]
    fn screw() {
        ModSum::<u8>::with_options()
            .width(8)
            .check(0xdd)
            .build()
            .unwrap();

        ModSum::<u16>::with_options()
            .width(8)
            .check(0xdd)
            .build()
            .unwrap();
    }
    #[test]
    fn this() {
        ModSum::<u8>::with_options()
            .width(5)
            .modulo(17)
            .check(1)
            .build()
            .unwrap();
    }
    #[test]
    fn checksum_type() {
        let chk = ModSum::<u16>::with_options()
            .width(16)
            .init(0xff00)
            .modulo(0xffff)
            .check(0xde)
            .build()
            .unwrap();
        // 0xff*0x101 = 0xffff
        let many_255: Vec<_> = std::iter::repeat(0xffu8).take(0x101).collect();
        assert_eq!(chk.digest(many_255.as_slice()).unwrap(), 0xff00);
        let x = Vec::from("implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR");
        let y = Vec::from("This program comes with ABSOLUTELY NO WARRANTY; for details typ");
        let merchantibility = chk.digest("MERCHANTABILITY".as_bytes()).unwrap();
        let ith_absolutely_ = chk.digest("ith ABSOLUTELY ".as_bytes()).unwrap();
        assert_eq!(
            chk.find_checksum_segments(
                &[x, y],
                &[merchantibility, ith_absolutely_],
                Relativity::Start
            ),
            vec![(vec![20], vec![RelativeIndex::FromStart(35)])]
        );
    }
}

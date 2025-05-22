use std::{fmt::Display, str::FromStr};
mod rev;
pub use rev::reverse_polyhash;

use crate::{
    bitnum::Modnum,
    checksum::{parse_hex, CheckBuilderErr, Checksum, Digest, LinearCheck},
    endian::{Endian, SignedInt, Signedness, WordSpec},
    keyval::KeyValIter,
};

#[derive(Debug, Clone)]
pub struct PolyHashBuilder<S> {
    width: Option<usize>,
    factor: Option<S>,
    init: Option<S>,
    addout: Option<S>,
    input_endian: Option<Endian>,
    output_endian: Option<Endian>,
    wordsize: Option<usize>,
    signedness: Option<Signedness>,
    check: Option<S>,
    name: Option<String>,
}

impl<S: Modnum> FromStr for PolyHashBuilder<S> {
    fn from_str(s: &str) -> Result<PolyHashBuilder<S>, CheckBuilderErr> {
        let mut sum = PolyHash::<S>::with_options();
        for x in KeyValIter::new(s) {
            let (current_key, current_val) = match x {
                Err(key) => return Err(CheckBuilderErr::MalformedString(key)),
                Ok(s) => s,
            };
            let sum_op = match current_key.as_str() {
                "width" => usize::from_str(&current_val).ok().map(|x| sum.width(x)),
                "factor" => Some(sum.factor(parse_hex::<S>(&current_val, "factor")?)),
                "init" => Some(sum.init(parse_hex::<S>(&current_val, "init")?)),
                "addout" => Some(sum.addout(parse_hex::<S>(&current_val, "addout")?)),
                "in_endian" => Endian::from_str(&current_val).ok().map(|x| sum.inendian(x)),
                "wordsize" => usize::from_str(&current_val).ok().map(|x| sum.wordsize(x)),
                "out_endian" => Endian::from_str(&current_val)
                    .ok()
                    .map(|x| sum.outendian(x)),
                "signedness" => Signedness::from_str(&current_val)
                    .ok()
                    .map(|x| sum.signedness(x)),
                "name" => Some(sum.name(&current_val)),
                _ => return Err(CheckBuilderErr::UnknownKey(current_key)),
            };
            match sum_op {
                Some(c) => sum = c.clone(),
                None => return Err(CheckBuilderErr::MalformedString(current_key)),
            }
        }
        Ok(sum)
    }
    type Err = CheckBuilderErr;
}

impl<S: Modnum> PolyHashBuilder<S> {
    /// The total width, in bits, of the sum. Mandatory.
    ///
    /// The checksum is calculated modulo 2^width.
    pub fn width(&mut self, w: usize) -> &mut Self {
        self.width = Some(w);
        self
    }
    /// The factor by which the sum gets multiplied before adding
    /// the next input byte.
    pub fn factor(&mut self, w: S) -> &mut Self {
        self.factor = Some(w);
        self
    }
    /// The initial value of the checksum, default 0.
    pub fn init(&mut self, i: S) -> &mut Self {
        self.init = Some(i);
        self
    }
    /// The value that gets added to the checksum at the out.
    pub fn addout(&mut self, i: S) -> &mut Self {
        self.addout = Some(i);
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
    fn signedness(&mut self, s: Signedness) -> &mut Self {
        self.signedness = Some(s);
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
    pub fn build(&self) -> Result<PolyHash<S>, CheckBuilderErr> {
        let width = self
            .width
            .ok_or(CheckBuilderErr::MissingParameter("width"))?;
        if width > 64 {
            return Err(CheckBuilderErr::ValueOutOfRange("width"));
        }
        let factor = self
            .factor
            .ok_or(CheckBuilderErr::MissingParameter("factor"))?;
        if factor & S::one() == S::zero() || factor == S::one() {
            return Err(CheckBuilderErr::ValueOutOfRange("factor"));
        };
        let init = mask(width, self.init.unwrap_or_else(S::zero));
        let addout = mask(width, self.addout.unwrap_or_else(S::zero));
        let wordsize = self.wordsize.unwrap_or(8);
        if wordsize == 0 || wordsize % 8 != 0 || wordsize > 64 {
            return Err(CheckBuilderErr::ValueOutOfRange("wordsize"));
        }
        let wordspec = WordSpec {
            input_endian: self.input_endian.unwrap_or(Endian::Big),
            wordsize,
            output_endian: self.output_endian.unwrap_or(Endian::Big),
            signedness: self.signedness.unwrap_or(Signedness::Unsigned),
        };
        let s = PolyHash {
            width,
            factor,
            init,
            addout,
            wordspec,
            name: self.name.clone(),
        };
        match self.check {
            Some(c) => {
                let mut sum = s.init();
                for &x in b"123456789" {
                    sum = s.dig_word(sum, SignedInt::pos(x as u64));
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

#[derive(Debug, PartialEq, Eq)]
pub struct PolyHash<S> {
    width: usize,
    factor: S,
    init: S,
    addout: S,
    wordspec: WordSpec,
    name: Option<String>,
}

fn mask<S: Modnum>(width: usize, word: S) -> S {
    if word.bits() > width {
        word & ((S::one() << width) - S::one())
    } else {
        word
    }
}

impl<S: Modnum> PolyHash<S> {
    fn mask(&self, word: S) -> S {
        mask(self.width, word)
    }

    pub fn with_options() -> PolyHashBuilder<S> {
        PolyHashBuilder {
            width: None,
            factor: None,
            init: None,
            addout: None,
            input_endian: None,
            output_endian: None,
            signedness: None,
            wordsize: None,
            check: None,
            name: None,
        }
    }
}

impl<S: Modnum> Display for PolyHash<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.name {
            Some(n) => write!(f, "{}", n),
            None => {
                write!(
                    f,
                    "polyhash width={} factor={:#x} init={:#x} addout={:#x} signedness={}",
                    self.width, self.factor, self.init, self.addout, self.wordspec.signedness
                )?;
                if self.wordspec.word_bytes() != 1 {
                    write!(
                        f,
                        " in_endian={} wordsize={}",
                        self.wordspec.input_endian, self.wordspec.wordsize,
                    )?
                };
                if self.width > 8 {
                    write!(f, " out_endian={}", self.wordspec.output_endian)?;
                }
                Ok(())
            }
        }
    }
}

impl<S: Modnum> FromStr for PolyHash<S> {
    type Err = CheckBuilderErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        PolyHashBuilder::from_str(s)?.build()
    }
}

impl<S: Modnum> Digest for PolyHash<S> {
    type Sum = S;

    fn init(&self) -> Self::Sum {
        self.init
    }

    fn dig_word(&self, sum: Self::Sum, word: SignedInt<u64>) -> Self::Sum {
        let mut value = S::mod_from(word.value, &S::zero());
        if word.negative {
            value = value.wrapping_neg();
        }
        self.mask(sum.wrapping_mul(&self.factor).wrapping_add(&value))
    }

    fn finalize(&self, sum: Self::Sum) -> Self::Sum {
        self.mask(sum.wrapping_add(&self.addout))
    }

    fn to_bytes(&self, s: Self::Sum) -> Vec<u8> {
        self.wordspec.output_to_bytes(s, self.width)
    }

    fn checksum_from_bytes(&self, bytes: &[u8]) -> Option<Self::Sum> {
        Checksum::from_bytes(bytes, self.wordspec.output_endian, self.width)
    }

    fn wordspec(&self) -> WordSpec {
        self.wordspec
    }
}

impl<S: Modnum> LinearCheck for PolyHash<S> {
    type Shift = S;

    fn init_shift(&self) -> Self::Shift {
        S::one()
    }

    fn inc_shift(&self, shift: Self::Shift) -> Self::Shift {
        self.factor.wrapping_mul(&shift)
    }

    fn shift(&self, sum: Self::Sum, shift: &Self::Shift) -> Self::Sum {
        self.mask(sum.wrapping_mul(shift))
    }

    fn add(&self, sum_a: Self::Sum, sum_b: &Self::Sum) -> Self::Sum {
        self.mask(sum_a.wrapping_add(sum_b))
    }

    fn negate(&self, sum: Self::Sum) -> Self::Sum {
        self.mask(S::zero().wrapping_sub(&sum))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::tests::*;

    #[test]
    fn sdbm() {
        let sdbm = PolyHash::<u32>::with_options()
            .width(32)
            .factor(0x1003f)
            .name("sdbm")
            .build()
            .unwrap();
        test_shifts(&sdbm);
        test_find(&sdbm);
        test_prop(&sdbm);
        check_example(&sdbm, 0x694c5cd2);
    }

    #[test]
    fn djb2() {
        let djb2 = PolyHash::<u32>::with_options()
            .width(32)
            .factor(33)
            .init(5381)
            .name("djb2")
            .build()
            .unwrap();
        test_shifts(&djb2);
        test_find(&djb2);
        test_prop(&djb2);
        check_example(&djb2, 0x84a046e5);
    }

    #[test]
    fn masking() {
        let polyhash = PolyHash::<u32>::with_options()
            .width(7)
            .factor(0x2f)
            .build()
            .unwrap();
        assert_eq!(
            polyhash
                .digest(&[0, 18, 232, 236, 87, 255, 203, 100])
                .unwrap(),
            0x77
        );
    }
}

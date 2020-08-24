use crate::bitnum::BitNum;
use crate::checksum::{CheckBuilderErr, Digest, LinearCheck};
use crate::keyval::KeyValIter;
use std::fmt::Display;
use std::str::FromStr;

#[derive(Clone)]
pub struct FletcherBuilder<Sum: BitNum> {
    width: Option<usize>,
    modulo: Sum,
    init: Sum,
    addout: Sum,
    swap: bool,
    check: Option<Sum>,
    name: Option<String>,
}

impl<S: BitNum> FletcherBuilder<S> {
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
    pub fn addout(&mut self, o: S) -> &mut Self {
        self.addout = o;
        self
    }
    pub fn swap(&mut self, s: bool) -> &mut Self {
        self.swap = s;
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
    pub fn build(&self) -> Result<Fletcher<S>, CheckBuilderErr> {
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
        let modulo = if self.modulo == S::zero() {
            S::one() << hwidth
        } else {
            self.modulo
        };
        let mut fletch = Fletcher {
            hwidth,
            modulo,
            init: self.init,
            addout: self.addout,
            swap: self.swap,
            mask,
            name: self.name.clone(),
        };
        let (mut s, mut c) = fletch.from_compact(self.init);
        s = s % modulo;
        c = c % modulo;
        fletch.init = fletch.to_compact((s, c));
        let (mut s, mut c) = fletch.from_compact(self.addout);
        s = s % modulo;
        c = c % modulo;
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
    modulo: Sum,
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
                "<fletcher width={} modulo={:#x} init={:#x} addout={:#x} swap={}>",
                2 * self.hwidth,
                self.modulo,
                self.init,
                self.addout,
                self.swap
            ),
        }
    }
}

impl<Sum: BitNum> Fletcher<Sum> {
    pub fn with_options() -> FletcherBuilder<Sum> {
        FletcherBuilder {
            width: None,
            modulo: Sum::zero(),
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

impl<Sum: BitNum> FromStr for Fletcher<Sum> {
    /// Construct a new fletcher sum
    ///
    /// Example:
    ///
    /// width=16 poly=0x8005 init=0x0000 refin=true refout=true xorout=0x0000 check=0xbb3d residue=0x0000 name="CRC-16/ARC"
    ///
    /// Note: the `residue` parameter is currently ignored.
    fn from_str(s: &str) -> Result<Fletcher<Sum>, CheckBuilderErr> {
        let mut fletch = Self::with_options();
        for x in KeyValIter::new(s) {
            let (current_key, current_val) = match x {
                Err(key) => return Err(CheckBuilderErr::MalformedString(key)),
                Ok(s) => s,
            };
            let fletch_op = match current_key.as_str() {
                "width" => usize::from_str(&current_val).ok().map(|x| fletch.width(x)),
                "modulo" => Sum::from_dec_or_hex(&current_val)
                    .ok()
                    .map(|x| fletch.modulo(x)),
                "init" => Sum::from_dec_or_hex(&current_val)
                    .ok()
                    .map(|x| fletch.init(x)),
                "addout" => Sum::from_dec_or_hex(&current_val)
                    .ok()
                    .map(|x| fletch.addout(x)),
                "swap" => bool::from_str(&current_val).ok().map(|x| fletch.swap(x)),
                "check" => Sum::from_dec_or_hex(&current_val)
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
        fletch.build()
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
        s = (s + S::from(byte) % self.modulo) % self.modulo;
        c = (c + s) % self.modulo;
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
        (shift + S::one()) % self.modulo
    }
    fn shift(&self, sum: Self::Sum, shift: &Self::Shift) -> Self::Sum {
        let (s, mut c) = self.from_compact(sum);
        c = (c + s * *shift) % self.modulo;
        self.to_compact((s, c))
    }
    fn add(&self, sum_a: Self::Sum, sum_b: &Self::Sum) -> Self::Sum {
        let (sa, ca) = self.from_compact(sum_a);
        let (sb, cb) = self.from_compact(*sum_b);
        self.to_compact(((sa + sb) % self.modulo, (ca + cb) % self.modulo))
    }
    fn negate(&self, sum: Self::Sum) -> Self::Sum {
        let (mut s, mut c) = self.from_compact(sum);
        if s != S::zero() {
            s = self.modulo - s;
        }
        if c != S::zero() {
            c = self.modulo - c;
        }
        self.to_compact((s, c))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::tests::{check_example, test_find, test_shifts, test_prop};
    use std::str::FromStr;
    #[test]
    fn adler32() {
        let adel = Fletcher::<u32>::with_options()
            .width(32)
            .init(1)
            .modulo(65521)
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
            .modulo(65521)
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
            .modulo(0xffu16)
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
        let f8 = Fletcher::<u8>::from_str("width=8 modulo=15 init=0 addout=0 swap=false check=0xc")
            .unwrap();
        test_shifts(&f8);
        test_prop(&f8);
        check_example(&f8, 0x6);
    }
}

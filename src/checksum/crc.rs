use super::{CheckBuilderErr, Digest, LinearCheck};
use crate::bitnum::BitNum;
use crate::keyval::KeyValIter;
use std::fmt::Display;
use std::str::FromStr;
/// A builder for a CRC algorithm.
///
/// The Sum type is one of u8, u16, u32, u64 or u128 and must be able to hold `width` bits.
///
/// The parameters are the same as of the Rocksoft^TM Model CRC Algorithm:
/// * `width`: the width in bits of the sum values
/// * `poly`: the generator polynomial (without highest bit)
/// * `init`: the initial value for the checksum
/// * `refin`: whether to reflect the input bytes
/// * `refout`: whether to reflect the sum
/// * `xorout`: what to XOR the output with
/// * `check`: the checksum of the ASCII string "123456789" (is checked on `build()`, optional)
/// * `name`: an optional name for the algorithm
///
/// For more information on the parameters (and CRCs in general), see "A PAINLESS GUIDE CRC ERROR DETECTION ALGORITHMS"
/// or https://reveng.sourceforge.io/crc-catalogue/legend.htm (which is also a source of parameters for various common algorithms)
#[derive(Clone)]
pub struct CRCBuilder<Sum: BitNum> {
    width: Option<usize>,
    poly: Option<Sum>,
    init: Sum,
    xorout: Sum,
    refin: bool,
    refout: bool,
    check: Option<Sum>,
    name: Option<String>,
}

impl<Sum: BitNum> CRCBuilder<Sum> {
    /// Sets the poly, mandatory
    pub fn poly(&mut self, p: Sum) -> &mut Self {
        self.poly = Some(p);
        self
    }
    /// Sets the width, mandatory
    pub fn width(&mut self, w: usize) -> &mut Self {
        self.width = Some(w);
        self
    }
    /// Sets the `init` parameter, default is 0.
    pub fn init(&mut self, s: Sum) -> &mut Self {
        self.init = s;
        self
    }
    /// Sets the `xorout` parameter, default is 0.
    pub fn xorout(&mut self, s: Sum) -> &mut Self {
        self.xorout = s;
        self
    }
    /// Sets the `refin` parameter, default is false.
    pub fn refin(&mut self, i: bool) -> &mut Self {
        self.refin = i;
        self
    }
    /// Sets the `refout` parameter, default is false.
    pub fn refout(&mut self, o: bool) -> &mut Self {
        self.refout = o;
        self
    }
    /// Sets the `check` parameter, no check is done if this is left out.
    pub fn check(&mut self, c: Sum) -> &mut Self {
        self.check = Some(c);
        self
    }
    /// Sets the name, no name is set otherwise.
    pub fn name(&mut self, s: &str) -> &mut Self {
        self.name = Some(s.to_owned());
        self
    }
    /// Build the object for the algorithm, generating the lookup table and verifying that
    /// the parameters are valid.
    pub fn build(&self) -> Result<CRC<Sum>, CheckBuilderErr> {
        let width = match self.width {
            None => return Err(CheckBuilderErr::MissingParameter("width")),
            Some(w) => w,
        };
        let mask = Sum::one() << (width - 1);
        let mask = mask ^ (mask - Sum::one());
        let poly = match self.poly {
            None => return Err(CheckBuilderErr::MissingParameter("poly")),
            Some(p) => p,
        };
        // the type needs at least 8 bit so that we can comfortably add bytes to it
        // (i guess it is kind of already impliead by the from<u8> trait)
        if poly.bits() < width || poly.bits() < 8 {
            return Err(CheckBuilderErr::ValueOutOfRange("width"));
        }
        // the generator polynomial needs to have the lowest bit set in order to be useful
        if poly & Sum::one() != Sum::one() {
            return Err(CheckBuilderErr::ValueOutOfRange("poly"));
        }
        if poly & !mask != Sum::zero() {
            return Err(CheckBuilderErr::ValueOutOfRange("poly"));
        }
        if self.init & !mask != Sum::zero() {
            return Err(CheckBuilderErr::ValueOutOfRange("init"));
        }
        if self.xorout & !mask != Sum::zero() {
            return Err(CheckBuilderErr::ValueOutOfRange("xorout"));
        }
        let crc = CRC {
            width,
            poly,
            init: self.init,
            xorout: self.xorout,
            refin: self.refin,
            refout: self.refout,
            mask,
            name: self.name.clone(),
            table: CRC::<Sum>::generate_crc_table(poly, width),
        };
        match self.check {
            Some(chk) => {
                if chk & !mask != Sum::zero() {
                    Err(CheckBuilderErr::ValueOutOfRange("check"))
                } else if crc.digest(&b"123456789"[..]).unwrap() != chk {
                    Err(CheckBuilderErr::CheckFail)
                } else {
                    Ok(crc)
                }
            }
            None => Ok(crc),
        }
    }
}

pub struct CRC<Sum: BitNum> {
    init: Sum,
    xorout: Sum,
    refin: bool,
    refout: bool,
    poly: Sum,
    mask: Sum,
    width: usize,
    name: Option<String>,
    table: Box<[Sum; 256]>,
}

impl<Sum: BitNum> Display for CRC<Sum> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.name {
            Some(n) => write!(f, "{}", n),
            None => write!(
                f,
                "<crc width={} poly={:#x} init={:#x} xorout={:#x} refin={} refout={}>",
                self.width, self.poly, self.init, self.xorout, self.refin, self.refout
            ),
        }
    }
}

impl<Sum: BitNum> CRC<Sum> {
    /// Construct a new CRC from parameters, see also the documentation for CRCBuilder.
    pub fn with_options() -> CRCBuilder<Sum> {
        CRCBuilder {
            poly: None,
            init: Sum::zero(),
            xorout: Sum::zero(),
            width: None,
            refin: false,
            refout: false,
            check: None,
            name: Some(String::default()),
        }
    }
    /// Construct the CRC table, given the bitwidth and the generator.
    /// Just your typical crc table generator implementation, nothing to see here.
    fn generate_crc_table(generator: Sum, width: usize) -> Box<[Sum; 256]> {
        let mut table = Box::new([Sum::zero(); 256]);
        let mut crc = Sum::one() << (width - 1);
        for i in 0..8 {
            let addition = if Sum::zero() != crc & Sum::one() << (width - 1) {
                // prevent overflow
                crc = crc ^ Sum::one() << (width - 1);
                generator
            } else {
                Sum::zero()
            };
            crc = (crc << 1usize) ^ addition;
            let validrange = 1usize << i;
            for x in 0..validrange {
                table[validrange + x] = table[x] ^ crc;
            }
        }
        table
    }
    /// Get the CRC lookup table entry indexed by x.
    fn get_table_entry(&self, x: Sum) -> Sum {
        let index: u8 = match x.try_into() {
            Ok(byte) => byte,
            Err(_) => panic!("Internal error: non-byte index into CRC lookup table"),
        };
        self.table[index as usize]
    }
    /// Reflects `sum` if `refout` is set.
    fn regularize(&self, sum: Sum) -> Sum {
        if self.refout {
            sum.revbits() >> (sum.bits() - self.width)
        } else {
            sum
        }
    }
}
impl<Sum: BitNum> FromStr for CRC<Sum> {
    /// Construct a new CRC from a string specification.
    ///
    /// Example (courtesy of the crc reveng catalogue):
    ///
    /// width=16 poly=0x8005 init=0x0000 refin=true refout=true xorout=0x0000 check=0xbb3d residue=0x0000 name="CRC-16/ARC"
    ///
    /// Note: the `residue` parameter is currently ignored.
    fn from_str(s: &str) -> Result<CRC<Sum>, CheckBuilderErr> {
        let mut crc = Self::with_options();
        for x in KeyValIter::new(s) {
            let (current_key, current_val) = match x {
                Err(key) => return Err(CheckBuilderErr::MalformedString(key)),
                Ok(s) => s,
            };
            let crc_op = match current_key.as_str() {
                // I would love to return a ValueOutOfRange error here, but I don't know how
                // I would go about it
                "width" => usize::from_str(&current_val).ok().map(|x| crc.width(x)),
                "poly" => Sum::from_dec_or_hex(&current_val).ok().map(|x| crc.poly(x)),
                "init" => Sum::from_dec_or_hex(&current_val).ok().map(|x| crc.init(x)),
                "xorout" => Sum::from_dec_or_hex(&current_val)
                    .ok()
                    .map(|x| crc.xorout(x)),
                "refin" => bool::from_str(&current_val).ok().map(|x| crc.refin(x)),
                "refout" => bool::from_str(&current_val).ok().map(|x| crc.refout(x)),
                "residue" => Some(&mut crc),
                "check" => Sum::from_dec_or_hex(&current_val)
                    .ok()
                    .map(|x| crc.check(x)),
                "name" => Some(crc.name(&current_val)),
                _ => return Err(CheckBuilderErr::UnknownKey(current_key)),
            };
            match crc_op {
                Some(c) => crc = c.clone(),
                None => return Err(CheckBuilderErr::MalformedString(current_key)),
            }
        }
        crc.build()
    }
    type Err = CheckBuilderErr;
}

impl<S: BitNum> Digest for CRC<S> {
    type Sum = S;
    fn init(&self) -> Self::Sum {
        // note: if refout is set, the sum value is always left in reflected form,
        // because it is needed for the linearity conditions of LinearCheck.
        self.regularize(self.init)
    }
    fn dig_byte(&self, sum: Self::Sum, byte: u8) -> Self::Sum {
        // sum is reflected both at beginning and end to do operations on it in unreflected state
        // (this could be prevented by implementing a proper implementation for the reflected case)
        let refsum = self.regularize(sum);
        let inbyte = if self.refin {
            byte.reverse_bits()
        } else {
            byte
        };
        self.regularize(if self.width <= 8 {
            // if the width is less than 8, we have to be careful not to do negative shift values
            let overhang = (refsum << (8 - self.width)) ^ S::from(inbyte);
            self.get_table_entry(overhang)
        } else {
            // your typical CRC reduction implemented through CRC lookup table
            let overhang = refsum >> (self.width - 8) ^ S::from(inbyte);
            let l_remain = (refsum << 8) & self.mask;
            self.get_table_entry(overhang) ^ l_remain
        })
    }
    fn finalize(&self, sum: Self::Sum) -> Self::Sum {
        sum ^ self.xorout
    }
}

impl<S: BitNum> LinearCheck for CRC<S> {
    type Shift = S;
    fn shift(&self, sum: Self::Sum, shift: &Self::Shift) -> Self::Sum {
        let sum = self.regularize(sum);
        // note: this is just a carry-less multiply modulo the generator polynomial
        // we keep a lo_part and a hi_part because we don't know the type double the size
        let mut lo_part = if *shift & Self::Shift::one() != Self::Shift::zero() {
            sum
        } else {
            Self::Sum::zero()
        };
        let mut hi_part = Self::Sum::zero();
        for i in 1..self.width {
            if *shift >> i & Self::Shift::one() != Self::Shift::zero() {
                lo_part = lo_part ^ sum << i;
                hi_part = hi_part ^ sum >> (self.width - i);
            }
        }
        lo_part = lo_part & self.mask;

        // we reduce by generator polynomial bytewise through lookup table
        let mut bits_left = self.width;
        while bits_left > 0 {
            let shift_amount = bits_left.min(8);
            bits_left -= shift_amount;
            let new_part = self.get_table_entry(hi_part >> (self.width - shift_amount));
            // be careful not to overshoot
            if shift_amount >= hi_part.bits() {
                hi_part = new_part
            } else {
                hi_part = (hi_part << shift_amount) & self.mask ^ new_part;
            }
        }
        self.regularize(hi_part ^ lo_part)
    }
    fn add(&self, sum_a: Self::Sum, sum_b: &Self::Sum) -> Self::Sum {
        sum_a ^ *sum_b
    }
    fn negate(&self, sum: Self::Sum) -> Self::Sum {
        // laughs in characteristic 2
        sum
    }
    fn init_shift(&self) -> Self::Shift {
        Self::Shift::one()
    }
    fn inc_shift(&self, shift: Self::Shift) -> Self::Shift {
        // note: shifts are always unreflected
        if self.width <= 8 {
            let overhang = shift << (8 - self.width);
            self.get_table_entry(overhang)
        } else {
            let overhang = shift >> (self.width - 8);
            let l_remain = (shift << 8) & self.mask;
            self.get_table_entry(overhang) ^ l_remain
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::tests::{check_example, test_find, test_prop, test_shifts};
    #[test]
    fn cms_16() {
        assert!(CRC::<u32>::with_options()
            .poly(0x8005)
            .width(16)
            .init(0xffff)
            .check(0xaee7)
            .build()
            .is_ok());
        let crc = CRC::<u16>::with_options()
            .poly(0x8005)
            .width(16)
            .init(0xffff)
            .check(0xaee7)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0x6bd6);
    }
    #[test]
    fn gsm_3() {
        let crc = CRC::<u8>::with_options()
            .poly(0x3)
            .width(3)
            .xorout(0x7)
            .check(0x4)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_prop(&crc);
        check_example(&crc, 7);
        let crc = CRC::<u128>::with_options()
            .poly(0x3)
            .width(3)
            .xorout(0x7)
            .check(0x4)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_prop(&crc);
        check_example(&crc, 7);
    }
    #[test]
    fn rohc_7() {
        let crc = CRC::<u8>::with_options()
            .poly(0x4f)
            .width(7)
            .init(0x7f)
            .refin(true)
            .refout(true)
            .check(0x53)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_prop(&crc);
        check_example(&crc, 0x25);
        let crc = CRC::<u32>::with_options()
            .poly(0x4f)
            .width(7)
            .init(0x7f)
            .refin(true)
            .refout(true)
            .check(0x53)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_prop(&crc);
        check_example(&crc, 0x25);
    }
    #[test]
    fn usb_5() {
        let crc = CRC::<u8>::with_options()
            .poly(0x05)
            .width(5)
            .init(0x1f)
            .refin(true)
            .refout(true)
            .xorout(0x1f)
            .check(0x19)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_prop(&crc);
        check_example(&crc, 0x17);
    }
    #[test]
    fn umts_12() {
        let crc = CRC::<u16>::with_options()
            .poly(0x80f)
            .width(12)
            .refout(true)
            .check(0xdaf)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0x35a);
    }
    #[test]
    fn en13757_16() {
        let crc = CRC::<u16>::with_options()
            .poly(0x3d65)
            .width(16)
            .xorout(0xffff)
            .check(0xc2b7)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0x69e2);
    }
    #[test]
    fn mpt1327_15() {
        let crc = CRC::<u16>::with_options()
            .poly(0x6815)
            .width(15)
            .xorout(0x0001)
            .check(0x2566)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0x1993);
    }
    #[test]
    fn canfd_17() {
        let crc = CRC::<u32>::with_options()
            .poly(0x1685b)
            .width(17)
            .check(0x04f03)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0x00f396);
    }
    #[test]
    fn bzip2_32() {
        let crc = CRC::<u32>::with_options()
            .poly(0x04c11db7)
            .width(32)
            .init(0xffffffff)
            .xorout(0xffffffff)
            .check(0xfc891918)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0xe8c5033d);
    }
    #[test]
    fn iscsi_32() {
        let crc = CRC::<u32>::with_options()
            .poly(0x1edc6f41)
            .width(32)
            .init(0xffffffff)
            .xorout(0xffffffff)
            .refin(true)
            .refout(true)
            .check(0xe3069283)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0x5a513507);
    }
    #[test]
    fn gsm_40() {
        let crc = CRC::<u64>::with_options()
            .poly(0x0004820009)
            .width(40)
            .xorout(0xffffffffff)
            .check(0xd4164fc646)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0x4165335176)
    }
    #[test]
    fn xz_64() {
        let crc = CRC::<u64>::with_options()
            .poly(0x42f0e1eba9ea3693)
            .width(64)
            .init(0xffffffffffffffff)
            .refin(true)
            .refout(true)
            .xorout(0xffffffffffffffff)
            .check(0x995dc9bbdf1939fa)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0xb03d0f148fcab729);
    }
    #[test]
    fn darc_82() {
        let crc = CRC::<u128>::with_options()
            .poly(0x0308c0111011401440411)
            .width(82)
            .refin(true)
            .refout(true)
            .check(0x09ea83f625023801fd612)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0x030c57c0142280dfd62847)
    }
    #[test]
    fn parity_1() {
        let crc = CRC::<u8>::with_options()
            .poly(1)
            .width(1)
            .check(1)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_prop(&crc);
        check_example(&crc, 0);
    }
    #[test]
    fn i4321_8() {
        let crc = CRC::<u8>::with_options()
            .poly(0x7)
            .width(8)
            .xorout(0x55)
            .check(0xa1)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_prop(&crc);
        check_example(&crc, 0x96);
    }
    #[test]
    fn isoiec144433a_16() {
        let crc = CRC::<u16>::with_options()
            .poly(0x1021)
            .width(16)
            .init(0xc6c6u16)
            .refin(true)
            .refout(true)
            .check(0xbf05)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0x6b68);
    }
    #[test]
    fn arc_16() {
        let crc = CRC::<u16>::from_str("width=16 poly=0x8005 init=0x0000 refin=true refout=true xorout=0x0000 check=0xbb3d residue=0x0000 name=\"CRC-16/ARC\"")
            .unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0xf15e);
    }
    #[test]
    fn something_16() {
        let crc = CRC::<u16>::from_str("init=0x5ff\npoly=0x4465     width=15").unwrap();
        test_shifts(&crc);
        test_find(&crc);
        test_prop(&crc);
        check_example(&crc, 0x2cfa);
        assert!(CRC::<u16>::from_str("init=0x5ff\npoly=0x4465     width=\"15").is_err());
        CRC::<u16>::from_str("  init=0533\n\t\npoly=0x4465     width=\"15\"   ").unwrap();
    }
}

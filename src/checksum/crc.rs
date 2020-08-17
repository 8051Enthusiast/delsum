use super::{Digest, LinearCheck};
use crate::bitnum::BitNum;
use std::fmt::Display;
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
pub struct CRCBuilder<Sum: BitNum> {
    poly: Sum,
    init: Sum,
    xorout: Sum,
    width: usize,
    refin: bool,
    refout: bool,
    check: Option<Sum>,
    name: Option<String>,
}

#[derive(Debug)]
pub enum CRCBuilderErr {
    CheckFail,
    InvalidWidth,
    InvalidPoly,
}

impl<Sum: BitNum> CRCBuilder<Sum> {
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
    /// the parameters are correct.
    pub fn build(&self) -> Result<CRC<Sum>, CRCBuilderErr> {
        // the type needs at least 8 bit so that we can comfortably add bytes to it
        // (i guess it is kind of already impliead by the from<u8> trait)
        if self.poly.bits() < self.width || self.poly.bits() < 8 {
            return Err(CRCBuilderErr::InvalidWidth);
        }
        // the generator polynomial needs to have the lowest bit set in order to be useful
        if self.poly & Sum::one() != Sum::one() {
            return Err(CRCBuilderErr::InvalidPoly);
        }
        let mask = Sum::one() << (self.width - 1);
        let mask = mask ^ (mask - Sum::one());
        let crc = CRC {
            width: self.width,
            poly: self.poly,
            init: self.init,
            xorout: self.xorout,
            refin: self.refin,
            refout: self.refout,
            mask,
            name: self.name.clone(),
            table: CRC::<Sum>::generate_crc_table(self.poly, self.width),
        };
        match self.check {
            Some(chk) => {
                if crc.digest(&b"123456789"[..]).unwrap() != chk {
                    Err(CRCBuilderErr::CheckFail)
                } else {
                    Ok(crc)
                }
            }
            None => Ok(crc),
        }
    }
}

impl<Sum: BitNum> Display for CRC<Sum> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.name {
            Some(n) => write!(f, "{}", n),
            None => write!(f, "<CRC with width {} and generator {:#x}>", self.width, self.poly)
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

impl<Sum: BitNum> CRC<Sum> {
    /// Construct a new CRC from parameters, see also the documentation for CRCBuilder.
    pub fn with_options(poly: Sum, width: usize) -> CRCBuilder<Sum> {
        CRCBuilder {
            poly,
            init: Sum::zero(),
            xorout: Sum::zero(),
            width,
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
        // laughs in GF(2)
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

mod tests {
    use super::*;
    #[allow(dead_code)]
    fn test_shifts<T: LinearCheck>(crc: &T) {
        let test_sum = crc
            .digest(&b"T\x00\x00\x00E\x00\x00\x00S\x00\x00\x00\x00T"[..])
            .unwrap();
        let shift3 = crc.shift_n(3);
        let shift4 = crc.inc_shift(shift3.clone());
        let mut new_sum = crc.init();
        new_sum = crc.dig_byte(new_sum, b'T');
        new_sum = crc.shift(new_sum, &shift3);
        new_sum = crc.dig_byte(new_sum, b'E');
        new_sum = crc.shift(new_sum, &shift3);
        new_sum = crc.dig_byte(new_sum, b'S');
        new_sum = crc.shift(new_sum, &shift4);
        new_sum = crc.dig_byte(new_sum, b'T');
        assert_eq!(test_sum, crc.finalize(new_sum));
    }
    #[allow(dead_code)]
    fn test_find<L: LinearCheck>(crc: &L, sum: L::Sum) {
        assert_eq!(
            crc.find_checksum_segments(&b"a123456789X1235H123456789Y"[..], sum),
            vec![(vec![1], vec![10]), (vec![16], vec![25])]
        );
    }
    #[test]
    fn cms_16() {
        assert!(CRC::<u32>::with_options(0x8005, 16)
            .init(0xffff)
            .check(0xaee7)
            .build()
            .is_ok());
        let crc = CRC::<u16>::with_options(0x8005, 16)
            .init(0xffff)
            .check(0xaee7)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc, 0xaee7);
    }
    #[test]
    fn gsm_3() {
        let crc = CRC::<u8>::with_options(0x3, 3)
            .xorout(0x7)
            .check(0x4)
            .build()
            .unwrap();
        test_shifts(&crc);
        let crc = CRC::<u128>::with_options(0x3, 3)
            .xorout(0x7)
            .check(0x4)
            .build()
            .unwrap();
        test_shifts(&crc);
    }
    #[test]
    fn rohc_7() {
        let crc = CRC::<u8>::with_options(0x4f, 7)
            .init(0x7f)
            .refin(true)
            .refout(true)
            .check(0x53)
            .build()
            .unwrap();
        test_shifts(&crc);
        let crc = CRC::<u32>::with_options(0x4f, 7)
            .init(0x7f)
            .refin(true)
            .refout(true)
            .check(0x53)
            .build()
            .unwrap();
        test_shifts(&crc);
    }
    #[test]
    fn usb_5() {
        let crc = CRC::<u8>::with_options(0x05, 5)
            .init(0x1f)
            .refin(true)
            .refout(true)
            .xorout(0x1f)
            .check(0x19)
            .build()
            .unwrap();
        test_shifts(&crc);
    }
    #[test]
    fn umts_12() {
        let crc = CRC::<u16>::with_options(0x80f, 12)
            .refout(true)
            .check(0xdaf)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc, 0xdaf);
    }
    #[test]
    fn en13757_16() {
        let crc = CRC::<u16>::with_options(0x3d65, 16)
            .xorout(0xffff)
            .check(0xc2b7)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc, 0xc2b7);
    }
    #[test]
    fn mpt1327_15() {
        let crc = CRC::<u16>::with_options(0x6815, 15)
            .xorout(0x0001)
            .check(0x2566)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc, 0x2566);
    }
    #[test]
    fn canfd_17() {
        let crc = CRC::<u32>::with_options(0x1685b, 17)
            .check(0x04f03)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc, 0x04f03);
    }
    #[test]
    fn bzip2_32() {
        let crc = CRC::<u32>::with_options(0x04c11db7, 32)
            .init(0xffffffff)
            .xorout(0xffffffff)
            .check(0xfc891918)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc, 0xfc891918);
    }
    #[test]
    fn iscsi_32() {
        let crc = CRC::<u32>::with_options(0x1edc6f41, 32)
            .init(0xffffffff)
            .xorout(0xffffffff)
            .refin(true)
            .refout(true)
            .check(0xe3069283)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc, 0xe3069283);
    }
    #[test]
    fn gsm_40() {
        let crc = CRC::<u64>::with_options(0x0004820009, 40)
            .xorout(0xffffffffff)
            .check(0xd4164fc646)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc, 0xd4164fc646);
    }
    #[test]
    fn xz_64() {
        let crc = CRC::<u64>::with_options(0x42f0e1eba9ea3693, 64)
            .init(0xffffffffffffffff)
            .refin(true)
            .refout(true)
            .xorout(0xffffffffffffffff)
            .check(0x995dc9bbdf1939fa)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc, 0x995dc9bbdf1939fa);
    }
    #[test]
    fn darc_82() {
        let crc = CRC::<u128>::with_options(0x0308c0111011401440411, 82)
            .refin(true)
            .refout(true)
            .check(0x09ea83f625023801fd612)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc, 0x09ea83f625023801fd612);
    }
    #[test]
    fn parity_1() {
        let crc = CRC::<u8>::with_options(1, 1).check(1).build().unwrap();
        test_shifts(&crc);
    }
    #[test]
    fn i4321_8() {
        let crc = CRC::<u8>::with_options(0x7, 8)
            .xorout(0x55)
            .check(0xa1)
            .build()
            .unwrap();
        test_shifts(&crc);
    }
    #[test]
    fn isoiec144433a_16() {
        let crc = CRC::<u16>::with_options(0x1021, 16)
            .init(0xc6c6u16)
            .refin(true)
            .refout(true)
            .check(0xbf05)
            .build()
            .unwrap();
        test_shifts(&crc);
        test_find(&crc, 0xbf05);
    }
}

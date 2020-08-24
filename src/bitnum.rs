use num_traits::Num;
use std::ops;
// Me: can I have a trait for either u8, u16, u32, u64 or u128?
// Mom: We have a trait for either u8, u16, u32, u64 or u128 at home
// trait for either u8, u16, u32, u64 or u128 at home:
pub trait BitNum:
    Num
    + ops::BitXor<Output = Self>
    + ops::Shl<usize, Output = Self>
    + ops::Shr<usize, Output = Self>
    + ops::BitAnd<Output = Self>
    + ops::Not<Output = Self>
    + Clone
    + Copy
    + Eq
    + Ord
    + From<u8>
    + std::convert::TryInto<u8>
    + std::fmt::Debug
    + std::fmt::LowerHex
    + std::fmt::UpperHex
{
    fn revbits(self) -> Self;
    fn bits(&self) -> usize;
    fn from_dec_or_hex(s: &str) -> Result<Self, Self::FromStrRadixErr> {
        match s.strip_prefix("0x") {
            Some(remain) => Self::from_str_radix(remain, 16),
            None => Self::from_str_radix(s, 10),
        }
    }
}

impl BitNum for u8 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {
        8
    }
}
impl BitNum for u16 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {
        16
    }
}
impl BitNum for u32 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {
        32
    }
}
impl BitNum for u64 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {
        64
    }
}
impl BitNum for u128 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {
        128
    }
}

pub trait Modnum: BitNum {
    fn add_mod(self, rhs: &Self, modulo: &Self) -> Self;
    fn mul_mod(self, rhs: &Self, modulo: &Self) -> Self;
}
// the same stuff a bunch of times (not u128 because i can't be bothered)
impl Modnum for u8 {
    fn add_mod(self, rhs: &Self, modulo: &Self) -> Self {
        if *modulo != 0 {
            ((u16::from(self) + u16::from(*rhs)) % u16::from(*modulo)) as u8
        } else {
            (u16::from(self) + u16::from(*rhs)) as u8
        }
    }
    fn mul_mod(self, rhs: &Self, modulo: &Self) -> Self {
        if *modulo != 0 {
            ((u16::from(self) * u16::from(*rhs)) % u16::from(*modulo)) as u8
        } else {
            (u16::from(self) * u16::from(*rhs)) as u8
        }
    }
}
impl Modnum for u16 {
    fn add_mod(self, rhs: &Self, modulo: &Self) -> Self {
        if *modulo != 0 {
            ((u32::from(self) + u32::from(*rhs)) % u32::from(*modulo)) as u16
        } else {
            (u32::from(self) + u32::from(*rhs)) as u16
        }
    }
    fn mul_mod(self, rhs: &Self, modulo: &Self) -> Self {
        if *modulo != 0 {
            ((u32::from(self) * u32::from(*rhs)) % u32::from(*modulo)) as u16
        } else {
            (u32::from(self) * u32::from(*rhs)) as u16
        }
    }
}
impl Modnum for u32 {
    fn add_mod(self, rhs: &Self, modulo: &Self) -> Self {
        if *modulo != 0 {
            ((u64::from(self) + u64::from(*rhs)) % u64::from(*modulo)) as u32
        } else {
            (u64::from(self) + u64::from(*rhs)) as u32
        }
    }
    fn mul_mod(self, rhs: &Self, modulo: &Self) -> Self {
        if *modulo != 0 {
            ((u64::from(self) * u64::from(*rhs)) % u64::from(*modulo)) as u32
        } else {
            (u64::from(self) * u64::from(*rhs)) as u32
        }
    }
}
impl Modnum for u64 {
    fn add_mod(self, rhs: &Self, modulo: &Self) -> Self {
        if *modulo != 0 {
            ((u128::from(self) + u128::from(*rhs)) % u128::from(*modulo)) as u64
        } else {
            (u128::from(self) + u128::from(*rhs)) as u64
        }
    }
    fn mul_mod(self, rhs: &Self, modulo: &Self) -> Self {
        if *modulo != 0 {
            ((u128::from(self) * u128::from(*rhs)) % u128::from(*modulo)) as u64
        } else {
            (u128::from(self) * u128::from(*rhs)) as u64
        }
    }
}

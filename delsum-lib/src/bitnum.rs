use num_traits::{Num, One};
use std::{convert::TryInto, ops};
/// Me: can I have a trait for either u8, u16, u32, u64 or u128?
/// Mom: We have a trait for either u8, u16, u32, u64 or u128 at home
/// trait for either u8, u16, u32, u64 or u128 at home:
pub trait BitNum:
    Num
    + num_traits::ops::wrapping::WrappingSub
    + num_traits::ops::wrapping::WrappingAdd
    + num_traits::ops::wrapping::WrappingMul
    + num_traits::ops::checked::CheckedSub
    + num_traits::ops::checked::CheckedAdd
    + num_traits::ops::checked::CheckedMul
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
    + Send
    + Sync
{
    fn revbits(self) -> Self;
    fn bits(&self) -> usize;
    fn trail_zeros(&self) -> u32;
    fn from_hex(s: &str) -> Result<Self, Self::FromStrRadixErr> {
        match s.strip_prefix("0x") {
            Some(remain) => Self::from_str_radix(remain, 16),
            None => Self::from_str_radix(s, 16),
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

    fn trail_zeros(&self) -> u32 {
        self.trailing_zeros()
    }
}
impl BitNum for u16 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {
        16
    }

    fn trail_zeros(&self) -> u32 {
        self.trailing_zeros()
    }
}
impl BitNum for u32 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {
        32
    }

    fn trail_zeros(&self) -> u32 {
        self.trailing_zeros()
    }
}
impl BitNum for u64 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {
        64
    }

    fn trail_zeros(&self) -> u32 {
        self.trailing_zeros()
    }
}
impl BitNum for u128 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {
        128
    }

    fn trail_zeros(&self) -> u32 {
        self.trailing_zeros()
    }
}

/// For the modsum, we need a wider type for temporary reduction modulo some number,
/// so this is implemented in this type (and there's probably no need for an u128 ModSum anyway)
pub trait Modnum: BitNum {
    type Double: BitNum + ops::Rem<Output = Self::Double> + From<Self> + TryInto<Self>;
    /// cuts Self::Double in half (ignores the upper half of bits)
    fn from_double(n: Self::Double) -> Self {
        let masked_n = n % (Self::Double::one() << (n.bits() / 2));
        match masked_n.try_into() {
            Ok(k) => k,
            Err(_) => panic!("Half of Double does not fit into original type!"),
        }
    }
    /// add numbers modulo some other number (if 0 then the modulo is 2^n where n is the number of bits)
    fn add_mod(self, rhs: &Self, modulo: &Self) -> Self {
        let dself = Self::Double::from(self);
        let drhs = Self::Double::from(*rhs);
        Self::from_double(if modulo.is_zero() {
            dself + drhs
        } else {
            (dself + drhs) % Self::Double::from(*modulo)
        })
    }
    /// multiply numbers modulo some other number (if 0 then the modulo is 2^n where n is the number of bits)
    fn mul_mod(self, rhs: &Self, modulo: &Self) -> Self {
        let dself = Self::Double::from(self);
        let drhs = Self::Double::from(*rhs);
        Self::from_double(if modulo.is_zero() {
            dself * drhs
        } else {
            (dself * drhs) % Self::Double::from(*modulo)
        })
    }
    /// negate modulo number (if 0 then modulo is 2^n where n is the number of bits)
    fn neg_mod(self, modulo: &Self) -> Self {
        if modulo.is_zero() {
            Self::zero().wrapping_sub(&self)
        } else {
            *modulo - self
        }
    }
    /// convert from u64 with some modulo to the number (0 is again 2^n)
    fn mod_from(n: u64, modulo: &Self) -> Self;
}
// the same stuff a bunch of times (not u128 because i can't be bothered)
impl Modnum for u8 {
    type Double = u16;
    fn mod_from(n: u64, modulo: &Self) -> Self {
        if *modulo != 0 {
            (n % *modulo as u64) as u8
        } else {
            n as u8
        }
    }
}
impl Modnum for u16 {
    type Double = u32;
    fn mod_from(n: u64, modulo: &Self) -> Self {
        if *modulo != 0 {
            (n % *modulo as u64) as u16
        } else {
            n as u16
        }
    }
}
impl Modnum for u32 {
    type Double = u64;
    fn mod_from(n: u64, modulo: &Self) -> Self {
        if *modulo != 0 {
            (n % *modulo as u64) as u32
        } else {
            n as u32
        }
    }
}
impl Modnum for u64 {
    type Double = u128;
    fn mod_from(n: u64, modulo: &Self) -> Self {
        if *modulo != 0 {
            n % *modulo
        } else {
            n
        }
    }
}

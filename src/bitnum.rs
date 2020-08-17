use num_traits::Num;
use std::ops;
// Me: can I have a trait for either u8, u16, u32, u64 or u128?
// Mom: We have a trait for either u8, u16, u32, u64 or u128 at home
// trait for either u8, u16, u32, u64 or u128 at home:
pub trait BitNum: Num + ops::BitXor<Output = Self> + ops::Shl<usize, Output = Self> + ops::Shr<usize, Output = Self> + ops::BitAnd<Output = Self>
                + Clone + Copy + Eq + Ord + From<u8> + std::convert::TryInto<u8> + std::fmt::Debug + std::fmt::LowerHex + std::fmt::UpperHex {
    fn revbits(self) -> Self;
    fn bits(&self) -> usize;
}

impl BitNum for u8 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {8}
}
impl BitNum for u16 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {16}
}
impl BitNum for u32 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {32}
}
impl BitNum for u64 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {64}
}
impl BitNum for u128 {
    fn revbits(self) -> Self {
        self.reverse_bits()
    }
    fn bits(&self) -> usize {128}
}
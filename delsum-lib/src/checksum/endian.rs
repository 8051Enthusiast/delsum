use std::fmt::Display;
use std::str::FromStr;

use crate::bitnum::BitNum;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endian {
    Big,
    Little,
}

impl Display for Endian {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Endian::Little => "little",
            Endian::Big => "big"
        };
        write!(f, "{}", s)
    }
}

impl FromStr for Endian {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_ref() {
            "little" => Ok(Endian::Little),
            "big" => Ok(Endian::Big),
            _ => Err(())
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordSpec {
    pub input_endian: Endian,
    pub word_bytes: usize,
    pub output_endian: Endian,
}

pub fn int_to_bytes<N: BitNum>(n: N, e: Endian, bits: usize) -> Vec<u8> {
    let n_bytes = (bits + 7) / 8;
    let mut ret = Vec::new();
    for x in 0..n_bytes {
        let shift = match e {
            Endian::Little => 8 * x as usize,
            Endian::Big => 8 * (n_bytes - 1 - x) as usize,
        };
        if let Ok(a) = (n >> shift & N::from(0xffu8)).try_into() {
            ret.push(a)
        } else {
            unreachable!()
        }
    }
    ret
}

fn bytes_to_u64(bytes: &[u8], e: Endian) -> u64 {
    let mut ret = 0;
    for (i, &x) in bytes.iter().enumerate() {
        let shift = 8 * match e {
            Endian::Big => (bytes.len() - 1 - i),
            Endian::Little => i,
        };
        ret |= (x as u64) << shift;
    }
    ret
}

impl WordSpec {
    pub fn output_to_bytes<N: BitNum>(&self, s: N, bits: usize) -> Vec<u8> {
        int_to_bytes(s, self.output_endian, bits)
    }
    pub fn iter_words(self, bytes: &'_ [u8]) -> impl Iterator<Item = u64> + '_ {
        (0..(bytes.len() / self.word_bytes as usize))
            .map(move |i| &bytes[self.word_bytes * i..self.word_bytes * (i + 1)])
            .map(move |x| bytes_to_u64(x, self.input_endian))
    }
}

impl Default for WordSpec {
    fn default() -> Self {
        // default is chosen for backwards compability
        WordSpec {
            input_endian: Endian::Big,
            word_bytes: 1,
            output_endian: Endian::Big,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn itobyte() {
        assert_eq!(
            int_to_bytes(0x324234u64, Endian::Little, 24),
            vec![0x34, 0x42, 0x32]
        );
        assert_eq!(
            int_to_bytes(0x324234u64, Endian::Little, 23),
            vec![0x34, 0x42, 0x32]
        );
        assert_eq!(
            int_to_bytes(0x324234u64, Endian::Big, 23),
            vec![0x32, 0x42, 0x34]
        );
        assert_eq!(
            int_to_bytes(0x324234u64, Endian::Little, 40),
            vec![0x34, 0x42, 0x32, 0x00, 0x00]
        );
        assert_eq!(
            int_to_bytes(0x324234u64, Endian::Big, 40),
            vec![0x00, 0x00, 0x32, 0x42, 0x34]
        );
    }
    #[test]
    fn iter_words() {
        let mut ws = WordSpec {
            input_endian: Endian::Little,
            word_bytes: 4,
            output_endian: Endian::Little,
        };
        let test_vec = vec![
            0x01, 0xfe, 0x03, 0xfc, 0x05, 0xfa, 0x07, 0xf8, 0x09, 0xf6, 0x11, 0xf4, 0x13,
        ];
        let words: Vec<_> = ws.iter_words(&test_vec).collect();
        assert_eq!(words, vec![0xfc03fe01, 0xf807fa05, 0xf411f609]);
        let words: Vec<_> = ws.iter_words(&test_vec[..12]).collect();
        assert_eq!(words, vec![0xfc03fe01, 0xf807fa05, 0xf411f609]);
        let words: Vec<_> = ws.iter_words(&test_vec[..11]).collect();
        assert_eq!(words, vec![0xfc03fe01, 0xf807fa05]);
        ws.word_bytes = 3;
        let words: Vec<_> = ws.iter_words(&test_vec).collect();
        assert_eq!(words, vec![0x03fe01, 0xfa05fc, 0x09f807, 0xf411f6]);
        ws.input_endian = Endian::Big;
        let words: Vec<_> = ws.iter_words(&test_vec).collect();
        assert_eq!(words, vec![0x01fe03, 0xfc05fa, 0x07f809, 0xf611f4]);
    }
}

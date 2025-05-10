use std::{
    convert::{TryFrom, TryInto},
    fmt::Display,
    num::ParseIntError,
    str::FromStr,
};

use crate::checksum::Relativity;

/// Turns Result<Iterator, Error> into Iterator<Result<Iterator::Item, Error>> so that
/// on Err, only the single error is iterated, and else the items of the iterator
pub(crate) fn unresult_iter<I, E>(x: Result<I, E>) -> impl Iterator<Item = Result<I::Item, E>>
where
    I: std::iter::Iterator,
{
    let (i, e) = match x {
        Ok(i) => (Some(i.map(Ok)), None),
        Err(e) => (None, Some(std::iter::once(Err(e)))),
    };
    e.into_iter().flatten().chain(i.into_iter().flatten())
}

/// Small function for small cartesian products of small data types
pub(crate) fn cart_prod<T: Clone, U: Clone>(a: &[T], b: &[U]) -> Vec<(T, U)> {
    let mut v = Vec::new();
    for x in a {
        for y in b {
            v.push((x.clone(), y.clone()))
        }
    }
    v
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct InclRange<T: Copy + Ord> {
    start: T,
    end: T,
}

impl<T: Copy + Ord> InclRange<T> {
    pub fn start(&self) -> T {
        self.start
    }
    pub fn end(&self) -> T {
        self.end
    }
}

pub type SignedInclRange = InclRange<isize>;

impl SignedInclRange {
    pub fn new(start: isize, end: isize) -> Option<Self> {
        SignedInclRange { start, end }.valid()
    }
    fn valid(self) -> Option<Self> {
        if (self.start >= 0) == (self.end >= 0) && self.start > self.end {
            None
        } else {
            Some(self)
        }
    }
    pub fn set_start(mut self, start: isize) -> Option<Self> {
        self.start = start;
        self.valid()
    }
    pub fn set_end(mut self, end: isize) -> Option<Self> {
        self.end = end;
        self.valid()
    }
    pub fn to_unsigned(self, len: usize) -> Option<UnsignedInclRange> {
        let unsigned_index = |idx: isize| {
            if idx < 0 {
                len.checked_sub(idx.wrapping_neg() as usize)
            } else {
                Some(idx as usize)
            }
            .and_then(|x| if x < len { Some(x) } else { None })
        };
        UnsignedInclRange::new(unsigned_index(self.start)?, unsigned_index(self.end)?)
    }

    pub fn limit(mut self, len: usize) -> Option<Self> {
        let signed_max = match isize::try_from(len - 1) {
            // if the len does not fit in an isize, limiting will do nothing anyway
            Err(_) => return Some(self),
            Ok(l) => l,
        };
        self.start = self.start.max(-signed_max - 1);
        self.end = self.end.min(signed_max);
        self.valid()
    }

    pub fn slice<T>(self, slice: &[T]) -> Option<&[T]> {
        let unsigned = self.to_unsigned(slice.len())?;
        Some(&slice[unsigned.start..=unsigned.end])
    }
}

pub fn read_signed_maybe_hex(s: &str) -> Result<isize, ParseIntError> {
    s.strip_prefix("0x")
        .map(|s0x| isize::from_str_radix(s0x, 16))
        .unwrap_or_else(|| s.parse())
}

impl FromStr for SignedInclRange {
    // should be enough for now
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let from_maybe_hex = |s: &str| match s {
            "" => Ok(None),
            otherwise => read_signed_maybe_hex(otherwise).map(Some),
        };
        let split = s
            .split(':')
            .map(from_maybe_hex)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let invalid = String::from("Range with non-positive length");
        match *split.as_slice() {
            [Some(x)] => SignedInclRange::new(x, x).ok_or(invalid),
            [Some(start), Some(end)] => SignedInclRange::new(start, end).ok_or(invalid),
            [None, None] => SignedInclRange::new(0, -1).ok_or(invalid),
            [Some(start), None] => SignedInclRange::new(start, -1).ok_or(invalid),
            [None, Some(end)] => SignedInclRange::new(0, end).ok_or(invalid),
            _ => Err(String::from(
                "Wrong number of colons, range must be either 0xab:0xcd or 0xab by itself",
            )),
        }
        .and_then(|x| {
            if (x.start >= 0) != (x.end >= 0) {
                Err(String::from("Range start sign is mismatched with end sign"))
            } else {
                Ok(x)
            }
        })
    }
}

impl Display for SignedInclRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let print_signed = |x: isize, f: &mut std::fmt::Formatter| {
            if x < 0 {
                write!(f, "-0x{:x}", -x)
            } else {
                write!(f, "0x{:x}", x)
            }
        };
        print_signed(self.start, f)?;
        write!(f, ":")?;
        print_signed(self.end, f)
    }
}

pub type UnsignedInclRange = InclRange<usize>;

impl UnsignedInclRange {
    pub fn new(start: usize, end: usize) -> Option<Self> {
        UnsignedInclRange { start, end }.valid()
    }
    fn valid(self) -> Option<Self> {
        if self.start > self.end {
            None
        } else {
            Some(self)
        }
    }
    pub fn set_start(mut self, start: usize) -> Option<Self> {
        self.start = start;
        self.valid()
    }
    pub fn set_end(mut self, end: usize) -> Option<Self> {
        self.end = end;
        self.valid()
    }
    pub fn contains(&self, idx: usize) -> bool {
        idx >= self.start && idx <= self.end
    }
    pub fn len(&self) -> usize {
        self.end - self.start + 1
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn to_signed(
        self,
        start_rel: Relativity,
        end_rel: Relativity,
        len: usize,
    ) -> Option<SignedInclRange> {
        if self.start >= len || self.end >= len {
            return None;
        }
        let signed_index = |idx: usize, rel| match rel {
            Relativity::Start => idx.try_into().ok(),
            Relativity::End => (len - idx)
                .try_into()
                .ok()
                .map(<isize as std::ops::Neg>::neg),
        };
        SignedInclRange::new(
            signed_index(self.start, start_rel)?,
            signed_index(self.end, end_rel)?,
        )
    }
}

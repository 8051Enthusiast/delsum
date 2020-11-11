use crate::bitnum::BitNum;
use bitvec::prelude::*;
use lazy_static::lazy_static;
use rand::prelude::*;
use std::cmp::Ordering;
use std::convert::TryFrom;

// beware, only cs101 and madness lies ahead

lazy_static! {
    static ref WHEEL: BitVec = {
        let mut v = BitVec::repeat(false, 6720);
        for p in &[2, 3, 5, 7] {
            let mut cur = 0;
            while cur < v.len() {
                v.set(cur, true);
                cur += p;
            }
        }
        v
    };
}
#[derive(Clone)]
struct PrimeSieve {
    sieve: BitVec,
    max_prime: usize,
}
impl PrimeSieve {
    fn new() -> PrimeSieve {
        let mut sieve = BitVec::repeat(false, 6720);
        sieve.set(0, true);
        sieve.set(1, true);
        PrimeSieve {
            sieve,
            max_prime: 1,
        }
    }
    fn extend(&mut self) {
        let old_size = self.sieve.len();
        let mut new_sieve_part: BitVec = WHEEL
            .iter()
            .cycle()
            // double
            .take(old_size)
            .collect();
        for (i, &is_composite) in self.sieve.iter().enumerate().take(self.max_prime).skip(11) {
            if is_composite {
                continue;
            }
            let mut current_pos = match old_size % i {
                0 => 0,
                x => i - x,
            };
            while current_pos < new_sieve_part.len() {
                new_sieve_part.set(current_pos, true);
                current_pos += i;
            }
        }
        self.sieve.append(&mut new_sieve_part);
    }
    fn update(&mut self, p: usize) {
        if self.max_prime < p {
            let mut current_pos = 2 * p;
            while current_pos < self.sieve.len() {
                self.sieve.set(current_pos, true);
                current_pos += p
            }
            self.max_prime = p;
        }
    }
    fn next_prime(&mut self, p: usize) -> usize {
        let old_len = self.sieve.len();
        if let Some((i, _)) = self.sieve[p + 1..].iter().enumerate().find(|(_, x)| !**x) {
            let q = p + 1 + i;
            self.update(q);
            return q;
        }
        self.extend();
        match self.sieve[old_len..].iter().enumerate().find(|(_, x)| !**x) {
            Some((j, _)) => {
                self.update(old_len + j);
                old_len + j
            }
            // for some reason the compiler can't determine that there's always a
            // prime between n and 2n smh
            None => unreachable!(),
        }
    }
    fn iter(&'_ mut self) -> PrimeIterator<'_> {
        PrimeIterator {
            sieve: self,
            current_prime: 1,
        }
    }
    fn iter_from(&'_ mut self, n: u64) -> PrimeIterator<'_> {
        if (self.max_prime as u64) < n {
            loop {
                let next = self.next_prime(self.max_prime);
                if next as u64 > n {
                    break;
                }
            }
        }
        PrimeIterator {
            sieve: self,
            current_prime: n.max(1) as usize,
        }
    }
    fn is_prime(&self, n: u64) -> Option<bool> {
        if n > self.max_prime as u64 {
            return None;
        }
        self.sieve.get(n as usize).copied()
    }
}
struct PrimeIterator<'a> {
    sieve: &'a mut PrimeSieve,
    current_prime: usize,
}

impl<'a> Iterator for PrimeIterator<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.current_prime = self.sieve.next_prime(self.current_prime);
        Some(self.current_prime as u64)
    }
}

pub fn gcd<N: BitNum>(mut a: N, mut b: N) -> N {
    if a.is_zero() {
        return b;
    }
    if b.is_zero() {
        return a;
    }
    let a_shift = a.trail_zeros() as usize;
    let b_shift = b.trail_zeros() as usize;
    a = a >> a_shift;
    b = b >> b_shift;
    let common_shift = a_shift.min(b_shift);
    loop {
        if b > a {
            std::mem::swap(&mut a, &mut b);
        }
        a = a - b;
        if a.is_zero() {
            return b << common_shift;
        }
        a = a >> a.trail_zeros() as usize;
    }
}
/// calculates the inverse of x modulo 2^64
/// assumes x is not divisible by 2
fn word_inverse(x: u64) -> u64 {
    // ancient motorola magic
    let mut y = 1;
    let mut prod = x;
    while prod != 1 {
        prod &= !1u64;
        prod &= prod.wrapping_neg();
        y = prod.wrapping_add(y);
        prod = x.wrapping_mul(y);
    }
    y
}

#[inline]
fn split_u128(x: u128) -> [u64; 2] {
    [x as u64, (x >> 64) as u64]
}

#[inline]
fn make_u128(x: &[u64; 2]) -> u128 {
    x[0] as u128 + ((x[1] as u128) << 64)
}

trait FactorNum: BitNum + From<u64> + Into<u128> + rand::distributions::uniform::SampleUniform {
    fn mon_mul_raw(self, a: Self, b: Self, n_inv: u64) -> Self;
    fn checked_pow(self, e: u8) -> Option<Self>;
    fn mod_neg(self, a: Self) -> Self {
        if a.is_zero() {
            a
        } else {
            self - a
        }
    }
    fn mod_add(self, a: Self, b: Self) -> Self;
    fn as_u64(self) -> u64;
}

impl FactorNum for u64 {
    fn mon_mul_raw(self, a: Self, b: Self, n_inv: u64) -> Self {
        let n = self;
        let [t0, t1] = split_u128(a as u128 * b as u128);
        let m = t0.wrapping_mul(n_inv);
        let [zero, carry] = split_u128(t0 as u128 + m as u128 * n as u128);
        assert_eq!(zero, 0);
        let (t0, carry) = t1.overflowing_add(carry);
        if carry || t0 >= n {
            t0.wrapping_sub(n)
        } else {
            t0
        }
    }
    fn checked_pow(self, mut e: u8) -> Option<Self> {
        let mut ret = 1u64;
        for _ in 0..8 {
            ret = ret.checked_mul(ret)?;
            if e & 0x80 != 0 {
                ret = ret.checked_mul(self)?;
            }
            e <<= 1;
        }
        Some(ret)
    }
    fn mod_add(self, a: Self, b: Self) -> Self {
        let (mut ret, ov) = a.overflowing_add(b);
        if ov || ret > self {
            ret = ret.wrapping_sub(self);
        }
        ret
    }
    fn as_u64(self) -> u64 {
        self
    }
}

impl FactorNum for u128 {
    fn mon_mul_raw(self, a: Self, b: Self, n_inv: u64) -> Self {
        let a = split_u128(a);
        let b = split_u128(b);
        let n = split_u128(self);
        // first iteration
        let [t0, carry] = split_u128(a[0] as u128 * b[0] as u128);
        let [t1, t2] = split_u128(a[1] as u128 * b[0] as u128 + carry as u128);
        let m = t0.wrapping_mul(n_inv);
        let [zero, carry] = split_u128(m as u128 * n[0] as u128 + t0 as u128);
        assert_eq!(zero, 0);
        let [t0, carry] = split_u128(m as u128 * n[1] as u128 + t1 as u128 + carry as u128);
        let [t1, t2] = split_u128(t2 as u128 + carry as u128);
        // second iteration
        let [t0, carry] = split_u128(a[0] as u128 * b[1] as u128 + t0 as u128);
        let [t1, carry] = split_u128(a[1] as u128 * b[1] as u128 + t1 as u128 + carry as u128);
        let (t2, ov1) = t2.overflowing_add(carry);
        let m = t0.wrapping_mul(n_inv);
        let [zero, carry] = split_u128(m as u128 * n[0] as u128 + t0 as u128);
        assert_eq!(zero, 0);
        let [t0, carry] = split_u128(m as u128 * n[1] as u128 + carry as u128 + t1 as u128);
        let (t1, ov2) = t2.overflowing_add(carry);
        if ov1 || ov2 || t1 > n[1] || t1 == n[1] && t0 >= n[0] {
            make_u128(&[t0, t1]).wrapping_sub(make_u128(&n))
        } else {
            make_u128(&[t0, t1])
        }
    }
    fn checked_pow(self, mut e: u8) -> Option<Self> {
        let mut ret = 1u128;
        for _ in 0..8 {
            ret = ret.checked_mul(ret)?;
            if e & 0x80 != 0 {
                ret = ret.checked_mul(self)?;
            }
            e <<= 1;
        }
        Some(ret)
    }
    fn mod_add(self, a: Self, b: Self) -> Self {
        let (mut ret, ov) = a.overflowing_add(b);
        if ov || ret > self {
            ret = ret.wrapping_sub(self);
        }
        ret
    }
    fn as_u64(self) -> u64 {
        self as u64
    }
}

struct MonContext<T: FactorNum> {
    n: T,
    one: T,
    r_squared: T,
    n_inv: u64,
}

impl<T: FactorNum> MonContext<T> {
    fn new(n: T) -> MonContext<T> {
        let n_inv = word_inverse(n.as_u64()).wrapping_neg();
        // we pretty much only have to call this function once per
        // factorization, so doing this inefficiently is ok i guess
        let mut one = T::one();
        let two = T::one() + T::one();
        for _ in 0..n.bits() {
            one = one
                .checked_mul(&two)
                .map(|x| if x > n { x - n } else { x })
                .unwrap_or_else(|| one.wrapping_mul(&two).wrapping_sub(&n));
        }
        let mut r_squared = one;
        for _ in 0..n.bits() {
            r_squared = r_squared
                .checked_mul(&two)
                .map(|x| if x > n { x - n } else { x })
                .unwrap_or_else(|| r_squared.wrapping_mul(&two).wrapping_sub(&n));
        }
        MonContext {
            n,
            one,
            r_squared,
            n_inv,
        }
    }
    fn mon_mul(&self, a: T, b: T) -> T {
        self.n.mon_mul_raw(a, b, self.n_inv)
    }
    fn mon_powermod(&self, a: T, mut b: u128) -> T {
        let mut ret = a;
        if b == 0 {
            return self.one;
        }
        if a.is_zero() {
            return T::zero();
        }
        let leading_zero = b.leading_zeros();
        b <<= leading_zero;
        for _ in (leading_zero + 1)..128 {
            b <<= 1;
            ret = self.mon_mul(ret, ret);
            if b & (1 << 127) != 0 {
                ret = self.mon_mul(ret, a);
            }
        }
        ret
    }
    fn to_mon(&self, a: T) -> T {
        self.mon_mul(a, self.r_squared)
    }
    fn from_mon(&self, a: T) -> T {
        self.mon_mul(a, T::one())
    }
    // new_n divides n
    fn update(&mut self, new_n: T) {
        self.n = new_n;
        self.n_inv = word_inverse(new_n.as_u64()).wrapping_neg();
        self.r_squared = self.r_squared % new_n;
        self.one = self.n.mon_mul_raw(self.r_squared, T::one(), self.n_inv);
    }
}

fn get_exact_root<N: FactorNum>(n: N, e: u8) -> Result<N, N> {
    let mut n_shift = n;
    let mut shift_amount = 0u8;
    while !n_shift.is_zero() {
        n_shift = n_shift >> e as usize;
        shift_amount += 1;
    }
    let mut estimate = N::zero();
    for s in (0..shift_amount).rev() {
        estimate = estimate << 1;
        match (estimate + N::one()).checked_pow(e) {
            None => continue,
            Some(x) => {
                if x <= (n >> (s * e) as usize) {
                    estimate = estimate + N::one()
                }
            }
        }
    }
    let p = estimate.checked_pow(e).unwrap();
    if p == n {
        Ok(estimate)
    } else {
        Err(estimate)
    }
}

// fun fact: this entire function gets loop-unrolled by llvm when optimizing
fn maximum_power_iter(bound: u64, mut base: u64) -> u64 {
    let mut stack = [0u64; 6];
    let mut stack_idx = 0;
    loop {
        stack[stack_idx] = base;
        base = match base.checked_mul(base).map(|x| (x, x.cmp(&bound))) {
            Some((x, Ordering::Equal)) => return x,
            Some((x, Ordering::Less)) => x,
            _ => break,
        };
        stack_idx += 1;
    }
    loop {
        let current_square = stack[stack_idx];
        base = match base.checked_mul(current_square).map(|x| (x, x.cmp(&bound))) {
            Some((x, Ordering::Equal)) => return x,
            Some((x, Ordering::Less)) => x,
            _ => base,
        };
        if stack_idx == 0 {
            return base;
        }
        stack_idx -= 1;
    }
}

fn maximum_power(bound: u64, base: u64) -> u64 {
    if base > bound {
        return 1;
    }
    if bound == 0 || base <= 1 {
        panic!("invalid arguments to maximum_power");
    }
    maximum_power_iter(bound, base)
}

static PRIMES: [u8; 31] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127,
];

#[allow(unused)]
/// made this function thinking i'd need this,
/// turns out i didn't
/// i still have an emotional bond to it, so i won't delete it
fn perfect_power<N: FactorNum>(mut n: N) -> (N, u8) {
    let mut current_power = 1;
    // PRIMES is in P
    for p in PRIMES.iter() {
        if (n >> *p as usize).is_zero() {
            break;
        }
        while let Ok(b) = get_exact_root(n, *p) {
            current_power *= p;
            n = b;
        }
    }
    (n, current_power)
}

fn next_factor<N: FactorNum>(
    mon: &mut MonContext<N>,
    current_p: &mut u64,
    current_power: &mut N,
    sieve: &mut PrimeSieve,
    bound: u64,
) -> Option<N> {
    let neg_one = mon.n.mod_neg(mon.one);
    for p in sieve.iter_from(*current_p).take_while(|&x| x <= bound) {
        *current_p = p;
        let q = maximum_power(bound, p);
        *current_power = mon.mon_powermod(*current_power, q as u128);
        let p_minus_one = mon.n.mod_add(*current_power, neg_one);
        let g = gcd(p_minus_one, mon.n);
        if !g.is_one() {
            let n = mon.n / g;
            *current_power = *current_power % n;
            mon.update(n);
            return Some(g);
        }
    }
    None
}

fn trial_div<N: FactorNum>(mut n: N, sieve: &mut PrimeSieve, mut bound: u64) -> Vec<(u128, u8)> {
    let mut ret = Vec::new();
    for p in sieve.iter() {
        if bound < p {
            break;
        }
        let mut mult = 0u8;
        while (n % p.into()).is_zero() {
            n = n / p.into();
            mult += 1;
        }
        if mult > 0 {
            let sqrt = get_exact_root(n, 2).unwrap_or_else(|x| x).as_u64();
            bound = sqrt.min(bound);
            ret.push((p as u128, mult));
        }
    }
    if !n.is_one() {
        ret.push((n.into(), 1));
    }
    ret
}

fn merge_factors(a: &[(u128, u8)], b: &[(u128, u8)]) -> Vec<(u128, u8)> {
    let mut ret = Vec::new();
    let mut a_idx = 0;
    let mut b_idx = 0;
    while a_idx < a.len() && b_idx < b.len() {
        match a[a_idx].0.cmp(&b[b_idx].0) {
            Ordering::Less => {
                ret.push(a[a_idx]);
                a_idx += 1;
            }
            Ordering::Greater => {
                ret.push(b[b_idx]);
                b_idx += 1;
            }
            Ordering::Equal => {
                let p = a[a_idx].0;
                let mult = a[a_idx].1 + b[b_idx].1;
                ret.push((p, mult));
                a_idx += 1;
                b_idx += 1;
            }
        }
    }
    ret.extend_from_slice(&a[a_idx..]);
    ret.extend_from_slice(&b[b_idx..]);
    ret
}

fn p1fac<N: FactorNum>(
    n: &mut N,
    start_power: N,
    prime: &mut u64,
    sieve: &mut PrimeSieve,
    mut bound: u64,
    lower_n: u64,
) -> (N, Vec<N>) {
    let mut ret = Vec::new();
    let mut mon = MonContext::new(*n);
    let mut power = mon.to_mon(start_power);
    while mon.n > N::from(lower_n) {
        if is_prob_prime(mon.n) {
            *n = N::one();
            ret.push(mon.n);
            return (N::zero(), ret);
        }
        let fac_op = next_factor(&mut mon, prime, &mut power, sieve, (bound + 1) >> 1);
        let fac = match fac_op {
            Some(f) => f,
            None => break,
        };
        let sqrt = get_exact_root(mon.n, 2).unwrap_or_else(|x| x);
        bound = bound.min(sqrt.as_u64());
        ret.push(fac);
    }
    *n = mon.n;
    (mon.from_mon(power), ret)
}

fn is_prob_prime<N: FactorNum>(n: N) -> bool {
    if (n & N::one()).is_zero() {
        return false;
    }
    if n < N::from(128u8) {
        let nu8 = match n.try_into() {
            Ok(x) => x,
            Err(_) => panic!("Can't convert number < 128 into u8 for some reason???"),
        };
        return PRIMES.binary_search(&nu8).is_ok();
    }
    let n1 = n - N::one();
    let mon = MonContext::new(n);
    let trail_zero = n1.trail_zeros() as usize;
    let d = n1 >> trail_zero;
    let mut rng = thread_rng();
    let minus_one = n.mod_neg(mon.one);
    // 32 rounds for a 2^-64 probability of false positive
    'a: for _ in 0..32 {
        let mut witness = mon.to_mon(rng.gen_range(N::one() + N::one(), n1));
        witness = mon.mon_powermod(witness, d.into());
        if witness == mon.one || witness == minus_one {
            continue;
        }
        for _ in 0..trail_zero - 1 {
            witness = mon.mon_mul(witness, witness);
            if witness == minus_one {
                continue 'a;
            }
        }
        return false;
    }
    true
}

fn factor(mut num: u128, bound: u128) -> Vec<(u128, u8)> {
    let mut prime_factors = Vec::new();
    if num == 0 {
        panic!("refusing to factor 0");
    }
    let tz = num.trailing_zeros();
    if tz > 0 {
        prime_factors.push((2u128, tz as u8));
        num >>= tz;
    }
    let mut sieve = PrimeSieve::new();
    let mut prime = 1;
    let sqrt = get_exact_root(num, 2).unwrap_or_else(|x| x);
    let bound = bound.min(sqrt) as u64;
    let (power, maybe_prime128) = p1fac(&mut num, 2u128, &mut prime, &mut sieve, bound, u64::MAX);
    let maybe_prime64 = if let Ok(mut x) = u64::try_from(num) {
        let new_power = power as u64 % x;
        let (_, maybe_prime64) = p1fac(&mut x, new_power, &mut prime, &mut sieve, bound, 1);
        if x != 1 {
            prime_factors.push((x as u128, 1));
        }
        maybe_prime64
    } else {
        // prime enough for our purpose
        prime_factors.push((num, 1));
        vec![]
    };
    for factor in maybe_prime128 {
        prime_factors = if is_prob_prime(factor) {
            merge_factors(&[(factor, 1)], &prime_factors)
        } else {
            merge_factors(&trial_div(factor, &mut sieve, bound), &prime_factors)
        };
    }
    for factor in maybe_prime64 {
        prime_factors = if sieve.is_prime(factor).unwrap_or_else(|| is_prob_prime(factor)) {
            merge_factors(&[(factor as u128, 1)], &prime_factors)
        } else {
            let trial_fac: Vec<_> = trial_div(factor, &mut sieve, bound)
                .iter()
                .map(|(p, e)| (*p as u128, *e))
                .collect();
            merge_factors(&trial_fac, &prime_factors)
        }
    }
    prime_factors
}

fn div_combs(mut cur: u128, facs: &[(u128, u8)], low: u128, high: u128) -> Vec<u128> {
    let mut ret = Vec::new();
    match facs.split_last() {
        Some(((fac, mul), other_facs)) => {
            ret.append(&mut div_combs(cur, other_facs, low, high));
            for _ in 0..*mul {
                let (new_cur, ov) = fac.overflowing_mul(cur);
                if ov || new_cur > high {
                    break;
                } else {
                    cur = new_cur;
                    ret.append(&mut div_combs(cur, other_facs, low, high));
                }
            }
        }
        None => {
            if cur >= low && cur <= high {
                ret.push(cur);
            }
        }
    }
    ret
}

pub fn divisors_range(number: u128, low: u128, high: u128) -> Vec<u128> {
    let switch_div = number / low > high;
    let (new_high, new_low) = if switch_div {
        (number / low, number / high)
    } else {
        (high, low)
    };
    let facs = factor(number, new_high);
    let mut divs = div_combs(1, &facs, new_low, new_high);
    if switch_div {
        divs = divs
            .iter()
            .map(|x| number / x)
            .filter(|&x| x <= high && x >= low)
            .collect();
    }
    divs.sort_unstable();
    divs
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;
    #[test]
    fn sieve_correct_primes() {
        let mut sieve = PrimeSieve::new();
        let primes_20: Vec<_> = sieve.iter().take(20).collect();
        assert_eq!(
            primes_20,
            vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        );
    }
    #[test]
    fn sieve_extend() {
        let mut sieve = PrimeSieve::new();
        let mut prime_pi = |s| sieve.iter().take_while(|x| *x <= s).count();
        assert_eq!(prime_pi(6720), 867);
        assert_eq!(prime_pi(6732), 867);
        assert_eq!(prime_pi(6733), 868);
        assert_eq!(prime_pi(6800), 875);
        assert_eq!(prime_pi(15000), 1754);
        assert_eq!(prime_pi(1500000), 114155);
        assert_eq!(prime_pi(150000), 13848);
        let mut prev_prime = |s| sieve.iter().take_while(|x| *x < s).last();
        assert_eq!(prev_prime(150000), Some(149993));
        assert_eq!(prev_prime(1500000), Some(1499977));
    }
    #[test]
    fn sieve_iter_from() {
        let mut sieve = PrimeSieve::new();
        let mut next_prime = |s| sieve.iter_from(s).next().unwrap();
        assert_eq!(next_prime(1), 2);
        assert_eq!(next_prime(10), 11);
        assert_eq!(next_prime(6720), 6733);
        assert_eq!(next_prime(10000), 10007);
        assert_eq!(next_prime(1000), 1009);
        assert_eq!(next_prime(65536), 65537);
        assert_eq!(next_prime(65537), 65539);
        assert_eq!(next_prime(0), 2);
    }
    #[test]
    fn gcd_u128() {
        assert_eq!(
            gcd::<u128>(
                806181092647704672515872930213587774,
                701634946365911841808990539472970814
            ),
            13497319857138975198
        );
        assert_eq!(
            gcd::<u128>(806181092647704672515872930213587774, 0),
            806181092647704672515872930213587774
        );
        assert_eq!(
            gcd::<u128>(0, 806181092647704672515872930213587774),
            806181092647704672515872930213587774
        );
    }
    #[quickcheck]
    fn qc_gcd(a: u64, b: u64, c: u64) -> TestResult {
        let a_c = a as u128 * c as u128;
        let b_c = b as u128 * c as u128;
        let result = gcd(a_c, b_c);
        if c == 0 && result == 0 {
            return TestResult::passed();
        }
        TestResult::from_bool(gcd(a_c, b_c) % c as u128 == 0)
    }
    #[quickcheck]
    fn qc_word_inverse(x: u64) -> TestResult {
        if x == 0 {
            return TestResult::discard();
        }
        let x_red = x >> x.trailing_zeros();
        TestResult::from_bool(x_red.wrapping_mul(word_inverse(x_red)) == 1)
    }
    #[test]
    fn max_power() {
        assert_eq!(maximum_power(18446744073709551615, 2), 9223372036854775808);
        assert_eq!(maximum_power(1, 2), 1);
        assert_eq!(maximum_power(27, 3), 27);
        assert_eq!(maximum_power(17515755615234375, 5), 11920928955078125);
        assert_eq!(maximum_power(17515755615234375, 3), 16677181699666569);
    }
    #[test]
    fn exact_root() {
        assert_eq!(get_exact_root(10000u64, 2), Ok(100u64));
        assert_eq!(get_exact_root(10000u64, 3), Err(21u64));
        assert_eq!(get_exact_root(10000u128, 2), Ok(100u128));
        assert_eq!(get_exact_root(18446744073709551616u128, 32), Ok(4u128));
        assert_eq!(get_exact_root(18446744073709551615u64, 2), Err(4294967295));
        assert_eq!(
            get_exact_root(22528399544939174411840147874772641u128, 72),
            Ok(3)
        );
        assert_eq!(
            get_exact_root(170141183460469231731687303715884105728u128, 127),
            Ok(2)
        );
        assert_eq!(get_exact_root(0u64, 13), Ok(0));
    }
    #[test]
    fn perfect_max_power() {
        assert_eq!(perfect_power(100u64), (10, 2));
        assert_eq!(perfect_power(100u128), (10, 2));
        assert_eq!(perfect_power(100u128), (10, 2));
        assert_eq!(perfect_power(9223372036854775808u64), (2, 63));
        assert_eq!(
            perfect_power(170141183460469231731687303715884105728u128),
            (2, 127)
        );
        assert_eq!(perfect_power(1413u64), (1413, 1));
        assert_eq!(perfect_power(0u64), (0, 1));
    }
    #[test]
    fn test_factor1() {
        assert_eq!(
            factor(1243713847913758136, 1243713847913758136),
            vec![
                (2, 3),
                (17, 1),
                (61, 1),
                (151, 1),
                (269, 1),
                (12071, 1),
                (305759, 1)
            ]
        );
    }
    #[test]
    fn test_factor2() {
        assert_eq!(
            factor(1243713847913758136471, 1243713847913758136471),
            vec![(2267, 1), (22243673, 1), (24663939581, 1),]
        );
    }
    #[test]
    fn test_factor3() {
        assert_eq!(
            factor(
                4957391571224061778730779231513u128,
                4957391571224061778730779231513u128
            ),
            vec![(12282749, 1), (403606030801741676780237, 1)]
        )
    }
    #[test]
    fn primes() {
        assert_eq!(is_prob_prime(1433u64), true);
        assert_eq!(is_prob_prime(1431u128), false);
        assert_eq!(is_prob_prime(283408420968530627042721395551u128), true);
        assert_eq!(is_prob_prime(4957391571224061778730779295067043u128), true);
        assert_eq!(is_prob_prime(4957391571224061778730779295067041u128), false);
        assert_eq!(is_prob_prime(4957391571224061778730779295067042u128), false);
        assert_eq!(is_prob_prime(3u64), true);
        assert_eq!(is_prob_prime(9u64), false);
    }
    #[test]
    fn combs() {
        assert_eq!(
            divisors_range(21627638661127035875894778593280, 65530, 65536),
            vec![65530, 65531, 65532, 65533, 65534, 65535, 65536]
        );
        assert_eq!(
            divisors_range(21627638661127035875894778593280, 65531, 65535),
            vec![65531, 65532, 65533, 65534, 65535]
        );
        assert_eq!(
            divisors_range(13875613813, 7000000, 8000000),
            vec![7299113]
        );
        assert_eq!(
            divisors_range(1749336101070558, 500000, 600000),
            vec![513799, 517881, 528171, 531973, 536851, 540813, 542542, 542543, 553322, 555529, 566566, 572894, 577122, 598262]
        );
        assert_eq!(
            divisors_range(287000, 1000, 1500),
            vec![1000, 1025, 1148, 1400, 1435]
        );
    }
}

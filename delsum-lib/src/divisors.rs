use crate::bitnum::BitNum;
// binary gcd algorithm
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

// given a factorization `facs`, finds all combination divisors of facs with low <= d <= high
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

/// Finds all divisors of number with low <= d <= high
pub fn divisors_range(number: u128, low: u128, high: u128) -> Vec<u128> {
    let switch_div = number / low > high;
    let (new_high, new_low) = if switch_div {
        (number / low, number / high)
    } else {
        (high, low)
    };
    let (facs, p) = num_prime::nt_funcs::factors(number, None);
    assert!(p.is_none());
    let facs = facs
        .into_iter()
        .map(|(p, e)| (p, e as u8))
        .collect::<Vec<_>>();
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
        assert_eq!(divisors_range(13875613813, 7000000, 8000000), vec![7299113]);
        assert_eq!(
            divisors_range(1749336101070558, 500000, 600000),
            vec![
                513799, 517881, 528171, 531973, 536851, 540813, 542542, 542543, 553322, 555529,
                566566, 572894, 577122, 598262
            ]
        );
        assert_eq!(
            divisors_range(287000, 1000, 1500),
            vec![1000, 1025, 1148, 1400, 1435]
        );
    }
}

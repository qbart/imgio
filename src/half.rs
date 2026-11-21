//! IEEE 754 binary16 conversion.

/// Converts a half (as bits) to `f32` exactly.
pub fn to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 31) as u32;
    let man = (h & 1023) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // subnormal: normalize
            let shift = man.leading_zeros() - 21; // bring the top set bit to bit 10
            let man = (man << shift) & 1023;
            sign | ((113 - shift) << 23) | (man << 13)
        }
        31 => sign | 0x7F80_0000 | (man << 13),
        e => sign | ((e + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Converts `f32` to half bits, rounding to nearest even. Overflow gives infinity.
pub fn from_f32(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let man = x & 0x7F_FFFF;
    if exp == 0xFF {
        let nan = if man != 0 {
            0x200 | (man >> 13) as u16
        } else {
            0
        };
        return sign | 0x7C00 | nan;
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if rem > halfway || (rem == halfway && r & 1 != 0) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 != 0) {
        r += 1; // may carry into the exponent, up to infinity: correct
    }
    sign | r as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all() {
        for h in 0..=u16::MAX {
            let f = to_f32(h);
            if f.is_nan() {
                assert!(to_f32(from_f32(f)).is_nan());
            } else {
                assert_eq!(from_f32(f), h, "{h:#06x}");
            }
        }
    }

    #[test]
    fn rounding() {
        assert_eq!(from_f32(1.0), 0x3C00);
        assert_eq!(from_f32(65504.0), 0x7BFF);
        assert_eq!(from_f32(65520.0), 0x7C00); // rounds up to inf
        assert_eq!(from_f32(1e-8), 0);
        assert_eq!(from_f32(5.960_464_5e-8), 1); // smallest subnormal
        // halfway between 1.0 and next half (1 + 2^-10): ties to even (1.0)
        assert_eq!(from_f32(1.0 + 2f32.powi(-11)), 0x3C00);
        assert_eq!(from_f32(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
        assert_eq!(to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(to_f32(0xC000), -2.0);
    }
}

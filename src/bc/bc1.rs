//! BC1-BC5 block decoders (D3D10 behavior: BC2/BC3 color is always 4-color).

fn expand565(c: u16) -> [u32; 3] {
    let r = ((c >> 11) & 31) as u32;
    let g = ((c >> 5) & 63) as u32;
    let b = (c & 31) as u32;
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

/// Decodes the 8-byte color part to 16 RGBA8 pixels. `four_color` forces 4-color mode (BC2/BC3).
pub(crate) fn color(block: &[u8; 8], four_color: bool, out: &mut [[u8; 4]; 16]) {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let e0 = expand565(c0);
    let e1 = expand565(c1);
    let mut pal = [[0u8; 4]; 4];
    for c in 0..3 {
        pal[0][c] = e0[c] as u8;
        pal[1][c] = e1[c] as u8;
    }
    pal[0][3] = 255;
    pal[1][3] = 255;
    if four_color || c0 > c1 {
        for c in 0..3 {
            pal[2][c] = ((e0[c] * 2 + e1[c]) / 3) as u8;
            pal[3][c] = ((e0[c] + e1[c] * 2) / 3) as u8;
        }
        pal[2][3] = 255;
        pal[3][3] = 255;
    } else {
        for c in 0..3 {
            pal[2][c] = ((e0[c] + e1[c]) / 2) as u8;
        }
        pal[2][3] = 255;
        // pal[3] stays transparent black
    }
    let sel = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for (i, px) in out.iter_mut().enumerate() {
        *px = pal[((sel >> (i * 2)) & 3) as usize];
    }
}

/// BC2 explicit 4-bit alpha (8 bytes) into the alpha channel.
pub(crate) fn alpha4(block: &[u8; 8], out: &mut [[u8; 4]; 16]) {
    let bits = u64::from_le_bytes(*block);
    for (i, px) in out.iter_mut().enumerate() {
        let a = ((bits >> (i * 4)) & 15) as u8;
        px[3] = a | (a << 4);
    }
}

/// BC4 unsigned channel (also BC3 alpha) to 16 values.
pub(crate) fn bc4u(block: &[u8; 8], out: &mut [u8; 16]) {
    let r0 = block[0] as u32;
    let r1 = block[1] as u32;
    let mut pal = [0u8; 8];
    pal[0] = r0 as u8;
    pal[1] = r1 as u8;
    if r0 > r1 {
        for i in 1..7u32 {
            pal[i as usize + 1] = (((7 - i) * r0 + i * r1) / 7) as u8;
        }
    } else {
        for i in 1..5u32 {
            pal[i as usize + 1] = (((5 - i) * r0 + i * r1) / 5) as u8;
        }
        pal[6] = 0;
        pal[7] = 255;
    }
    let mut idx = [0u8; 8];
    idx[..6].copy_from_slice(&block[2..8]);
    let bits = u64::from_le_bytes(idx);
    for (i, v) in out.iter_mut().enumerate() {
        *v = pal[((bits >> (i * 3)) & 7) as usize];
    }
}

/// BC4 signed channel to 16 values (two's complement bytes). Interpolants round down, like
/// the unsigned path.
pub(crate) fn bc4s(block: &[u8; 8], out: &mut [u8; 16]) {
    let r0 = (block[0] as i8).max(-127) as i32;
    let r1 = (block[1] as i8).max(-127) as i32;
    let mut pal = [0i32; 8];
    pal[0] = r0;
    pal[1] = r1;
    if r0 > r1 {
        for i in 1..7i32 {
            pal[i as usize + 1] = ((7 - i) * r0 + i * r1).div_euclid(7);
        }
    } else {
        for i in 1..5i32 {
            pal[i as usize + 1] = ((5 - i) * r0 + i * r1).div_euclid(5);
        }
        pal[6] = -127;
        pal[7] = 127;
    }
    let mut idx = [0u8; 8];
    idx[..6].copy_from_slice(&block[2..8]);
    let bits = u64::from_le_bytes(idx);
    for (i, v) in out.iter_mut().enumerate() {
        *v = pal[((bits >> (i * 3)) & 7) as usize] as i8 as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bc1_four_and_three_color() {
        // c0 = white, c1 = black, selectors 0,1,2,3 repeating.
        let b = [0xFF, 0xFF, 0x00, 0x00, 0xE4, 0xE4, 0xE4, 0xE4];
        let mut px = [[0u8; 4]; 16];
        color(&b, false, &mut px);
        assert_eq!(px[0], [255, 255, 255, 255]);
        assert_eq!(px[1], [0, 0, 0, 255]);
        assert_eq!(px[2], [170, 170, 170, 255]);
        assert_eq!(px[3], [85, 85, 85, 255]);
        // swapped endpoints: 3-color mode with transparent black.
        let b = [0x00, 0x00, 0xFF, 0xFF, 0xE4, 0xE4, 0xE4, 0xE4];
        color(&b, false, &mut px);
        assert_eq!(px[2], [127, 127, 127, 255]);
        assert_eq!(px[3], [0, 0, 0, 0]);
        color(&b, true, &mut px);
        assert_eq!(px[3], [170, 170, 170, 255]);
    }

    #[test]
    fn bc4_palettes() {
        // r0=255, r1=0, indices 0..7 in the first 8 pixels.
        let b = [255, 0, 0b10001000, 0b11000110, 0b11111010, 0, 0, 0];
        let mut v = [0u8; 16];
        bc4u(&b, &mut v);
        assert_eq!(&v[..8], &[255, 0, 218, 182, 145, 109, 72, 36]);
        let b = [0, 255, 0b10001000, 0b11000110, 0b11111010, 0, 0, 0];
        bc4u(&b, &mut v);
        assert_eq!(&v[..8], &[0, 255, 51, 102, 153, 204, 0, 255]);
        let b = [0x80, 0x7F, 0b10001000, 0b11000110, 0b11111010, 0, 0, 0];
        bc4s(&b, &mut v);
        assert_eq!(v[0] as i8, -127);
        assert_eq!(v[6] as i8, -127);
        assert_eq!(v[7] as i8, 127);
    }

    #[test]
    fn bc2_alpha() {
        let b = [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE];
        let mut px = [[0u8; 4]; 16];
        alpha4(&b, &mut px);
        for (i, p) in px.iter().enumerate() {
            assert_eq!(p[3], (i as u8) * 17);
        }
    }
}

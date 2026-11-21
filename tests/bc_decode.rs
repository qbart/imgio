//! BC decoders against Pillow's independent decoders (see scripts/gen_bc_testdata.py).

use imgio::Format;
use std::error::Error;

type R = Result<(), Box<dyn Error>>;

fn load(name: &str) -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/bc/");
    Ok((
        std::fs::read(format!("{dir}{name}.bc"))?,
        std::fs::read(format!("{dir}{name}.expected"))?,
    ))
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 31) as i32;
    let man = (h & 1023) as f32;
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    sign * v
}

/// Decodes `name`, maps each output pixel with `map` to Pillow's channel layout and compares.
fn check(
    name: &str,
    format: Format,
    w: u32,
    h: u32,
    expected_channels: usize,
    tolerance: u8,
    map: impl Fn(&[u8]) -> Vec<u8>,
) -> R {
    let (src, expected) = load(name)?;
    let out = imgio::decode_to_buffer(format, &src, w, h)?;
    let px = format.decoded().block_bytes();
    assert_eq!(out.len(), (w * h) as usize * px);
    let mut mismatches = 0;
    for (i, (o, e)) in out
        .chunks_exact(px)
        .zip(expected.chunks_exact(expected_channels))
        .enumerate()
    {
        let got = map(o);
        let close =
            got.len() == e.len() && got.iter().zip(e).all(|(g, e)| g.abs_diff(*e) <= tolerance);
        // Known difference: invalid BC7 blocks decode to transparent black (D3D spec), Pillow
        // returns opaque black.
        let invalid_bc7 =
            matches!(format, Format::Bc7Unorm) && got == [0, 0, 0, 0] && e == [0, 0, 0, 255];
        if !close && !invalid_bc7 {
            if mismatches < 5 {
                eprintln!("{name}: pixel {i}: got {got:?}, expected {e:?}");
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{name}: {mismatches} mismatching pixels");
    Ok(())
}

fn rgba(p: &[u8]) -> Vec<u8> {
    p.to_vec()
}

#[test]
fn bc1() -> R {
    check("bc1_unorm", Format::Bc1Unorm, 64, 64, 4, 0, rgba)?;
    check("bc1_unorm_odd", Format::Bc1Srgb, 30, 18, 4, 0, rgba)
}

#[test]
fn bc2() -> R {
    check("bc2_unorm", Format::Bc2Unorm, 64, 64, 4, 0, rgba)
}

#[test]
fn bc3() -> R {
    check("bc3_unorm", Format::Bc3Unorm, 64, 64, 4, 0, rgba)
}

#[test]
fn bc4() -> R {
    check("bc4_unorm", Format::Bc4Unorm, 64, 64, 1, 0, rgba)
}

#[test]
fn bc5() -> R {
    check("bc5_unorm", Format::Bc5Unorm, 64, 64, 3, 0, |p| {
        vec![p[0], p[1], 0]
    })?;
    // Pillow shows signed values offset by 128 and does not clamp a -128 endpoint to -127
    // as the spec requires, hence the tolerance of 1.
    check("bc5_snorm", Format::Bc5Snorm, 64, 64, 3, 1, |p| {
        vec![
            (p[0] as i8 as i32 + 128) as u8,
            (p[1] as i8 as i32 + 128) as u8,
            128,
        ]
    })
}

/// Pillow clamps BC6H to [0, 1] and truncates to 8 bits. Its interpolation also drops the
/// `+ 32` rounding of the spec (DirectXTex and basisu keep it), so allow 1 step of difference.
fn bc6h_to_u8(p: &[u8]) -> Vec<u8> {
    (0..3)
        .map(|c| {
            let v = half_to_f32(u16::from_le_bytes([p[c * 2], p[c * 2 + 1]]));
            (v.clamp(0.0, 1.0) * 255.0) as u8
        })
        .collect()
}

#[test]
fn bc6h() -> R {
    // Pillow's signed BC6H decoder is wrong (bcdec agrees with us), so only unsigned here.
    check("bc6h_ufloat", Format::Bc6hUfloat, 64, 64, 3, 1, bc6h_to_u8)
}

/// Bit-exact comparison with bcdec_rs (a second, independent decoder).
fn check_bcdec(name: &str, format: Format, w: u32, h: u32) -> R {
    let (src, _) = load(name)?;
    let out = imgio::decode_to_buffer(format, &src, w, h)?;
    let px = format.decoded().block_bytes();
    let bw = (w / 4) as usize;
    let mut mismatches = 0;
    for (b, blk) in src.as_chunks::<16>().0.iter().enumerate() {
        let mut want = [0u8; 16 * 8];
        match format {
            Format::Bc6hUfloat | Format::Bc6hSfloat => {
                let mut halves = [0u16; 48];
                bcdec_rs::bc6h_half(blk, &mut halves, 12, format == Format::Bc6hSfloat);
                for i in 0..16 {
                    for c in 0..3 {
                        want[i * 8 + c * 2..i * 8 + c * 2 + 2]
                            .copy_from_slice(&halves[i * 3 + c].to_le_bytes());
                    }
                    want[i * 8 + 6..i * 8 + 8].copy_from_slice(&0x3C00u16.to_le_bytes());
                }
            }
            _ => bcdec_rs::bc7(blk, &mut want, 16),
        }
        let (bx, by) = (b % bw, b / bw);
        for y in 0..4 {
            let o = ((by * 4 + y) * w as usize + bx * 4) * px;
            let got = &out[o..o + 4 * px];
            let exp = &want[y * 4 * px..(y + 1) * 4 * px];
            // Invalid BC7 blocks: transparent black here (D3D spec), opaque in bcdec.
            if got != exp && !(blk[0] == 0 && got.iter().all(|&v| v == 0)) {
                mismatches += 1;
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "{name}: {mismatches} mismatching block rows vs bcdec"
    );
    Ok(())
}

#[test]
fn bc6h_bcdec() -> R {
    check_bcdec("bc6h_ufloat", Format::Bc6hUfloat, 64, 64)?;
    check_bcdec("bc6h_sfloat", Format::Bc6hSfloat, 64, 64)
}

#[test]
fn bc7_bcdec() -> R {
    check_bcdec("bc7_unorm", Format::Bc7Unorm, 128, 128)
}

#[test]
fn bc7() -> R {
    check("bc7_unorm", Format::Bc7Unorm, 128, 128, 4, 0, rgba)
}

#[test]
fn errors() {
    let mut dst = [0u8; 64];
    assert!(imgio::decode(Format::Rgba8Unorm, &[0; 16], 4, 4, &mut dst).is_err());
    assert!(imgio::decode(Format::Bc7Unorm, &[0; 15], 4, 4, &mut dst).is_err());
    assert!(imgio::decode(Format::Bc7Unorm, &[0; 16], 4, 4, &mut dst[..63]).is_err());
    assert!(imgio::decode(Format::Bc7Unorm, &[0; 16], 4, 4, &mut dst).is_ok());
    assert!(dst.iter().all(|&b| b == 0)); // invalid mode -> transparent black
}

//! ETC1/ETC2/EAC decode vs. Ericsson etcdec golden hashes (testdata/etc/etcdec_golden.txt, from
//! ktx-rs `tests/resources/golden/etcunpack.txt`): random blocks, every format, odd sizes.

use imgio::Format;

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
}

fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}

#[test]
fn matches_etcdec() -> Result<(), Box<dyn std::error::Error>> {
    // Golden order: formats x sizes x r16 caps [3, 1, 2, 0] x srgb [false, true].
    let formats = [
        Format::Etc1Rgb,
        Format::Etc2Rgb,
        Format::Etc2Srgb,
        Format::Etc2Rgba,
        Format::Etc2RgbaSrgb,
        Format::Etc2Rgba1,
        Format::Etc2Rgba1Srgb,
        Format::EacR11Unorm,
        Format::EacR11Snorm,
        Format::EacRg11Unorm,
        Format::EacRg11Snorm,
    ];
    let sizes = [(4u32, 4u32), (8, 8), (5, 7), (1, 1), (13, 6), (16, 12)];
    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/etc/etcdec_golden.txt"
    ))?;
    let golden: Vec<&str> = golden.lines().collect();
    let mut checked = 0;
    for (fi, format) in formats.iter().enumerate() {
        for (si, (w, h)) in sizes.iter().enumerate() {
            // Full-capability context: r16 caps 3, sRGB supported.
            let c = ((fi * sizes.len() + si) * 8 + 1) as u32;
            let mut l = Lcg(c.wrapping_mul(7919).wrapping_add(13));
            let src: Vec<u8> = (0..format.surface_bytes(*w, *h)?)
                .map(|_| l.next() as u8)
                .collect();
            let out = imgio::decode_to_buffer(*format, &src, *w, *h)?;
            // etcdec writes RGB without alpha for the opaque formats.
            let rgb_only = matches!(format, Format::Etc1Rgb | Format::Etc2Rgb | Format::Etc2Srgb);
            let hash = if rgb_only {
                fnv(out.chunks_exact(4).flat_map(|p| p[..3].iter().copied()))
            } else {
                fnv(out.iter().copied())
            };
            let want = golden
                .get(c as usize)
                .and_then(|l| l.rsplit(' ').next())
                .unwrap_or("");
            assert_eq!(format!("{hash:016x}"), want, "{format:?} {w}x{h}");
            checked += 1;
        }
    }
    assert_eq!(checked, 66);
    Ok(())
}

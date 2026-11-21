//! JPEG 2000 decoder vs OpenJPEG (testdata/file/jpeg2000, `scripts/gen_file_testdata_jxlj2kvdb.py`):
//! files written by OpenJPEG (imagecodecs, Pillow, `opj_compress`), expected pixels from
//! OpenJPEG's decode or the lossless source. `manifest.txt`: `file format width height reduce
//! max_diff` (max_diff in output units; 0 = exact).

#[cfg(not(feature = "jpeg2000"))]
#[test]
fn without_feature_unsupported() {
    let data = [0xFF, 0x4F, 0xFF, 0x51, 0, 0];
    assert_eq!(
        imgio::file::jpeg2000::decode(&data).err(),
        Some(imgio::Error::Unsupported)
    );
}

#[cfg(feature = "jpeg2000")]
mod with_feature {
    use imgio::Error;
    use imgio::file::jpeg2000::{self, DecodeOptions};
    use imgio::file::{self, FileFormat};
    use std::error::Error as StdError;

    type R = Result<(), Box<dyn StdError>>;

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/file/jpeg2000/");

    struct Case {
        file: String,
        format: String,
        width: u32,
        height: u32,
        reduce: u8,
        max_diff: u32,
    }

    fn manifest() -> Result<Vec<Case>, Box<dyn StdError>> {
        let text = std::fs::read_to_string(format!("{DIR}manifest.txt"))?;
        let mut cases = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 6 {
                return Err(format!("bad manifest line {line}").into());
            }
            cases.push(Case {
                file: f[0].to_string(),
                format: f[1].to_string(),
                width: f[2].parse()?,
                height: f[3].parse()?,
                reduce: f[4].parse()?,
                max_diff: f[5].parse()?,
            });
        }
        Ok(cases)
    }

    /// Per-sample absolute differences (16-bit little endian samples when `wide`).
    fn diffs(a: &[u8], b: &[u8], wide: bool) -> Vec<u32> {
        if wide {
            let (a, b) = (a.as_chunks::<2>().0, b.as_chunks::<2>().0);
            a.iter()
                .zip(b)
                .map(|(x, y)| u32::from(u16::from_le_bytes(*x).abs_diff(u16::from_le_bytes(*y))))
                .collect()
        } else {
            a.iter()
                .zip(b)
                .map(|(x, y)| u32::from(x.abs_diff(*y)))
                .collect()
        }
    }

    #[test]
    fn decodes_match_openjpeg() -> R {
        let cases = manifest()?;
        assert!(cases.len() >= 20);
        for c in &cases {
            let data = std::fs::read(format!("{DIR}{}", c.file))?;
            let suffix = if c.reduce == 0 {
                String::new()
            } else {
                format!(".r{}", c.reduce)
            };
            let expected = std::fs::read(format!("{DIR}{}{suffix}.raw", c.file))?;
            let opts = DecodeOptions {
                reduce: c.reduce,
                ..DecodeOptions::default()
            };
            let img = jpeg2000::decode_with(&data, &opts)
                .map_err(|e| format!("{} r{}: {e}", c.file, c.reduce))?;
            let name = format!("{} r{}", c.file, c.reduce);
            assert_eq!(format!("{:?}", img.format), c.format, "{name}");
            assert_eq!((img.width, img.height), (c.width, c.height), "{name}");
            assert_eq!(img.data.len(), expected.len(), "{name}");
            let wide = c.format.contains("16");
            let d = diffs(&img.data, &expected, wide)
                .into_iter()
                .max()
                .unwrap_or(0);
            assert!(d <= c.max_diff, "{name}: max diff {d} > {}", c.max_diff);
            if c.max_diff == 0 {
                assert!(img.data[..] == expected[..], "{name}");
            }
        }
        Ok(())
    }

    #[test]
    fn lossy_psnr() -> R {
        // The 9/7 and colour-converted files also stay close on average (not only the max).
        for c in manifest()?.iter().filter(|c| c.max_diff > 0) {
            let data = std::fs::read(format!("{DIR}{}", c.file))?;
            let suffix = if c.reduce == 0 {
                String::new()
            } else {
                format!(".r{}", c.reduce)
            };
            let expected = std::fs::read(format!("{DIR}{}{suffix}.raw", c.file))?;
            let opts = DecodeOptions {
                reduce: c.reduce,
                ..DecodeOptions::default()
            };
            let img = jpeg2000::decode_with(&data, &opts)?;
            let d = diffs(&img.data, &expected, c.format.contains("16"));
            let (sum, n) = (d.iter().map(|&v| u64::from(v)).sum::<u64>(), d.len());
            // Mean error well below the max bound: only rounding of the float pipelines differs.
            let mean = sum as f64 / n as f64;
            assert!(
                mean <= c.max_diff as f64 / 2.0,
                "{} r{}: mean diff {mean}",
                c.file,
                c.reduce
            );
        }
        Ok(())
    }

    #[test]
    fn info_and_dispatch() -> R {
        for c in manifest()?.iter().filter(|c| c.reduce == 0) {
            let data = std::fs::read(format!("{DIR}{}", c.file))?;
            assert_eq!(
                FileFormat::detect(&data),
                Some(FileFormat::Jpeg2000),
                "{}",
                c.file
            );
            let info = jpeg2000::info(&data)?;
            assert_eq!((info.width, info.height), (c.width, c.height), "{}", c.file);
            assert_eq!(info.codestream, c.file.ends_with(".j2k"), "{}", c.file);
            let img = file::decode(&data)?;
            assert_eq!(format!("{:?}", img.format), c.format, "{}", c.file);
            let anim = file::decode_animation(&data)?;
            assert_eq!(anim.frames.len(), 1);
        }
        let data = std::fs::read(format!("{DIR}rgba8_53.jp2"))?;
        let info = jpeg2000::info(&data)?;
        assert_eq!(
            (info.channels, info.has_alpha, info.bit_depth),
            (4, true, 8)
        );
        let data = std::fs::read(format!("{DIR}gray12_53.jp2"))?;
        let info = jpeg2000::info(&data)?;
        assert_eq!(
            (info.channels, info.has_alpha, info.bit_depth),
            (1, false, 12)
        );
        Ok(())
    }

    #[test]
    fn reduce_clamps_and_limits() -> R {
        let data = std::fs::read(format!("{DIR}rgba8_53.jp2"))?;
        // 3 resolution levels (OpenJPEG limits the requested 4 for 64x48): at most 2 can be
        // skipped (64x48 -> 16x12), as OpenJPEG `reduce` allows.
        let opts = DecodeOptions {
            reduce: 200,
            ..DecodeOptions::default()
        };
        let img = jpeg2000::decode_with(&data, &opts)?;
        assert_eq!((img.width, img.height), (16, 12));
        let opts = DecodeOptions {
            max_pixels: 64 * 48 - 1,
            ..DecodeOptions::default()
        };
        assert_eq!(
            jpeg2000::decode_with(&data, &opts).err(),
            Some(Error::Unsupported)
        );
        // A codestream claiming 65536 x 65536 pixels is refused before allocating.
        let mut big = std::fs::read(format!("{DIR}rgb8_53_nomct.j2k"))?;
        big[8..16].copy_from_slice(&[0, 1, 0, 0, 0, 1, 0, 0]);
        big[24..32].copy_from_slice(&[0, 1, 0, 0, 0, 1, 0, 0]);
        let info = jpeg2000::info(&big)?;
        assert_eq!((info.width, info.height), (65536, 65536));
        assert_eq!(jpeg2000::decode(&big).err(), Some(Error::Unsupported));
        Ok(())
    }

    #[test]
    fn encode_unsupported() -> R {
        let img = file::Image::new(imgio::Format::Rgba8Srgb, 4, 4)?;
        let s = img.surface()?;
        assert_eq!(
            file::encode(FileFormat::Jpeg2000, &s).err(),
            Some(Error::Unsupported)
        );
        assert_eq!(
            jpeg2000::encode(&s, &Default::default()).err(),
            Some(Error::Unsupported)
        );
        Ok(())
    }

    #[test]
    fn corrupt_input_never_panics() -> R {
        let opts = DecodeOptions {
            reduce: 0,
            max_pixels: 1 << 16,
        };
        for c in manifest()?.iter().filter(|c| c.reduce == 0) {
            let data = std::fs::read(format!("{DIR}{}", c.file))?;
            let n = data.len();
            let step = (n / 150).max(1);
            for len in (0..n).step_by(step) {
                let _ = jpeg2000::decode_with(&data[..len], &opts);
                let _ = jpeg2000::info(&data[..len]);
            }
            let mut rng = 0x9E37_79B9_7F4A_7C15u64;
            // `J2K_FUZZ_ITERS` raises the count for a longer local run.
            let iters = std::env::var("J2K_FUZZ_ITERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(200u32);
            for i in 0..iters {
                let mut bad = data.clone();
                // Mostly headers (the first bytes), then anywhere.
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let span = if i % 2 == 0 { n.min(200) } else { n };
                let pos = (rng as usize) % span;
                bad[pos] ^= 1 << (rng >> 60 & 7);
                if i % 3 == 0 {
                    let pos2 = (rng >> 20) as usize % n;
                    bad[pos2] = (rng >> 40) as u8;
                }
                let _ = jpeg2000::decode_with(&bad, &opts);
            }
        }
        for junk in [
            &b""[..],
            &[0xFF, 0x4F, 0xFF, 0x51],
            b"\0\0\0\x0cjP  \r\n\x87\n",
        ] {
            assert!(jpeg2000::decode(junk).is_err());
        }
        Ok(())
    }
}

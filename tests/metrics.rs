//! Quality metrics against numpy / scikit-image (testdata/metrics, made by
//! scripts/gen_metrics_testdata.py).

use imgio::metrics::{self, CompareOptions};
use imgio::{Error, Format, Surface};
use std::error::Error as StdError;
use std::path::PathBuf;

type R<T = ()> = Result<T, Box<dyn StdError>>;

fn read(name: &str) -> R<Vec<u8>> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/metrics")
        .join(name);
    Ok(std::fs::read(&p).map_err(|e| format!("{name}: {e}"))?)
}

fn format(s: &str) -> R<Format> {
    Ok(match s {
        "R8Unorm" => Format::R8Unorm,
        "R16Unorm" => Format::R16Unorm,
        "Rgb8Unorm" => Format::Rgb8Unorm,
        "Rgba8Unorm" => Format::Rgba8Unorm,
        "Rgba16Float" => Format::Rgba16Float,
        _ => return Err(format!("format {s}").into()),
    })
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    let ok = (got - want).abs() <= tol * want.abs().max(1.0);
    assert!(ok, "{what}: got {got:e} want {want:e}");
}

#[test]
fn matches_numpy_and_skimage() -> R {
    let text = String::from_utf8(read("expected.txt")?)?;
    let mut cases = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (name, fmt) = (f[0], format(f[1])?);
        let (w, h): (u32, u32) = (f[2].parse()?, f[3].parse()?);
        let peak: f64 = f[4].parse()?;
        let vals: Vec<f64> = f[5..].iter().map(|v| v.parse()).collect::<Result<_, _>>()?;
        let nch = vals.len() / 4;
        let (a, b) = (read(&format!("{name}.a"))?, read(&format!("{name}.b"))?);
        let (sa, sb) = (Surface::new(fmt, w, h, &a)?, Surface::new(fmt, w, h, &b)?);
        let m = metrics::compare(&sa, &sb, &CompareOptions { peak, ssim: true })?;
        assert_eq!(m.channels, nch, "{name}");
        let mut mse_sum = 0.0;
        for c in 0..nch {
            let got = m.per_channel[c];
            let want = &vals[c * 4..c * 4 + 4];
            let what = |k: &str| format!("{name} ch{c} {k}");
            close(got.mse, want[0], 1e-12, &what("mse"));
            close(got.max_error, want[1], 1e-12, &what("max"));
            close(got.mean_error, want[2], 1e-12, &what("mean"));
            close(got.ssim, want[3], 1e-12, &what("ssim"));
            let psnr = if want[0] == 0.0 {
                f64::INFINITY
            } else {
                10.0 * (peak * peak / want[0]).log10()
            };
            if psnr.is_finite() {
                close(got.psnr, psnr, 1e-10, &what("psnr"));
            } else {
                assert_eq!(got.psnr, f64::INFINITY);
            }
            mse_sum += want[0];
            println!("{name:14} ch{c} psnr {:7.3} ssim {:.6}", got.psnr, got.ssim);
        }
        close(m.all.mse, mse_sum / nch as f64, 1e-12, name);
        let mean_ssim = (0..nch).map(|c| vals[c * 4 + 3]).sum::<f64>() / nch as f64;
        close(m.all.ssim, mean_ssim, 1e-12, name);
        assert!(m.rgb.max_error <= m.all.max_error);
        cases += 1;
    }
    assert_eq!(cases, 8);
    Ok(())
}

#[test]
fn edge_cases() -> R {
    let a = [10u8; 4 * 5 * 5];
    let mut b = a;
    b[0] = 20;
    let sa = Surface::new(Format::Rgba8Unorm, 5, 5, &a)?;
    let sb = Surface::new(Format::Rgba8Unorm, 5, 5, &b)?;
    // Under 11x11: no SSIM, other metrics still there.
    let m = metrics::compare(&sa, &sb, &CompareOptions::default())?;
    assert!(m.per_channel[0].ssim.is_nan());
    assert!(m.per_channel[0].psnr.is_finite());
    assert_eq!(m.per_channel[1].psnr, f64::INFINITY);
    let m = metrics::compare(
        &sa,
        &sb,
        &CompareOptions {
            ssim: false,
            ..Default::default()
        },
    )?;
    assert!(m.all.ssim.is_nan());
    // Different formats of the same size compare by value.
    let rgb = [10u8; 3 * 5 * 5];
    let s3 = Surface::new(Format::Rgb8Unorm, 5, 5, &rgb)?;
    let m = metrics::compare(&s3, &sa, &CompareOptions::default())?;
    assert_eq!((m.channels, m.all.mse), (3, 0.0));
    // Errors.
    let small = Surface::new(Format::Rgba8Unorm, 1, 1, &a[..4])?;
    assert_eq!(
        metrics::compare(&sa, &small, &CompareOptions::default()).err(),
        Some(Error::InvalidArgument)
    );
    let bad = CompareOptions {
        peak: 0.0,
        ssim: true,
    };
    assert_eq!(
        metrics::compare(&sa, &sb, &bad).err(),
        Some(Error::InvalidArgument)
    );
    let bc = Surface::new(Format::Bc1Unorm, 4, 4, &a[..8])?;
    assert_eq!(
        metrics::compare(&bc, &bc, &CompareOptions::default()).err(),
        Some(Error::Unsupported)
    );
    // Difference image.
    let d = metrics::difference(&sa, &sb, 4.0)?;
    assert_eq!(d.format, Format::Rgba8Unorm);
    assert_eq!(&d.data[..8], &[40, 0, 0, 255, 0, 0, 0, 255]);
    Ok(())
}

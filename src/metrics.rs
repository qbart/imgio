//! Image quality metrics: MSE, PSNR, max / mean absolute error and SSIM.
//!
//! [`compare`] measures two surfaces of the same size (formats may differ) on the stored
//! channel values (sRGB is not linearized, normalized like [`crate::convert`]: unorm 0..1,
//! snorm -1..1, floats as is), in f64:
//!
//! - MSE, PSNR = `10 log10(peak^2 / MSE)` (infinite for identical channels), max and mean
//!   absolute error, per channel and aggregated over RGB and over every channel of the first
//!   image's format (the aggregated MSE is the channel mean, as numpy over all samples).
//! - SSIM (Wang et al. 2004) as scikit-image `structural_similarity(data_range=peak,
//!   gaussian_weights=True, sigma=1.5, use_sample_covariance=False)`: 11x11 Gaussian window
//!   (truncate 3.5), mirrored edges (scipy `reflect`), K1 0.01, K2 0.03, mean over the image
//!   without a 5 pixel border; needs images of at least 11x11 (NaN otherwise). Aggregates are
//!   channel means (`channel_axis`).
//!
//! [`difference`] renders `|a - b| * amplify` as an RGBA8 image (alpha 255).
//!
//! [`Thresholds`] turns metrics into pass / fail (minimum PSNR / SSIM, maximum max / mean error
//! / MSE, per channel or on an aggregate), e.g. for image regression tests in CI.
//!
//! SSIM channels run in parallel.

use crate::file::Image;
use crate::resize::channels;
use crate::{Buffer, Error, Format, Result, Surface};
use vas::Vector;
use vas::storage::Dynamic;

/// Metrics of one channel (or an aggregate).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ChannelMetrics {
    pub mse: f64,
    /// dB; `f64::INFINITY` when MSE is 0.
    pub psnr: f64,
    pub max_error: f64,
    pub mean_error: f64,
    /// 1 for identical channels; NaN when SSIM was not computed or the image is under 11x11.
    pub ssim: f64,
}

/// Result of [`compare`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// Channels compared (of the first image's format): 1..=4.
    pub channels: usize,
    /// R, G, B, A; entries past `channels` are zero (SSIM NaN).
    pub per_channel: [ChannelMetrics; 4],
    /// Over R, G and B (or the channels there are, if fewer).
    pub rgb: ChannelMetrics,
    /// Over every compared channel.
    pub all: ChannelMetrics,
}

/// Options of [`compare`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompareOptions {
    /// Value range (PSNR peak and SSIM `data_range`): 1 for unorm, 2 for snorm, the expected
    /// maximum for HDR.
    pub peak: f64,
    /// Compute SSIM (the expensive part).
    pub ssim: bool,
}

impl Default for CompareOptions {
    fn default() -> Self {
        CompareOptions {
            peak: 1.0,
            ssim: true,
        }
    }
}

fn psnr(mse: f64, peak: f64) -> f64 {
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mse).log10()
    }
}

fn f32_at(b: &[u8], i: usize) -> f32 {
    let k = i * 4;
    f32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]])
}

fn rgba(s: &Surface) -> Result<Buffer> {
    if s.format().is_compressed() {
        return Err(Error::Unsupported);
    }
    crate::convert::convert_to_buffer(s, Format::Rgba32Float)
}

fn zeros(n: usize) -> Result<Vector<f64, Dynamic>> {
    let mut v = Vector::<f64, Dynamic>::default();
    v.resize(n, 0.0).map_err(|_| Error::OutOfMemory)?;
    Ok(v)
}

/// scipy `gaussian_filter` weights (sigma 1.5, truncate 3.5: radius 5).
fn gaussian_kernel() -> [f64; 11] {
    let sigma = 1.5f64;
    let mut k: [f64; 11] = core::array::from_fn(|i| {
        let x = i as f64 - 5.0;
        (-0.5 / (sigma * sigma) * x * x).exp()
    });
    let sum: f64 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// scipy `reflect` index (`d c b a | a b c d | d c b a`).
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n {
            i = 2 * n - i - 1;
        } else {
            return i as usize;
        }
    }
}

/// Separable Gaussian blur of `src` (`w` x `h`): axis 0 (rows) first, then axis 1, like
/// scipy.
fn blur(src: &[f64], w: usize, h: usize, k: &[f64; 11]) -> Result<Vector<f64, Dynamic>> {
    let mut tmp = zeros(w * h)?;
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for (j, kv) in k.iter().enumerate() {
                s += kv * src[reflect(y as isize + j as isize - 5, h) * w + x];
            }
            tmp[y * w + x] = s;
        }
    }
    let mut out = zeros(w * h)?;
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for (j, kv) in k.iter().enumerate() {
                s += kv * tmp[y * w + reflect(x as isize + j as isize - 5, w)];
            }
            out[y * w + x] = s;
        }
    }
    Ok(out)
}

/// SSIM of channel planes `x`, `y` (`w` x `h`, both at least 11).
fn ssim_plane(x: &[f64], y: &[f64], w: usize, h: usize, peak: f64) -> Result<f64> {
    let k = gaussian_kernel();
    let n = w * h;
    let mut xx = zeros(n)?;
    let mut yy = zeros(n)?;
    let mut xy = zeros(n)?;
    for i in 0..n {
        xx[i] = x[i] * x[i];
        yy[i] = y[i] * y[i];
        xy[i] = x[i] * y[i];
    }
    let ux = blur(x, w, h, &k)?;
    let uy = blur(y, w, h, &k)?;
    let uxx = blur(&xx, w, h, &k)?;
    let uyy = blur(&yy, w, h, &k)?;
    let uxy = blur(&xy, w, h, &k)?;
    let c1 = (0.01 * peak) * (0.01 * peak);
    let c2 = (0.03 * peak) * (0.03 * peak);
    let mut sum = 0.0;
    for yi in 5..h - 5 {
        for xi in 5..w - 5 {
            let i = yi * w + xi;
            let vx = uxx[i] - ux[i] * ux[i];
            let vy = uyy[i] - uy[i] * uy[i];
            let vxy = uxy[i] - ux[i] * uy[i];
            let a1 = 2.0 * ux[i] * uy[i] + c1;
            let a2 = 2.0 * vxy + c2;
            let b1 = ux[i] * ux[i] + uy[i] * uy[i] + c1;
            let b2 = vx + vy + c2;
            sum += a1 * a2 / (b1 * b2);
        }
    }
    Ok(sum / ((w - 10) * (h - 10)) as f64)
}

struct Job {
    x: Vector<f64, Dynamic>,
    y: Vector<f64, Dynamic>,
    ssim: f64,
}

fn aggregate(ch: &[ChannelMetrics], peak: f64) -> ChannelMetrics {
    let n = ch.len().max(1) as f64;
    let mse = ch.iter().map(|c| c.mse).sum::<f64>() / n;
    ChannelMetrics {
        mse,
        psnr: psnr(mse, peak),
        max_error: ch.iter().map(|c| c.max_error).fold(0.0, f64::max),
        mean_error: ch.iter().map(|c| c.mean_error).sum::<f64>() / n,
        ssim: ch.iter().map(|c| c.ssim).sum::<f64>() / n,
    }
}

/// Compares `a` (reference) and `b` (same size, any uncompressed formats).
///
/// Errors: `InvalidArgument` for different sizes or a non-positive peak, `Unsupported` for
/// block-compressed formats (decode them first).
pub fn compare(a: &Surface, b: &Surface, opts: &CompareOptions) -> Result<Metrics> {
    if (a.width(), a.height()) != (b.width(), b.height()) || opts.peak.is_nan() || opts.peak <= 0.0
    {
        return Err(Error::InvalidArgument);
    }
    let nch = channels(a.format()).ok_or(Error::Unsupported)?;
    let (fa, fb) = (rgba(a)?, rgba(b)?);
    let (w, h) = (a.width() as usize, a.height() as usize);
    let n = w * h;

    let mut per = [ChannelMetrics {
        ssim: f64::NAN,
        ..Default::default()
    }; 4];
    let mut jobs = Vector::<Job, Dynamic>::default();
    jobs.reserve(nch).map_err(|_| Error::OutOfMemory)?;
    for (c, m) in per.iter_mut().enumerate().take(nch) {
        let mut x = zeros(n)?;
        let mut y = zeros(n)?;
        let (mut se, mut ae, mut mx) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            x[i] = f64::from(f32_at(&fa, i * 4 + c));
            y[i] = f64::from(f32_at(&fb, i * 4 + c));
            let d = (x[i] - y[i]).abs();
            se += d * d;
            ae += d;
            mx = mx.max(d);
        }
        m.mse = se / n as f64;
        m.psnr = psnr(m.mse, opts.peak);
        m.max_error = mx;
        m.mean_error = ae / n as f64;
        jobs.push(Job {
            x,
            y,
            ssim: f64::NAN,
        })
        .map_err(|_| Error::OutOfMemory)?;
    }
    if opts.ssim && w >= 11 && h >= 11 {
        let peak = opts.peak;
        crate::parallel::try_for_each_item(&mut jobs, |j| {
            j.ssim = ssim_plane(&j.x, &j.y, w, h, peak)?;
            Ok(())
        })?;
        for (m, j) in per.iter_mut().zip(jobs.iter()) {
            m.ssim = j.ssim;
        }
    }
    Ok(Metrics {
        channels: nch,
        per_channel: per,
        rgb: aggregate(&per[..nch.min(3)], opts.peak),
        all: aggregate(&per[..nch], opts.peak),
    })
}

/// `|a - b| * amplify` per channel as an RGBA8 unorm image (alpha 255; channels the first
/// format lacks are 0).
pub fn difference(a: &Surface, b: &Surface, amplify: f32) -> Result<Image> {
    if (a.width(), a.height()) != (b.width(), b.height()) {
        return Err(Error::InvalidArgument);
    }
    let nch = channels(a.format()).ok_or(Error::Unsupported)?;
    let (fa, fb) = (rgba(a)?, rgba(b)?);
    let mut img = Image::new(Format::Rgba8Unorm, a.width(), a.height())?;
    for (i, px) in img.data.chunks_mut(4).enumerate() {
        for (c, v) in px.iter_mut().enumerate() {
            *v = if c == 3 {
                255
            } else if c < nch {
                let d = (f32_at(&fa, i * 4 + c) - f32_at(&fb, i * 4 + c)).abs() * amplify;
                (d.clamp(0.0, 1.0) * 255.0).round() as u8
            } else {
                0
            };
        }
    }
    Ok(img)
}

/// What [`Thresholds`] are checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scope {
    /// Every channel on its own (the strictest).
    #[default]
    Channels,
    /// The RGB aggregate.
    Rgb,
    /// The all-channel aggregate.
    All,
}

/// Pass / fail limits for [`Metrics`] (`None` = not checked), e.g. for CI image regression.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Thresholds {
    /// PSNR at least this (dB; identical channels are infinite).
    pub min_psnr: Option<f64>,
    /// SSIM at least this (skipped where SSIM was not computed: disabled or under 11x11).
    pub min_ssim: Option<f64>,
    /// Largest absolute difference at most this (0..1 units for unorm data).
    pub max_error: Option<f64>,
    /// Mean absolute difference at most this.
    pub max_mean_error: Option<f64>,
    /// MSE at most this.
    pub max_mse: Option<f64>,
    pub scope: Scope,
}

/// A failed limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Failure {
    /// `psnr`, `ssim`, `max_error`, `mean_error` or `mse`.
    pub metric: &'static str,
    /// `R`, `G`, `B`, `A` (channels of the first format: 1 = `R`, 2 = `R` `G`), `RGB` or `all`.
    pub channel: &'static str,
    pub value: f64,
    pub limit: f64,
}

impl Thresholds {
    /// Whether any limit is set.
    pub fn any(&self) -> bool {
        self.min_psnr.is_some()
            || self.min_ssim.is_some()
            || self.max_error.is_some()
            || self.max_mean_error.is_some()
            || self.max_mse.is_some()
    }

    /// The failed limits (empty = pass). NaN metrics fail every limit set on them except SSIM.
    pub fn check(&self, m: &Metrics) -> Result<Vector<Failure, Dynamic>> {
        let mut out = Vector::<Failure, Dynamic>::default();
        const NAMES: [&str; 4] = ["R", "G", "B", "A"];
        let sets: [(&'static str, &ChannelMetrics); 4];
        let list: &[(&'static str, &ChannelMetrics)] = match self.scope {
            Scope::Channels => {
                sets = core::array::from_fn(|i| (NAMES[i], &m.per_channel[i]));
                &sets[..m.channels.min(4)]
            }
            Scope::Rgb => &[("RGB", &m.rgb)],
            Scope::All => &[("all", &m.all)],
        };
        for &(channel, c) in list {
            let mut fail = |metric: &'static str, value: f64, limit: f64| {
                out.push(Failure {
                    metric,
                    channel,
                    value,
                    limit,
                })
                .map_err(|_| Error::OutOfMemory)
            };
            // NaN fails (`ge` / `le` are false for NaN).
            let ge = |v: f64, l: f64| v.partial_cmp(&l).is_some_and(|o| o.is_ge());
            let le = |v: f64, l: f64| v.partial_cmp(&l).is_some_and(|o| o.is_le());
            if let Some(l) = self.min_psnr
                && !ge(c.psnr, l)
            {
                fail("psnr", c.psnr, l)?;
            }
            if let Some(l) = self.min_ssim
                && !c.ssim.is_nan()
                && c.ssim < l
            {
                fail("ssim", c.ssim, l)?;
            }
            if let Some(l) = self.max_error
                && !le(c.max_error, l)
            {
                fail("max_error", c.max_error, l)?;
            }
            if let Some(l) = self.max_mean_error
                && !le(c.mean_error, l)
            {
                fail("mean_error", c.mean_error, l)?;
            }
            if let Some(l) = self.max_mse
                && !le(c.mse, l)
            {
                fail("mse", c.mse, l)?;
            }
        }
        Ok(out)
    }
}

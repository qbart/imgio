//! Channel packing: build each output channel from any channel of any input image.
//!
//! [`pack`] combines up to four sources into one image, e.g. ORM (occlusion, roughness,
//! metallic), Unity mask maps (metallic, occlusion, detail, smoothness = 1 - roughness), color +
//! alpha from a separate mask, or alpha from the luminance of a gray image. [`split`] does the
//! reverse and returns one single channel image per channel.
//!
//! Values are taken as stored (sRGB inputs are not linearized; packed channels are data), read
//! through an RGBA f32 copy of each input (missing channels read 0, alpha 1), then
//! `v = clamp?(bias + scale * (invert ? 1 - v : v))` and written to the output format (integer
//! formats are clamped and rounded). Inputs with another size than the output are resized
//! first with [`crate::resize`] (linear, no premultiply), so the output is exact for inputs of
//! the output size.

use crate::file::Image;
use crate::resize::{Fit, ResizeOptions, resize};
use crate::{Buffer, Error, Format, Result, Surface};
use vas::Vector;
use vas::storage::Dynamic;

/// Bytes of one RGBA f32 pixel.
const PX: usize = 16;

/// Inputs a [`pack`] call can reference (indices 0..64).
pub const MAX_INPUTS: usize = 64;

/// A channel of an input image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    R,
    G,
    B,
    A,
    /// Rec. 709 luminance of RGB (as stored).
    Luma,
    /// Mean of R, G and B.
    Average,
    /// Largest of R, G and B.
    Max,
}

/// Where one output channel comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Source {
    /// `input`-th image of the [`pack`] inputs, one of its channels.
    Input { input: usize, channel: Channel },
    /// The same value everywhere.
    Constant(f32),
}

/// One output channel: a source and an optional remap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelSpec {
    pub source: Source,
    /// `1 - v` before scale and bias (e.g. smoothness from roughness).
    pub invert: bool,
    pub scale: f32,
    pub bias: f32,
}

impl ChannelSpec {
    /// `channel` of input `input`, unchanged.
    pub fn input(input: usize, channel: Channel) -> Self {
        ChannelSpec {
            source: Source::Input { input, channel },
            invert: false,
            scale: 1.0,
            bias: 0.0,
        }
    }

    /// A constant value.
    pub fn constant(v: f32) -> Self {
        ChannelSpec {
            source: Source::Constant(v),
            invert: false,
            scale: 1.0,
            bias: 0.0,
        }
    }

    /// The same source, inverted.
    pub fn inverted(mut self) -> Self {
        self.invert = !self.invert;
        self
    }

    fn eval(&self, px: &[[f32; 4]]) -> f32 {
        let v = match self.source {
            Source::Constant(v) => v,
            Source::Input { input, channel } => {
                let p = px[input];
                match channel {
                    Channel::R => p[0],
                    Channel::G => p[1],
                    Channel::B => p[2],
                    Channel::A => p[3],
                    Channel::Luma => 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2],
                    Channel::Average => (p[0] + p[1] + p[2]) / 3.0,
                    Channel::Max => p[0].max(p[1]).max(p[2]),
                }
            }
        };
        let v = if self.invert { 1.0 - v } else { v };
        self.bias + self.scale * v
    }
}

/// Output size of [`pack`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PackSize {
    /// Every input must have the same size (`InvalidArgument` otherwise).
    #[default]
    Same,
    /// The largest width and height of the used inputs; smaller inputs are resized.
    Largest,
    /// The size of the first used input; the others are resized.
    First,
    /// A fixed size; inputs are resized.
    Fixed(u32, u32),
}

/// Options of [`pack`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackOptions {
    /// R, G, B, A of the output; `None` writes 0 (RGB) or 1 (alpha).
    pub channels: [Option<ChannelSpec>; 4],
    /// Output format (any uncompressed format).
    pub format: Format,
    pub size: PackSize,
    /// Clamp every channel to 0..1 before writing (float formats keep out-of-range values when
    /// false; integer formats always clamp).
    pub clamp: bool,
}

impl PackOptions {
    /// Every channel `None` (black, opaque), RGBA8 unorm, inputs of the same size.
    pub fn new(format: Format) -> Self {
        PackOptions {
            channels: [None; 4],
            format,
            size: PackSize::Same,
            clamp: true,
        }
    }

    /// Occlusion -> R, roughness -> G, metallic -> B (glTF metallicRoughness + occlusion), each
    /// from the R channel of inputs 0, 1 and 2.
    pub fn orm() -> Self {
        let mut o = PackOptions::new(Format::Rgba8Unorm);
        o.channels = [
            Some(ChannelSpec::input(0, Channel::R)),
            Some(ChannelSpec::input(1, Channel::R)),
            Some(ChannelSpec::input(2, Channel::R)),
            None,
        ];
        o
    }

    /// Unity HDRP mask map from metallic, occlusion, detail mask and roughness (inputs 0..3,
    /// R channel): R metallic, G occlusion, B detail, A smoothness = 1 - roughness.
    pub fn mask_map() -> Self {
        let mut o = PackOptions::new(Format::Rgba8Unorm);
        o.channels = [
            Some(ChannelSpec::input(0, Channel::R)),
            Some(ChannelSpec::input(1, Channel::R)),
            Some(ChannelSpec::input(2, Channel::R)),
            Some(ChannelSpec::input(3, Channel::R).inverted()),
        ];
        o
    }

    /// Sets output channel `i` (0 R .. 3 A).
    pub fn with(mut self, i: usize, spec: ChannelSpec) -> Self {
        if let Some(c) = self.channels.get_mut(i) {
            *c = Some(spec);
        }
        self
    }
}

fn f32_at(b: &[u8], i: usize) -> f32 {
    let k = i * 4;
    f32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]])
}

/// RGBA f32 copy of `src` at `w` x `h` (resized when the size differs).
fn work(src: &Surface, w: u32, h: u32) -> Result<Buffer> {
    if src.format().is_compressed() {
        return Err(Error::Unsupported);
    }
    if (src.width(), src.height()) == (w, h) {
        return crate::convert::convert_to_buffer(src, Format::Rgba32Float);
    }
    let f = crate::convert::convert_to_buffer(src, Format::Rgba32Float)?;
    let s = Surface::new(Format::Rgba32Float, src.width(), src.height(), &f)?;
    let opts = ResizeOptions {
        fit: Fit::Exact,
        srgb: Some(false),
        premultiply: false,
        ..Default::default()
    };
    Ok(resize(&s, w, h, &opts)?.data)
}

/// Builds an image whose channels come from `inputs` as `opts.channels` says.
///
/// Errors: `InvalidArgument` for an input index out of range (or >= [`MAX_INPUTS`]), mismatched sizes with
/// [`PackSize::Same`], or a zero fixed size; `Unsupported` for compressed formats.
pub fn pack(inputs: &[Surface], opts: &PackOptions) -> Result<Image> {
    if opts.format.is_compressed() {
        return Err(Error::Unsupported);
    }
    // Inputs actually used, in first-use order.
    let mut used = [false; MAX_INPUTS];
    let mut first = None;
    for spec in opts.channels.iter().flatten() {
        if let Source::Input { input, .. } = spec.source {
            if input >= inputs.len() || input >= MAX_INPUTS {
                return Err(Error::InvalidArgument);
            }
            used[input] = true;
            first.get_or_insert(input);
        }
    }
    let sizes = inputs
        .iter()
        .enumerate()
        .filter(|(i, _)| used.get(*i).copied().unwrap_or(false))
        .map(|(_, s)| (s.width(), s.height()));
    let (w, h) = match opts.size {
        PackSize::Fixed(w, h) => (w, h),
        PackSize::First => first
            .and_then(|i| inputs.get(i))
            .map(|s| (s.width(), s.height()))
            .unwrap_or((1, 1)),
        PackSize::Largest => sizes
            .fold(None, |acc: Option<(u32, u32)>, (w, h)| {
                Some(acc.map_or((w, h), |(aw, ah)| (aw.max(w), ah.max(h))))
            })
            .unwrap_or((1, 1)),
        PackSize::Same => {
            let mut it = sizes;
            let s0 = it.next().unwrap_or((1, 1));
            if it.any(|s| s != s0) {
                return Err(Error::InvalidArgument);
            }
            s0
        }
    };
    if w == 0 || h == 0 {
        return Err(Error::InvalidArgument);
    }

    let mut works = Vector::<Option<Buffer>, Dynamic>::default();
    works
        .reserve(inputs.len())
        .map_err(|_| Error::OutOfMemory)?;
    for (i, s) in inputs.iter().enumerate().take(MAX_INPUTS) {
        let b = if used.get(i).copied().unwrap_or(false) {
            Some(work(s, w, h)?)
        } else {
            None
        };
        works.push(b).map_err(|_| Error::OutOfMemory)?;
    }

    let n = w as usize * h as usize;
    let mut out = crate::buffer::zeroed(n * PX)?;
    let defaults = [0.0, 0.0, 0.0, 1.0];
    let clamp = opts.clamp;
    let row = w as usize * PX;
    let works = &works[..];
    crate::parallel::for_each_unit(&mut out, row, n, |y, line| {
        let mut px = [[0.0f32; 4]; MAX_INPUTS];
        for x in 0..w as usize {
            let i = y * w as usize + x;
            for (p, b) in px.iter_mut().zip(works) {
                if let Some(b) = b {
                    *p = core::array::from_fn(|c| f32_at(b, i * 4 + c));
                }
            }
            for (c, spec) in opts.channels.iter().enumerate() {
                let v = spec.map_or(defaults[c], |s| s.eval(&px));
                let v = if clamp { v.clamp(0.0, 1.0) } else { v };
                let k = x * PX + c * 4;
                line[k..k + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    });

    let mut img = Image::new(opts.format, w, h)?;
    let s = Surface::new(Format::Rgba32Float, w, h, &out)?;
    crate::convert::convert(&s, opts.format, &mut img.data)?;
    Ok(img)
}

/// One single channel image per channel of `src` (R, G, B, A), each in `format` (an R-only
/// format such as `R8Unorm`, `R16Unorm` or `R32Float`; values written to its R channel).
pub fn split(src: &Surface, format: Format) -> Result<[Image; 4]> {
    let ch = |c: Channel| {
        let mut o = PackOptions::new(format);
        o.channels[0] = Some(ChannelSpec::input(0, c));
        o.clamp = !format.is_hdr();
        pack(core::slice::from_ref(src), &o)
    };
    Ok([
        ch(Channel::R)?,
        ch(Channel::G)?,
        ch(Channel::B)?,
        ch(Channel::A)?,
    ])
}

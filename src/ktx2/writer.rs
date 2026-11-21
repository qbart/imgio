//! KTX2 writer for plain `VkFormat` payloads (KTX 2.0 specification; layout, alignment and
//! key/value ordering as KTX-Software `ktxTexture2_WriteToMemory`).

use super::{dfd::dfd, type_size_of, vk::vk_format};
use crate::{Buffer, Error, Format, Result};

const IDENTIFIER: [u8; 12] = super::IDENTIFIER;

/// Supercompression of the level data.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Supercompression {
    /// Raw levels.
    #[default]
    None,
    /// Zstandard at the given level (1..=22; byte-identical with libzstd / KTX-Software).
    Zstd(i32),
}

/// Shape of a texture to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ktx2Desc {
    pub format: Format,
    pub width: u32,
    pub height: u32,
    /// Array layers; 0 for a non-array texture.
    pub layers: u32,
    /// 1, or 6 for a cubemap (square).
    pub faces: u32,
    /// Mip levels (at least 1).
    pub levels: u32,
}

/// Options of [`write()`].
#[derive(Debug, Clone, Copy, Default)]
pub struct WriteOptions<'a> {
    pub supercompression: Supercompression,
    /// Key/value pairs, written sorted by key. Values are raw bytes: string values such as
    /// `KTXwriter` conventionally include their NUL terminator.
    pub key_values: &'a [(&'a str, &'a [u8])],
}

fn put(out: &mut [u8], at: usize, v: &[u8]) {
    out[at..at + v.len()].copy_from_slice(v);
}

fn lcm4(n: usize) -> usize {
    let g = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    n / g(n, 4) * 4
}

/// Writes a KTX2 file. `levels[i]` holds mip level `i`: `max(layers, 1) * faces` images of
/// `format.surface_bytes(level size)` bytes each, in layer then face order.
///
/// Errors: `Unsupported` (format without a `VkFormat`), `InvalidArgument` (shape, level count,
/// data sizes, empty / NUL-containing / duplicate keys, more than 64 keys), `OutOfMemory`.
pub fn write(desc: &Ktx2Desc, levels: &[&[u8]], opts: &WriteOptions) -> Result<Buffer> {
    let f = desc.format;
    let vk = vk_format(f).ok_or(Error::Unsupported)?;
    let dfd = dfd(f).ok_or(Error::Unsupported)?;
    let (w, h) = (desc.width, desc.height);
    let n = levels.len();
    if w == 0
        || h == 0
        || n == 0
        || n != desc.levels as usize
        || desc.levels > crate::image::mip_levels(w, h)
        || !(desc.faces == 1 || (desc.faces == 6 && w == h))
    {
        return Err(Error::InvalidArgument);
    }
    let images = desc.layers.max(1) as usize * desc.faces as usize;
    for (i, l) in levels.iter().enumerate() {
        let (lw, lh) = crate::image::mip_dim(w, h, i as u32);
        let size = f.surface_bytes(lw, lh)?.checked_mul(images);
        if size != Some(l.len()) {
            return Err(Error::InvalidArgument);
        }
    }

    // Supercompressed levels (kept in a vas vector, smallest last like the input).
    let mut packed = vas::Vector::<Buffer, vas::storage::Dynamic>::default();
    if let Supercompression::Zstd(level) = opts.supercompression {
        packed.reserve(n).map_err(|_| Error::OutOfMemory)?;
        for l in levels {
            let mut buf =
                crate::buffer::zeroed(zstd_pure_rs::prelude::ZSTD_compressBound(l.len()))?;
            let c = zstd_pure_rs::prelude::ZSTD_compress(&mut buf, l, level);
            if zstd_pure_rs::prelude::ZSTD_isError(c) {
                return Err(Error::InvalidArgument);
            }
            buf.truncate(c);
            packed.push(buf).map_err(|_| Error::OutOfMemory)?;
        }
    }
    let stored = |i: usize| -> &[u8] { packed.get(i).map_or(levels[i], |b| &b[..]) };

    // Key/value data, sorted by key.
    let mut order = [0usize; 64];
    let kvs = opts.key_values;
    if kvs.len() > order.len() {
        return Err(Error::InvalidArgument);
    }
    for (i, o) in order.iter_mut().enumerate().take(kvs.len()) {
        *o = i;
    }
    let order = &mut order[..kvs.len()];
    order.sort_unstable_by(|&a, &b| kvs[a].0.as_bytes().cmp(kvs[b].0.as_bytes()));
    let bad_key = |k: &str| k.is_empty() || k.as_bytes().contains(&0);
    if kvs.iter().any(|(k, _)| bad_key(k)) || order.windows(2).any(|p| kvs[p[0]].0 == kvs[p[1]].0) {
        return Err(Error::InvalidArgument);
    }
    let kvd_len: usize = kvs
        .iter()
        .map(|(k, v)| (4 + k.len() + 1 + v.len()).next_multiple_of(4))
        .sum();

    let dfd_ofs = 80 + 24 * n;
    let dfd_len = dfd.words().len() * 4;
    let kvd_ofs = dfd_ofs + dfd_len;
    let align = if packed.is_empty() {
        lcm4(f.block_bytes())
    } else {
        1
    };
    // Level offsets, smallest level first in the file.
    let mut offsets = [0usize; 32];
    let mut end = kvd_ofs + kvd_len;
    for i in (0..n).rev() {
        end = end.next_multiple_of(align);
        offsets[i] = end;
        end += stored(i).len();
    }

    let mut out = crate::buffer::zeroed(end)?;
    put(&mut out, 0, &IDENTIFIER);
    let scheme = if packed.is_empty() { 0u32 } else { 2 };
    let header = [
        vk,
        type_size_of(f),
        w,
        h,
        0,
        desc.layers,
        desc.faces,
        desc.levels,
        scheme,
        dfd_ofs as u32,
        dfd_len as u32,
        if kvd_len > 0 { kvd_ofs as u32 } else { 0 },
        kvd_len as u32,
    ];
    for (i, v) in header.iter().enumerate() {
        put(&mut out, 12 + 4 * i, &v.to_le_bytes());
    }
    // sgdByteOffset / sgdByteLength stay 0.
    for i in 0..n {
        let at = 80 + 24 * i;
        put(&mut out, at, &(offsets[i] as u64).to_le_bytes());
        put(&mut out, at + 8, &(stored(i).len() as u64).to_le_bytes());
        put(&mut out, at + 16, &(levels[i].len() as u64).to_le_bytes());
    }
    for (i, word) in dfd.words().iter().enumerate() {
        put(&mut out, dfd_ofs + 4 * i, &word.to_le_bytes());
    }
    let mut at = kvd_ofs;
    for &i in order.iter() {
        let (k, v) = kvs[i];
        let len = k.len() + 1 + v.len();
        put(&mut out, at, &(len as u32).to_le_bytes());
        put(&mut out, at + 4, k.as_bytes());
        put(&mut out, at + 4 + k.len() + 1, v);
        at += (4 + len).next_multiple_of(4);
    }
    for (i, &ofs) in offsets[..n].iter().enumerate() {
        put(&mut out, ofs, stored(i));
    }
    Ok(out)
}

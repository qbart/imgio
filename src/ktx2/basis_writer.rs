// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal (encoder/basisu_comp.cpp `create_ktx2_file`,
// `get_dfd`) and modified; see NOTICE.md.
//! KTX2 files from `.basis` files: the layout of basisu's own KTX2 writer, fed from a parsed
//! `.basis` (any payload: ETC1S, UASTC LDR / HDR 4x4, ASTC HDR 6x6, UASTC HDR 6x6i, ASTC LDR,
//! XUASTC LDR, XUBC7).

use crate::basis::{BasisFile, BasisTexFormat, BasisTexType};
use crate::{Buffer, Error, Result};

/// Options of [`write_basis`].
#[derive(Debug, Clone, Copy, Default)]
pub struct BasisKtx2Options<'a> {
    /// Zstandard level for the payloads basisu allows it on (UASTC LDR 4x4, UASTC HDR 4x4,
    /// ASTC HDR 6x6, ASTC LDR); `None` = no supercompression. Ignored for the others.
    pub zstd: Option<i32>,
    /// Key/value pairs, written sorted by key (values raw; strings include their NUL). basisu
    /// writes `KTXwriter` ("Basis Universal 2.50") and, for HDR, `KTXmapRange`.
    pub key_values: &'a [(&'a str, &'a [u8])],
}

const SS_NONE: u32 = 0;
const SS_BASISLZ: u32 = 1;
const SS_ZSTD: u32 = 2;
const SS_UASTC_HDR_6X6I: u32 = 4;
const SS_XUASTC_LDR: u32 = 5;
const SS_XUBC7: u32 = 6;
const SLICE_FLAG_HAS_ALPHA: u32 = 1;
const SLICE_FLAG_I_FRAME: u32 = 2;
const IMAGE_IS_P_FRAME: u32 = 2;

/// basisu's DFD templates (`g_ktx2_*_dfd`): model, dims-1, bytes plane 0, sample word 0,
/// lower, upper; the ETC1S alpha template has a second sample.
fn dfd_template(f: BasisTexFormat, alpha: bool) -> ([u32; 15], usize) {
    let mut d = [0u32; 15];
    let one = |model: u32, dim: u32, word0: u32, lower: u32, upper: u32| {
        let mut d = [0u32; 15];
        d[0] = 44;
        d[2] = 2 | (40 << 16);
        d[3] = model | (1 << 8) | (1 << 16);
        d[4] = dim | (dim << 8);
        d[5] = 16;
        d[7] = word0;
        d[9] = lower;
        d[10] = upper;
        d
    };
    match f {
        BasisTexFormat::Etc1s => {
            d[0] = if alpha { 60 } else { 44 };
            d[2] = 2 | ((if alpha { 56 } else { 40 }) << 16);
            d[3] = 163 | (1 << 8) | (2 << 16);
            d[4] = 3 | (3 << 8);
            d[5] = if alpha { 8 | (8 << 8) } else { 8 };
            d[7] = 63 << 16;
            d[10] = u32::MAX;
            if alpha {
                d[11] = 64 | (63 << 16) | (15 << 24);
                d[14] = u32::MAX;
            }
            (d, if alpha { 15 } else { 11 })
        }
        BasisTexFormat::UastcLdr4x4 => {
            let mut d = one(166, 3, (127 << 16) | (4 << 24), 0, u32::MAX);
            d[3] = 166 | (1 << 8) | (2 << 16);
            (d, 11)
        }
        BasisTexFormat::UastcHdr4x4 => {
            (one(167, 3, (127 << 16) | (0x80 << 24), 0, 0x3F80_0000), 11)
        }
        BasisTexFormat::AstcHdr6x6 => (
            one(162, 5, (127 << 16) | (0xC0 << 24), 0xBF80_0000, 0x3F80_0000),
            11,
        ),
        BasisTexFormat::UastcHdr6x6Intermediate => {
            (one(168, 5, (127 << 16) | (0x80 << 24), 0, 0x3F80_0000), 11)
        }
        BasisTexFormat::XuastcLdr(_) => (one(169, 3, 127 << 16, 0, u32::MAX), 11),
        BasisTexFormat::AstcLdr(_) => (one(162, 3, 127 << 16, 0, u32::MAX), 11),
        BasisTexFormat::Xubc7 => (one(170, 3, 127 << 16, 0, u32::MAX), 11),
    }
}

/// basisu `get_dfd`: the template with transfer function, UASTC channel and ASTC block size.
fn dfd(file: &BasisFile) -> Vec<u8> {
    let f = file.tex_format();
    let alpha = file.has_alpha();
    let (mut d, words) = dfd_template(f, alpha);
    // Transfer: linear for HDR, else the file's sRGB flag.
    let srgb = !f.is_hdr() && file.is_srgb();
    d[3] = (d[3] & !(0xFF << 16)) | ((if srgb { 2 } else { 1 }) << 16);
    let uastc = !matches!(f, BasisTexFormat::Etc1s);
    if uastc {
        let rgba = alpha
            && matches!(
                f,
                BasisTexFormat::UastcLdr4x4
                    | BasisTexFormat::XuastcLdr(_)
                    | BasisTexFormat::AstcLdr(_)
                    | BasisTexFormat::Xubc7
            );
        d[7] = (d[7] & !(0xF << 24)) | ((if rgba { 3 } else { 0 }) << 24);
    }
    if let BasisTexFormat::XuastcLdr(_) | BasisTexFormat::AstcLdr(_) = f {
        let (bw, bh) = f.block_dim();
        d[4] = (d[4] & !0xFFFF) | (bw - 1) | ((bh - 1) << 8);
    }
    d[..words].iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Writes a KTX2 file holding the payload of `basis` (a `.basis` file), laid out like
/// basisu's `-ktx2` output: images are the `.basis` images (2D arrays: layers; cubemap
/// arrays: faces of cubes; video: frames as layers), every level of image 0's chain.
///
/// Errors: `InvalidData` (unparsable or inconsistent `.basis`), `Unsupported` (volume
/// textures), `InvalidArgument` (bad keys), `OutOfMemory`.
pub fn write_basis(basis: &[u8], opts: &BasisKtx2Options) -> Result<Buffer> {
    let file = BasisFile::parse(basis)?;
    let f = file.tex_format();
    if file.tex_type() == BasisTexType::Volume {
        return Err(Error::Unsupported);
    }
    let slices = &file.slices;
    let base = slices
        .iter()
        .find(|s| s.level_index == 0)
        .ok_or(Error::InvalidData)?;
    let (width, height) = (base.orig_width, base.orig_height);
    let mut total_layers = 0u32;
    let mut total_levels = 0u32;
    for s in slices.iter() {
        total_layers = total_layers.max(s.image_index + 1);
        if s.image_index == 0 {
            total_levels = total_levels.max(s.level_index + 1);
        }
    }
    let mut total_faces = 1;
    if file.tex_type() == BasisTexType::CubemapArray {
        total_layers /= 6;
        total_faces = 6;
    }
    if total_levels == 0 || total_levels > 32 || total_layers == 0 {
        return Err(Error::InvalidData);
    }
    let can_zstd = matches!(
        f,
        BasisTexFormat::UastcLdr4x4
            | BasisTexFormat::UastcHdr4x4
            | BasisTexFormat::AstcHdr6x6
            | BasisTexFormat::AstcLdr(_)
    );
    let vk = match f {
        BasisTexFormat::UastcHdr4x4 => 1_000_066_000,
        BasisTexFormat::AstcHdr6x6 => 1_000_066_004,
        BasisTexFormat::AstcLdr(b) => {
            let i = crate::AstcBlock::ALL
                .iter()
                .position(|&x| x == b)
                .ok_or(Error::InvalidData)? as u32;
            157 + 2 * i + u32::from(file.is_srgb())
        }
        _ => 0,
    };
    let zstd = opts.zstd.filter(|_| can_zstd);
    let mut scheme = if zstd.is_some() { SS_ZSTD } else { SS_NONE };

    // Level data: slices appended in file order to their level.
    let levels = total_levels as usize;
    let mut level_data: Vec<Vec<u8>> = vec![Vec::new(); levels];
    let mut slice_ofs = vec![0usize; slices.len()];
    let slice_bytes = |i: usize| -> Result<&[u8]> {
        let s = &slices[i];
        basis
            .get(s.file_ofs as usize..(s.file_ofs as usize).saturating_add(s.file_size as usize))
            .ok_or(Error::InvalidData)
    };
    for (i, s) in slices.iter().enumerate() {
        let l = level_data
            .get_mut(s.level_index as usize)
            .ok_or(Error::InvalidData)?;
        slice_ofs[i] = l.len();
        l.try_reserve(s.file_size as usize)
            .map_err(|_| Error::OutOfMemory)?;
        l.extend_from_slice(slice_bytes(i)?);
    }
    let stored: Vec<Vec<u8>> = match zstd {
        Some(level) => {
            let mut out = Vec::new();
            for l in &level_data {
                let mut buf = vec![0u8; zstd_pure_rs::prelude::ZSTD_compressBound(l.len())];
                let n = zstd_pure_rs::prelude::ZSTD_compress(&mut buf, l, level);
                if zstd_pure_rs::prelude::ZSTD_isError(n) {
                    return Err(Error::InvalidArgument);
                }
                buf.truncate(n);
                out.push(buf);
            }
            out
        }
        None => level_data.clone(),
    };

    // Supercompression global data.
    let images = levels * total_layers as usize * total_faces as usize;
    let image_index = |s: &crate::basis::file::SliceDesc| {
        let (mut layer, mut face) = (s.image_index, 0);
        if total_faces == 6 {
            face = layer % 6;
            layer /= 6;
        }
        s.level_index as usize * (total_layers * total_faces) as usize
            + (layer * total_faces + face) as usize
    };
    let mut sgd = Vec::new();
    let put = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
    match f {
        BasisTexFormat::Etc1s => {
            let mut descs = vec![[0u32; 5]; images];
            for (i, s) in slices.iter().enumerate() {
                let d = descs.get_mut(image_index(s)).ok_or(Error::InvalidData)?;
                if s.flags & SLICE_FLAG_HAS_ALPHA != 0 {
                    d[3] = slice_ofs[i] as u32;
                    d[4] = s.file_size;
                } else {
                    if file.tex_type() == BasisTexType::VideoFrames
                        && s.flags & SLICE_FLAG_I_FRAME == 0
                    {
                        d[0] = IMAGE_IS_P_FRAME;
                    }
                    d[1] = slice_ofs[i] as u32;
                    d[2] = s.file_size;
                }
            }
            let sec = |s: crate::basis::file::Section| {
                basis
                    .get(s.ofs as usize..(s.ofs as usize).saturating_add(s.size as usize))
                    .ok_or(Error::InvalidData)
            };
            sgd.extend_from_slice(&(file.total_endpoints as u16).to_le_bytes());
            sgd.extend_from_slice(&(file.total_selectors as u16).to_le_bytes());
            put(&mut sgd, file.endpoint_cb.size);
            put(&mut sgd, file.selector_cb.size);
            put(&mut sgd, file.tables.size);
            put(&mut sgd, 0);
            for d in &descs {
                for &x in d {
                    put(&mut sgd, x);
                }
            }
            sgd.extend_from_slice(sec(file.endpoint_cb)?);
            sgd.extend_from_slice(sec(file.selector_cb)?);
            sgd.extend_from_slice(sec(file.tables)?);
            scheme = SS_BASISLZ;
        }
        BasisTexFormat::UastcHdr6x6Intermediate
        | BasisTexFormat::XuastcLdr(_)
        | BasisTexFormat::Xubc7 => {
            let mut descs = vec![[0u32; 3]; images];
            for (i, s) in slices.iter().enumerate() {
                let d = descs.get_mut(image_index(s)).ok_or(Error::InvalidData)?;
                let bytes = slice_bytes(i)?;
                d[0] = slice_ofs[i] as u32;
                d[1] = s.file_size;
                d[2] = match f {
                    BasisTexFormat::UastcHdr6x6Intermediate if bytes.len() >= 2 => {
                        u32::from(bytes[0]) | (u32::from(bytes[1]) << 8)
                    }
                    BasisTexFormat::UastcHdr6x6Intermediate => 0,
                    _ => bytes.first().map_or(0, |&b| u32::from(b) | (1 << 8)),
                };
            }
            for d in &descs {
                for &x in d {
                    put(&mut sgd, x);
                }
            }
            scheme = match f {
                BasisTexFormat::UastcHdr6x6Intermediate => SS_UASTC_HDR_6X6I,
                BasisTexFormat::Xubc7 => SS_XUBC7,
                _ => SS_XUASTC_LDR,
            };
        }
        _ => {}
    }

    // Key/value data (sorted), plus basisu's dummy key that 16-byte aligns the end of the KVD
    // when the levels are not supercompressed.
    let dfd = dfd(&file);
    let kvd_ofs = 80 + 24 * levels + dfd.len();
    let mut keys: Vec<(Vec<u8>, &[u8])> = Vec::new();
    for (k, v) in opts.key_values {
        if k.is_empty() || k.as_bytes().contains(&0) {
            return Err(Error::InvalidArgument);
        }
        let mut key = k.as_bytes().to_vec();
        key.push(0);
        keys.push((key, v));
    }
    let build = |keys: &mut Vec<(Vec<u8>, &[u8])>| -> Vec<u8> {
        keys.sort_by(|a, b| a.0.cmp(&b.0));
        let mut kvd = Vec::new();
        for (k, v) in keys.iter() {
            kvd.extend_from_slice(&((k.len() + v.len()) as u32).to_le_bytes());
            kvd.extend_from_slice(k);
            kvd.extend_from_slice(v);
            kvd.resize(kvd.len().next_multiple_of(4), 0);
        }
        kvd
    };
    let mut kvd = build(&mut keys);
    if scheme == SS_NONE {
        let pad = (16 - ((kvd_ofs + kvd.len()) & 15)) & 15;
        if pad != 0 {
            let pad = if pad < 6 { pad + 16 } else { pad };
            let mut key = vec![127u8; pad - 6];
            key.push(0);
            keys.push((key, &[0]));
            kvd = build(&mut keys);
        }
    }

    let mut out: Vec<u8> = vec![0; 80 + 24 * levels];
    out.extend_from_slice(&dfd);
    let (kvd_o, kvd_l) = if kvd.is_empty() {
        (0, 0)
    } else {
        (out.len() as u32, kvd.len() as u32)
    };
    out.extend_from_slice(&kvd);
    let (mut sgd_o, sgd_l) = (0u64, sgd.len() as u64);
    if !sgd.is_empty() {
        out.resize(out.len().next_multiple_of(8), 0);
        sgd_o = out.len() as u64;
        out.extend_from_slice(&sgd);
    }
    if scheme == SS_NONE {
        out.resize(out.len().next_multiple_of(16), 0);
    }
    let mut index = vec![(0u64, 0u64, 0u64); levels];
    for l in (0..levels).rev() {
        let uncompressed = if can_zstd {
            level_data[l].len() as u64
        } else {
            0
        };
        index[l] = (out.len() as u64, stored[l].len() as u64, uncompressed);
        out.extend_from_slice(&stored[l]);
    }
    // Header.
    out[..12].copy_from_slice(&super::IDENTIFIER);
    let words = [
        vk,
        1,
        width,
        height,
        0,
        if total_layers > 1 { total_layers } else { 0 },
        total_faces,
        total_levels,
        scheme,
        (80 + 24 * levels) as u32,
        dfd.len() as u32,
        kvd_o,
        kvd_l,
    ];
    for (i, w) in words.iter().enumerate() {
        out[12 + 4 * i..16 + 4 * i].copy_from_slice(&w.to_le_bytes());
    }
    out[64..72].copy_from_slice(&sgd_o.to_le_bytes());
    out[72..80].copy_from_slice(&sgd_l.to_le_bytes());
    for (l, (o, len, u)) in index.iter().enumerate() {
        let at = 80 + 24 * l;
        out[at..at + 8].copy_from_slice(&o.to_le_bytes());
        out[at + 8..at + 16].copy_from_slice(&len.to_le_bytes());
        out[at + 16..at + 24].copy_from_slice(&u.to_le_bytes());
    }
    let mut buf = crate::buffer::zeroed(out.len())?;
    buf.copy_from_slice(&out);
    Ok(buf)
}

/// Encodes `images` with [`crate::basis::encode_basis`] and writes the result as KTX2
/// ([`write_basis`]), adding `KTXwriter` = "imgio `<version>`" unless `ktx2.key_values` has one,
/// and the encoder's `DeblockFilterID` (ASTC LDR / XUASTC LDR deblocking, as basisu's
/// `m_key_values`) unless `ktx2.key_values` has one.
pub fn encode_basis(
    images: &[crate::Surface],
    opts: &crate::basis::BasisEncodeOptions,
    ktx2: &BasisKtx2Options,
) -> Result<Buffer> {
    use crate::basis::ll::basis_file::{
        BASISU_DEBLOCK_FILTER_ID_NAME, basis_file_key_values, find_key_value,
    };
    let basis = crate::basis::encode_basis(images, opts)?;
    const WRITER: &[u8] = concat!("imgio ", env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    let has = |key: &str| ktx2.key_values.iter().any(|(k, _)| *k == key);
    let basis_kv = basis_file_key_values(&basis).ok_or(Error::InvalidData)?;
    let mut kv: Vec<(&str, &[u8])> = ktx2.key_values.to_vec();
    if !has("KTXwriter") {
        kv.push(("KTXwriter", WRITER));
    }
    if let Some(d) = find_key_value(&basis_kv, BASISU_DEBLOCK_FILTER_ID_NAME)
        && !has(BASISU_DEBLOCK_FILTER_ID_NAME)
    {
        kv.push((BASISU_DEBLOCK_FILTER_ID_NAME, &d.value));
    }
    write_basis(
        &basis,
        &BasisKtx2Options {
            zstd: ktx2.zstd,
            key_values: &kv,
        },
    )
}

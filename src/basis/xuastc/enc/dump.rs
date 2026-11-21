// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0
//! Dev-only loader for `.xud` dumps of the C++ encoder's `ldr_astc_block_encode_image_output`
//! (written by a patched basisu build, see `docs/spec/xuastc_encode.md`). Used by `#[ignore]`d
//! comparison tests that read dumps from `$XU_DUMP_DIR`.
#![allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]

use super::{BlockInfo, DctCoeff, DctSyms, EncodeBlockOutput, EncodeImageOutput};
use crate::astc::helpers::LogAstcBlock;
use crate::basis::ll::comp::Image;

/// A loaded dump: the encoder input image and its block candidates.
pub(crate) struct Dump {
    pub image: Image,
    pub out: EncodeImageOutput,
    /// `log_astc_block::m_uses_suboptimal_cem_encoding` per candidate (not in [`LogAstcBlock`]).
    pub suboptimal_cem: Vec<Vec<bool>>,
}

struct R<'a>(&'a [u8]);

impl R<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        a
    }
    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.take(2).try_into().unwrap())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
}

fn read_log_blk(r: &mut R) -> (LogAstcBlock, bool) {
    let h = r.take(12).to_vec();
    let mut b = LogAstcBlock {
        error_flag: h[0] != 0,
        solid_color_flag_ldr: h[1] != 0,
        solid_color_flag_hdr: h[2] != 0,
        user_mode: h[3],
        grid_width: h[4],
        grid_height: h[5],
        dual_plane: h[6] != 0,
        weight_ise_range: h[7],
        endpoint_ise_range: h[8],
        color_component_selector: h[9],
        num_partitions: h[10],
        ..Default::default()
    };
    let suboptimal = h[11] != 0;
    b.partition_id = r.u16();
    b.color_endpoint_modes.copy_from_slice(r.take(4));
    b.weights.copy_from_slice(r.take(64));
    b.endpoints.copy_from_slice(r.take(18));
    (b, suboptimal)
}

/// Parses a `.xud` file (format "XUD1", see the C++ `xu_dump_enc_out`).
pub(crate) fn parse(data: &[u8]) -> Dump {
    let mut r = R(data);
    assert_eq!(r.take(4), b"XUD1");
    let (w, h) = (r.u32(), r.u32());
    let (bx, by) = (r.u32(), r.u32());
    let mut image = Image::with_size(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = r.take(4);
            *image.get_mut(x, y) = [p[0], p[1], p[2], p[3]];
        }
    }
    let mut out = EncodeImageOutput::new(bx, by);
    let mut suboptimal_cem = Vec::new();
    for bi in out.block_info.iter_mut() {
        let packed = r.u32();
        let fl = r.take(4).to_vec();
        let n = r.u32();
        let mut info = BlockInfo {
            packed_out_block_index: packed,
            low_freq_block_flag: fl[0] != 0,
            super_strong_edges: fl[1] != 0,
            very_strong_edges: fl[2] != 0,
            strong_edges: fl[3] != 0,
            ..Default::default()
        };
        let mut sub = Vec::new();
        for _ in 0..n {
            let trial_mode_index = r.u16() as i16;
            let blur_id = r.u16();
            let sse = r.u64();
            let (log_blk, s) = read_log_blk(&mut r);
            sub.push(s);
            let mut planes: [DctSyms; 2] = Default::default();
            for p in planes.iter_mut() {
                p.dc_sym = r.u32();
                p.num_dc_levels = r.u32();
                p.max_coeff_mag = r.u32();
                p.max_zigzag_index = r.u32();
                let nc = r.u32();
                for _ in 0..nc {
                    let num_zeros = r.u16();
                    let coeff = r.u16() as i16;
                    p.coeffs.push(DctCoeff { num_zeros, coeff });
                }
            }
            info.out_blocks.push(EncodeBlockOutput {
                trial_mode_index,
                blur_id,
                log_blk,
                packed_dct_plane_data: planes,
                sse,
            });
        }
        suboptimal_cem.push(sub);
        *bi = info;
    }
    assert!(r.0.is_empty(), "trailing dump bytes");
    Dump {
        image,
        out,
        suboptimal_cem,
    }
}

/// Loads `$XU_DUMP_DIR/name`, or `None` when the variable is unset (tests then skip).
pub(crate) fn load(name: &str) -> Option<Dump> {
    let dir = std::env::var("XU_DUMP_DIR").ok()?;
    let data = std::fs::read(format!("{dir}/{name}")).ok()?;
    Some(parse(&data))
}

/// Field-by-field equality of two logical blocks (`LogAstcBlock` has no `PartialEq`).
pub(crate) fn log_blk_eq(a: &LogAstcBlock, b: &LogAstcBlock) -> bool {
    a.error_flag == b.error_flag
        && a.solid_color_flag_ldr == b.solid_color_flag_ldr
        && a.solid_color_flag_hdr == b.solid_color_flag_hdr
        && a.user_mode == b.user_mode
        && a.grid_width == b.grid_width
        && a.grid_height == b.grid_height
        && a.dual_plane == b.dual_plane
        && a.weight_ise_range == b.weight_ise_range
        && a.endpoint_ise_range == b.endpoint_ise_range
        && a.color_component_selector == b.color_component_selector
        && a.num_partitions == b.num_partitions
        && a.partition_id == b.partition_id
        && a.color_endpoint_modes == b.color_endpoint_modes
        && a.weights == b.weights
        && a.endpoints == b.endpoints
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "needs XU_DUMP_DIR (C++ encoder dumps)"]
    fn parse_dumps() {
        let Ok(dir) = std::env::var("XU_DUMP_DIR") else {
            return;
        };
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_some_and(|x| x == "xud") {
                let d = super::parse(&std::fs::read(&p).unwrap());
                assert_eq!(
                    d.out.block_info.len() as u32,
                    d.out.blocks_x * d.out.blocks_y
                );
            }
        }
    }
}

// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! XUASTC LDR -> BC7 fast paths of basisu `basisu_lowlevel_xuastc_ldr_transcoder::
//! transcode_slice` (default decode flags: fast BC7 on, not high quality, no deblocking).
//!
//! - 4x4: solid blocks -> mode 5 solid, single-subset single-plane blocks -> mode 6 straight
//!   from the ASTC endpoints and weights, others decode + analytical encode.
//! - 8x6: tiles of 1x2 ASTC blocks (2x3 BC7 blocks); 6x6: tiles of 2x2 ASTC blocks (3x3 BC7
//!   blocks). BC7 blocks inside one simple ASTC block, or straddling two with matching
//!   endpoints / solid colors, are packed directly (mode 5/6/1); the rest decode the ASTC
//!   blocks and encode analytically. Tiles at the right / bottom edge duplicate the last ASTC
//!   column / row (C++ pointer clamping), so edge BC7 blocks read no clamped pixels.
//!
//! Each pair of ASTC block rows emits its own 3 rows of BC7 blocks, so pairs run in parallel.

use super::astc_ldr::decode_log;
use super::ll::bc7f::{
    PACK_BC7_FLAG_DEFAULT, fast_pack_bc7_auto_rgb, fast_pack_bc7_auto_rgba, pack_mode5_solid,
};
use super::ll::bc7f_xuastc as fx;
use crate::astc::helpers::LogAstcBlock;
use crate::parallel::try_for_each_unit;
use crate::{AstcBlock, Error, Result};

type Px = [u8; 4];

/// True when `block` has a BC7 fast path.
pub(crate) fn has_fast_path(block: AstcBlock) -> bool {
    matches!(block, AstcBlock::B4x4 | AstcBlock::B6x6 | AstcBlock::B8x6)
}

/// Decoded stream state shared by every tile.
struct Ctx<'a> {
    blocks: &'a [LogAstcBlock],
    bxs: usize,
    bys: usize,
    /// BC7 blocks per row.
    dbx: usize,
    srgb: bool,
    has_alpha: bool,
}

impl Ctx<'_> {
    fn blk(&self, bx: usize, by: usize) -> Result<&LogAstcBlock> {
        if bx >= self.bxs || by >= self.bys {
            return Err(Error::InvalidData);
        }
        self.blocks
            .get(by * self.bxs + bx)
            .ok_or(Error::InvalidData)
    }

    /// Analytical BC7 encode of decoded texels (`transcode_4x4_block` for BC7).
    fn encode(&self, px: &[Px; 16], out: &mut [u8]) {
        let mut b = [0u8; 16];
        if self.has_alpha {
            fast_pack_bc7_auto_rgba(&mut b, px, PACK_BC7_FLAG_DEFAULT);
        } else {
            fast_pack_bc7_auto_rgb(&mut b, px, PACK_BC7_FLAG_DEFAULT);
        }
        out.copy_from_slice(&b);
    }

    fn decode<const N: usize>(&self, b: &LogAstcBlock, w: u32, h: u32) -> Result<[Px; N]> {
        let mut px = [[0u8; 4]; N];
        decode_log(b, w, h, self.srgb, &mut px)?;
        Ok(px)
    }

    /// One row of 4x4 blocks.
    fn row_4x4(&self, by: usize, out: &mut [u8]) -> Result<()> {
        for (bx, o) in out.as_chunks_mut::<16>().0.iter_mut().enumerate() {
            let b = self.blk(bx, by)?;
            if b.solid_color_flag_ldr {
                pack_mode5_solid(o, &fx::solid_px(b));
            } else if !b.dual_plane && b.num_partitions == 1 {
                if !fx::pack_from_astc_4x4_single_subset(o, b) {
                    return Err(Error::InvalidData);
                }
            } else {
                let px = self.decode::<16>(b, 4, 4)?;
                self.encode(&px, o);
            }
        }
        Ok(())
    }

    /// Source block row range of row pair `p`: (first row, buffered rows, BC7 rows to emit).
    fn pair(&self, p: usize, out_rows: usize) -> (usize, usize, usize) {
        let by0 = p * 2;
        let nbuf = if by0 + 1 < self.bys { 2 } else { 1 };
        (by0, nbuf, ((nbuf * 6 + 3) >> 2).min(out_rows))
    }

    /// The 8x6 path: 1x2 ASTC blocks (8x12 texels) per tile of 2x3 BC7 blocks.
    fn pair_8x6(&self, p: usize, out: &mut [u8], row: usize) -> Result<()> {
        let (by0, nbuf, emit) = self.pair(p, out.len() / row);
        let mut dec = [[[0u8; 4]; 48]; 2];
        let mut up = [[0u8; 48]; 2];
        for src_bx in 0..self.bxs {
            let b = [self.blk(src_bx, by0)?, self.blk(src_bx, by0 + nbuf - 1)?];
            let mut has_dec = [false; 2];
            let dst = |dy: usize, dx: usize| {
                let dst_bx = src_bx * 2 + dx;
                (dst_bx < self.dbx).then_some(dy * row + dst_bx * 16)
            };
            let gather = |dec: &[[Px; 48]; 2], dx: usize, dy: usize| -> [Px; 16] {
                std::array::from_fn(|i| {
                    let (sx, sy) = (dx * 4 + (i & 3), dy * 4 + (i >> 2));
                    dec[sy / 6][sx + (sy % 6) * 8]
                })
            };

            if fx::blocks_same_solid_colors(b[0], b[1], 0) {
                let mut blk = [0u8; 16];
                pack_mode5_solid(&mut blk, &fx::solid_px(b[0]));
                for dy in 0..emit {
                    for dx in 0..2 {
                        let Some(o) = dst(dy, dx) else { break };
                        out[o..o + 16].copy_from_slice(&blk);
                    }
                }
                continue;
            }

            let hard = |x: &LogAstcBlock| x.dual_plane || x.num_partitions > 1;
            if hard(b[0]) && hard(b[1]) {
                for y in 0..2 {
                    dec[y] = self.decode::<48>(b[y], 8, 6)?;
                }
                for dy in 0..emit {
                    for dx in 0..2 {
                        let Some(o) = dst(dy, dx) else { break };
                        self.encode(&gather(&dec, dx, dy), &mut out[o..o + 16]);
                    }
                }
                continue;
            }

            for y in 0..2 {
                if !fx::astc_upsample_grid_weights(b[y], &mut up[y], 8, 6) {
                    return Err(Error::InvalidData);
                }
            }

            for dy in 0..emit {
                for dx in 0..2 {
                    let Some(o) = dst(dy, dx) else { break };
                    let o = &mut out[o..o + 16];
                    let (top_by, bot_by) = (dy * 4 / 6, (dy * 4 + 3) / 6);
                    let (b0, b1) = (b[top_by], b[bot_by]);
                    let mut blk = [0u8; 16];
                    let full = if hard(b0) || hard(b1) {
                        true
                    } else if top_by == bot_by {
                        if b0.solid_color_flag_ldr {
                            pack_mode5_solid(&mut blk, &fx::solid_px(b0));
                        } else {
                            fx::pack_from_astc_single_subset(
                                &mut blk,
                                b0,
                                &up[top_by],
                                dx * 4,
                                (dy * 4) % 6,
                                8,
                            );
                        }
                        false
                    } else if fx::blocks_same_single_subset_endpoints(b0, b1, 0) {
                        fx::pack_from_astc_to_single_subset_same_endpoints(
                            &mut blk,
                            b0,
                            &up[top_by],
                            b1,
                            &up[bot_by],
                            dx,
                            dy,
                            false,
                        );
                        false
                    } else if !fx::block_has_alpha(b0) && !fx::block_has_alpha(b1) {
                        fx::pack_from_astc_8x6_to_two_subsets_different_endpoints_hq(
                            &mut blk,
                            b0,
                            &up[top_by],
                            b1,
                            &up[bot_by],
                            dx,
                            self.srgb,
                        );
                        false
                    } else {
                        true
                    };
                    if !full {
                        o.copy_from_slice(&blk);
                        continue;
                    }
                    for y in [top_by, bot_by] {
                        if !has_dec[y] {
                            dec[y] = self.decode::<48>(b[y], 8, 6)?;
                            has_dec[y] = true;
                        }
                    }
                    self.encode(&gather(&dec, dx, dy), o);
                }
            }
        }
        Ok(())
    }

    /// The 6x6 path: 2x2 ASTC blocks (12x12 texels) per tile of 3x3 BC7 blocks.
    fn pair_6x6(&self, p: usize, out: &mut [u8], row: usize) -> Result<()> {
        const TOL: u32 = 1;
        let (by0, nbuf, emit) = self.pair(p, out.len() / row);
        let mut dec = [[[[0u8; 4]; 36]; 2]; 2]; // [x][y][texel]
        let mut up = [[[0u8; 36]; 2]; 2]; // [x][y][texel]
        for src_bx in (0..self.bxs).step_by(2) {
            let bx1 = (src_bx + 1).min(self.bxs - 1);
            let by1 = by0 + nbuf - 1;
            // [x][y]
            let b = [
                [self.blk(src_bx, by0)?, self.blk(src_bx, by1)?],
                [self.blk(bx1, by0)?, self.blk(bx1, by1)?],
            ];
            let mut has_dec = [[false; 2]; 2];
            let dst = |dy: usize, dx: usize| {
                let dst_bx = src_bx * 6 / 4 + dx;
                (dst_bx < self.dbx).then_some(dy * row + dst_bx * 16)
            };
            let gather = |dec: &[[[Px; 36]; 2]; 2], dx: usize, dy: usize| -> [Px; 16] {
                std::array::from_fn(|i| {
                    let (sx, sy) = (dx * 4 + (i & 3), dy * 4 + (i >> 2));
                    dec[sx / 6][sy / 6][(sx % 6) + (sy % 6) * 6]
                })
            };
            let all = [b[0][0], b[1][0], b[0][1], b[1][1]];

            let ul = b[0][0];
            let all_solid = all.iter().all(|x| {
                x.solid_color_flag_ldr && (0..4).all(|c| x.solid_color(c) == ul.solid_color(c))
            });
            if all_solid {
                let mut blk = [0u8; 16];
                pack_mode5_solid(&mut blk, &fx::solid_px(ul));
                for dy in 0..emit {
                    for dx in 0..3 {
                        let Some(o) = dst(dy, dx) else { break };
                        out[o..o + 16].copy_from_slice(&blk);
                    }
                }
                continue;
            }

            let hard = |x: &LogAstcBlock| x.dual_plane || x.num_partitions > 1;
            let num_hard = all.iter().filter(|x| hard(x)).count();
            if num_hard == 4 {
                for x in 0..2 {
                    for y in 0..2 {
                        dec[x][y] = self.decode::<36>(b[x][y], 6, 6)?;
                    }
                }
                for dy in 0..emit {
                    for dx in 0..3 {
                        let Some(o) = dst(dy, dx) else { break };
                        self.encode(&gather(&dec, dx, dy), &mut out[o..o + 16]);
                    }
                }
                continue;
            }

            for x in 0..2 {
                for y in 0..2 {
                    if !fx::astc_upsample_grid_weights(b[x][y], &mut up[x][y], 6, 6) {
                        return Err(Error::InvalidData);
                    }
                }
            }

            for dy in 0..emit {
                for dx in 0..3 {
                    if dx == 1 && dy == 1 {
                        continue; // the center block is handled last
                    }
                    let Some(o) = dst(dy, dx) else { break };
                    let o = &mut out[o..o + 16];
                    let (top_bx, top_by) = (dx * 4 / 6, dy * 4 / 6);
                    let (bot_bx, bot_by) = ((dx * 4 + 3) / 6, (dy * 4 + 3) / 6);
                    let (b0, b1) = (b[top_bx][top_by], b[bot_bx][bot_by]);
                    let (w0, w1) = (&up[top_bx][top_by], &up[bot_bx][bot_by]);
                    let mut blk = [0u8; 16];
                    let full = if hard(b0) || hard(b1) {
                        true
                    } else if top_bx == bot_bx && top_by == bot_by {
                        if b0.solid_color_flag_ldr {
                            pack_mode5_solid(&mut blk, &fx::solid_px(b0));
                        } else {
                            fx::pack_from_astc_single_subset(
                                &mut blk,
                                b0,
                                w0,
                                (dx * 4) % 6,
                                (dy * 4) % 6,
                                6,
                            );
                        }
                        false
                    } else if fx::blocks_same_solid_colors(b0, b1, 0) {
                        pack_mode5_solid(&mut blk, &fx::solid_px(b0));
                        false
                    } else if fx::blocks_same_single_subset_endpoints(b0, b1, TOL) {
                        fx::pack_from_astc_to_single_subset_same_endpoints(
                            &mut blk, b0, w0, b1, w1, dx, dy, true,
                        );
                        false
                    } else if !fx::block_has_alpha(b0) && !fx::block_has_alpha(b1) {
                        fx::pack_from_astc_6x6_to_two_subsets_different_endpoints(
                            &mut blk, b0, w0, b1, w1, dx, dy,
                        );
                        false
                    } else {
                        true
                    };
                    if !full {
                        o.copy_from_slice(&blk);
                        continue;
                    }
                    for (x, y) in [(top_bx, top_by), (bot_bx, bot_by)] {
                        if !has_dec[x][y] {
                            dec[x][y] = self.decode::<36>(b[x][y], 6, 6)?;
                            has_dec[x][y] = true;
                        }
                    }
                    self.encode(&gather(&dec, dx, dy), o);
                }
            }

            // The center BC7 block overlaps all 4 ASTC blocks.
            let Some(o) = dst(1, 1).filter(|_| emit > 1) else {
                continue;
            };
            let o = &mut out[o..o + 16];
            if num_hard == 0 {
                let same = |x: &LogAstcBlock, y: &LogAstcBlock| {
                    (fx::blocks_same_solid_colors(x, y, 1)
                        || fx::blocks_same_single_subset_endpoints(x, y, TOL))
                        && !fx::block_has_alpha(x)
                        && !fx::block_has_alpha(y)
                };
                let top = same(b[0][0], b[1][0]);
                let bot = same(b[0][1], b[1][1]);
                let left = same(b[0][0], b[0][1]);
                let right = same(b[1][0], b[1][1]);
                if (top && bot) || (left && right) {
                    let mut blk = [0u8; 16];
                    fx::pack_astc_6x6_to_two_subsets_middle_block(&mut blk, &b, &up, !(top && bot));
                    o.copy_from_slice(&blk);
                    continue;
                }
            }
            for x in 0..2 {
                for y in 0..2 {
                    if !has_dec[x][y] {
                        dec[x][y] = self.decode::<36>(b[x][y], 6, 6)?;
                        has_dec[x][y] = true;
                    }
                }
            }
            self.encode(&gather(&dec, 1, 1), o);
        }
        Ok(())
    }
}

/// Transcodes logical XUASTC LDR `blocks` (4x4, 6x6 or 8x6) to BC7 into `dst`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn transcode(
    blocks: &[LogAstcBlock],
    block: AstcBlock,
    blocks_x: u32,
    blocks_y: u32,
    width: u32,
    height: u32,
    srgb: bool,
    has_alpha: bool,
    dst: &mut [u8],
) -> Result<()> {
    let (bxs, bys) = (blocks_x as usize, blocks_y as usize);
    if bxs == 0 || bys == 0 || blocks.len() < bxs * bys {
        return Err(Error::InvalidData);
    }
    let (w, h) = (width as usize, height as usize);
    let (dbx, dby) = (w.div_ceil(4), h.div_ceil(4));
    let row = dbx * 16;
    let dst = dst.get_mut(..row * dby).ok_or(Error::InvalidArgument)?;
    let ctx = Ctx {
        blocks,
        bxs,
        bys,
        dbx,
        srgb,
        has_alpha,
    };
    let pixels = w.saturating_mul(h);
    match block {
        AstcBlock::B4x4 => try_for_each_unit(dst, row, pixels, |by, out| ctx.row_4x4(by, out)),
        AstcBlock::B6x6 => {
            try_for_each_unit(dst, row * 3, pixels, |p, out| ctx.pair_6x6(p, out, row))
        }
        AstcBlock::B8x6 => {
            try_for_each_unit(dst, row * 3, pixels, |p, out| ctx.pair_8x6(p, out, row))
        }
        _ => Err(Error::Unsupported),
    }
}

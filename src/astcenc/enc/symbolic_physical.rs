// SPDX-FileCopyrightText: 2011-2026 Arm Limited
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from astc-encoder and modified; see NOTICE.md.
//! Compressor side of `astcenc_symbolic_physical.cpp` and `astcenc_integer_sequence.cpp`:
//! `symbolic_to_physical` and `encode_ise`.

#![allow(clippy::needless_range_loop)]

use super::internal::*;
use super::tables::{
    COLOR_UQUANT_TO_SCRAMBLED_PQUANT, INTEGER_OF_QUINTS, INTEGER_OF_TRITS, QUANT_AND_XFER_TABLES,
};

/// `btq_counts`: (bits, trits, quints) per quant level.
const BTQ_COUNTS: [(u32, bool, bool); 21] = [
    (1, false, false),
    (0, true, false),
    (2, false, false),
    (0, false, true),
    (1, true, false),
    (3, false, false),
    (1, false, true),
    (2, true, false),
    (4, false, false),
    (2, false, true),
    (3, true, false),
    (5, false, false),
    (3, false, true),
    (4, true, false),
    (6, false, false),
    (4, false, true),
    (5, true, false),
    (7, false, false),
    (5, false, true),
    (6, true, false),
    (8, false, false),
];

isa_fn! {
/// `write_bits` (bytes past the end of `ptr` are not written).
fn write_bits(value: u32, bitcount: u32, bitoffset: u32, ptr: &mut [u8]) {
    let mut mask: u32 = (1u32 << bitcount) - 1;
    let mut value = value & mask;
    let p = (bitoffset >> 3) as usize;
    let bitoffset = bitoffset & 7;
    value <<= bitoffset;
    mask <<= bitoffset;
    mask = !mask;
    if let Some(b) = ptr.get_mut(p) {
        *b &= mask as u8;
        *b |= value as u8;
    }
    if let Some(b) = ptr.get_mut(p + 1) {
        *b &= (mask >> 8) as u8;
        *b |= (value >> 8) as u8;
    }
}
}

isa_fn! {
/// `encode_ise`: packs `character_count` values of `input_data` at `bit_offset`.
pub fn encode_ise(
    quant_level: u32,
    character_count: u32,
    input_data: &[u8],
    output_data: &mut [u8],
    mut bit_offset: u32,
) {
    let (bits, trits, quints) = BTQ_COUNTS[quant_level as usize];
    let mask: u32 = (1u32 << bits) - 1;
    let n = character_count as usize;
    let inp = |i: usize| u32::from(input_data.get(i).copied().unwrap_or(0));

    if trits {
        let mut i = 0usize;
        let full_trit_blocks = n / 5;
        let mut put = |pack: u32, count: u32, bit_offset: &mut u32| {
            write_bits(pack & 0xFF, count, *bit_offset, output_data);
            *bit_offset += count;
        };
        for _ in 0..full_trit_blocks {
            let t = |k: usize| (inp(i + k) >> bits) as usize;
            let tv =
                u32::from(INTEGER_OF_TRITS[t(4) * 81 + t(3) * 27 + t(2) * 9 + t(1) * 3 + t(0)]);
            // Element 0 + T0 + T1
            put(
                (inp(i) & mask) | ((tv & 0x3) << bits),
                bits + 2,
                &mut bit_offset,
            );
            // Element 1 + T2 + T3
            put(
                (inp(i + 1) & mask) | (((tv >> 2) & 0x3) << bits),
                bits + 2,
                &mut bit_offset,
            );
            // Element 2 + T4
            put(
                (inp(i + 2) & mask) | (((tv >> 4) & 0x1) << bits),
                bits + 1,
                &mut bit_offset,
            );
            // Element 3 + T5 + T6
            put(
                (inp(i + 3) & mask) | (((tv >> 5) & 0x3) << bits),
                bits + 2,
                &mut bit_offset,
            );
            // Element 4 + T7
            put(
                (inp(i + 4) & mask) | (((tv >> 7) & 0x1) << bits),
                bits + 1,
                &mut bit_offset,
            );
            i += 5;
        }

        // Loop tail for a partial block
        if i != n {
            let t = |k: usize| {
                if i + k >= n {
                    0
                } else {
                    (inp(i + k) >> bits) as usize
                }
            };
            let tv = u32::from(INTEGER_OF_TRITS[t(3) * 27 + t(2) * 9 + t(1) * 3 + t(0)]);
            const TBITS: [u32; 4] = [2, 2, 1, 2];
            const TSHIFT: [u32; 4] = [0, 2, 4, 5];
            let mut j = 0;
            while i < n {
                let pack = (inp(i) & mask) | (((tv >> TSHIFT[j]) & ((1 << TBITS[j]) - 1)) << bits);
                put(pack, bits + TBITS[j], &mut bit_offset);
                i += 1;
                j += 1;
            }
        }
    } else if quints {
        let mut i = 0usize;
        let full_quint_blocks = n / 3;
        let mut put = |pack: u32, count: u32, bit_offset: &mut u32| {
            write_bits(pack & 0xFF, count, *bit_offset, output_data);
            *bit_offset += count;
        };
        for _ in 0..full_quint_blocks {
            let q = |k: usize| (inp(i + k) >> bits) as usize;
            let tv = u32::from(INTEGER_OF_QUINTS[(q(2) * 5 + q(1)) * 5 + q(0)]);
            // Element 0
            put(
                (inp(i) & mask) | ((tv & 0x7) << bits),
                bits + 3,
                &mut bit_offset,
            );
            // Element 1
            put(
                (inp(i + 1) & mask) | (((tv >> 3) & 0x3) << bits),
                bits + 2,
                &mut bit_offset,
            );
            // Element 2
            put(
                (inp(i + 2) & mask) | (((tv >> 5) & 0x3) << bits),
                bits + 2,
                &mut bit_offset,
            );
            i += 3;
        }

        // Loop tail for a partial block
        if i != n {
            let q1 = if i + 1 >= n {
                0
            } else {
                (inp(i + 1) >> bits) as usize
            };
            let q0 = (inp(i) >> bits) as usize;
            let tv = u32::from(INTEGER_OF_QUINTS[q1 * 5 + q0]);
            const TBITS: [u32; 2] = [3, 2];
            const TSHIFT: [u32; 2] = [0, 3];
            let mut j = 0;
            while i < n {
                let pack = (inp(i) & mask) | (((tv >> TSHIFT[j]) & ((1 << TBITS[j]) - 1)) << bits);
                put(pack, bits + TBITS[j], &mut bit_offset);
                i += 1;
                j += 1;
            }
        }
    } else {
        for i in 0..n {
            write_bits(inp(i), bits, bit_offset, output_data);
            bit_offset += bits;
        }
    }
}
}

isa_fn! {
/// `bitrev8`.
fn bitrev8(p: u8) -> u8 {
    p.reverse_bits()
}
}

isa_fn! {
/// `symbolic_to_physical`.
pub fn symbolic_to_physical(
    bsd: &BlockSizeDescriptor,
    scb: &SymbolicCompressedBlock,
    pcb: &mut [u8; 16],
) {
    debug_assert!(scb.block_type != SYM_BTYPE_ERROR);

    // Constant color blocks; there is currently no attempt to coalesce larger void-extents
    let const_bytes: Option<[u8; 8]> = match scb.block_type {
        SYM_BTYPE_CONST_U16 => Some([0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]),
        SYM_BTYPE_CONST_F16 => Some([0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]),
        _ => None,
    };
    if let Some(cbytes) = const_bytes {
        pcb[..8].copy_from_slice(&cbytes);
        for i in 0..BLOCK_MAX_COMPONENTS {
            pcb[2 * i + 8] = (scb.constant_color[i] & 0xFF) as u8;
            pcb[2 * i + 9] = ((scb.constant_color[i] >> 8) & 0xFF) as u8;
        }
        return;
    }

    let partition_count = u32::from(scb.partition_count);

    // Compress the weights. They are encoded as an ordinary integer-sequence, then bit-reversed
    let mut weightbuf = [0u8; 16];

    let bm = bsd.get_block_mode(scb.block_mode as usize);
    let di = bsd.get_decimation_info(bm.decimation_mode as usize);
    let weight_count = di.weight_count as usize;
    let weight_quant_method = bm.get_weight_quant_mode();
    let weight_quant_levels = get_quant_level(weight_quant_method) as f32;
    let is_dual_plane = bm.is_dual_plane;

    let qat = &QUANT_AND_XFER_TABLES[weight_quant_method as usize];

    let real_weight_count = if is_dual_plane {
        2 * weight_count
    } else {
        weight_count
    };

    let bits_for_weights =
        super::get_ise_sequence_bitcount(real_weight_count as u32, weight_quant_method as usize);

    let quantize = |uqw: u8| {
        let qw = (f32::from(uqw) / 64.0) * (weight_quant_levels - 1.0);
        let qwi = (qw + 0.5) as usize;
        qat.scramble_map[qwi.min(31)]
    };
    let mut weights = [0u8; 64];
    if is_dual_plane {
        for i in 0..weight_count {
            weights[2 * i] = quantize(scb.weights[i]);
            weights[2 * i + 1] = quantize(scb.weights[i + WEIGHTS_PLANE2_OFFSET]);
        }
    } else {
        for i in 0..weight_count {
            weights[i] = quantize(scb.weights[i]);
        }
    }

    encode_ise(
        weight_quant_method,
        real_weight_count as u32,
        &weights,
        &mut weightbuf,
        0,
    );

    for i in 0..16 {
        pcb[i] = bitrev8(weightbuf[15 - i]);
    }

    write_bits(u32::from(scb.block_mode), 11, 0, pcb);
    write_bits(partition_count - 1, 2, 11, pcb);

    let mut below_weights_pos = 128 - bits_for_weights as i32;

    // Encode partition index and color endpoint types for blocks with 2+ partitions
    if partition_count > 1 {
        let pidx = u32::from(scb.partition_index);
        write_bits(pidx, 6, 13, pcb);
        write_bits(pidx >> 6, PARTITION_INDEX_BITS - 6, 19, pcb);

        if scb.color_formats_matched != 0 {
            write_bits(
                u32::from(scb.color_formats[0]) << 2,
                6,
                13 + PARTITION_INDEX_BITS,
                pcb,
            );
        } else {
            // Check endpoint types for each partition to determine the lowest class present
            let mut low_class = 4u32;
            for i in 0..partition_count as usize {
                let class_of_format = u32::from(scb.color_formats[i] >> 2);
                low_class = low_class.min(class_of_format);
            }
            if low_class == 3 {
                low_class = 2;
            }

            let mut encoded_type = low_class + 1;
            let mut bitpos = 2;
            for i in 0..partition_count as usize {
                let classbit_of_format =
                    (u32::from(scb.color_formats[i] >> 2)).wrapping_sub(low_class);
                encoded_type |= classbit_of_format << bitpos;
                bitpos += 1;
            }

            for i in 0..partition_count as usize {
                let lowbits_of_format = u32::from(scb.color_formats[i] & 3);
                encoded_type |= lowbits_of_format << bitpos;
                bitpos += 2;
            }

            let encoded_type_lowpart = encoded_type & 0x3F;
            let encoded_type_highpart = encoded_type >> 6;
            let encoded_type_highpart_size = (3 * partition_count) - 4;
            let encoded_type_highpart_pos = 128 - bits_for_weights - encoded_type_highpart_size;
            write_bits(encoded_type_lowpart, 6, 13 + PARTITION_INDEX_BITS, pcb);
            write_bits(
                encoded_type_highpart,
                encoded_type_highpart_size,
                encoded_type_highpart_pos,
                pcb,
            );
            below_weights_pos -= encoded_type_highpart_size as i32;
        }
    } else {
        write_bits(u32::from(scb.color_formats[0]), 4, 13, pcb);
    }

    // In dual-plane mode, encode the color component of the second plane of weights
    if is_dual_plane {
        write_bits(
            scb.plane2_component as u32,
            2,
            (below_weights_pos - 2) as u32,
            pcb,
        );
    }

    // Encode the color components
    let mut values_to_encode = [0u8; 32];
    let mut valuecount_to_encode = 0usize;

    let pack_table = &COLOR_UQUANT_TO_SCRAMBLED_PQUANT[(scb.quant_mode - QUANT_6) as usize];
    for i in 0..partition_count as usize {
        let vals = 2 * (scb.color_formats[i] as usize >> 2) + 2;
        for j in 0..vals {
            values_to_encode[j + valuecount_to_encode] =
                pack_table[scb.color_values[i][j] as usize];
        }
        valuecount_to_encode += vals;
    }

    encode_ise(
        scb.get_color_quant_mode(),
        valuecount_to_encode as u32,
        &values_to_encode,
        pcb,
        if partition_count == 1 {
            17
        } else {
            19 + PARTITION_INDEX_BITS
        },
    );
}
}

isa_fn! {
/// `physical_to_symbolic` with the compressor's block size descriptor (LZ-RDO decodes the
/// candidate blocks it splices together).
pub fn physical_to_symbolic(
    bsd: &BlockSizeDescriptor,
    pcb: &[u8; 16],
    scb: &mut SymbolicCompressedBlock,
) {
    use crate::astcenc::{decode_ise, read_bits};
    use super::tables::{
        COLOR_SCRAMBLED_PQUANT_TO_UQUANT, QUANT_MODE_TABLE, WEIGHT_UNSCRAMBLE_AND_UNQUANT,
    };

    scb.block_type = SYM_BTYPE_NONCONST;

    // Extract header fields
    let block_mode = read_bits(11, 0, pcb) as usize;
    if block_mode & 0x1FF == 0x1FC {
        // Constant color block: check what format the data has
        scb.block_type = if block_mode & 0x200 != 0 {
            SYM_BTYPE_CONST_F16
        } else {
            SYM_BTYPE_CONST_U16
        };
        scb.partition_count = 0;
        for i in 0..4 {
            scb.constant_color[i] = i32::from(pcb[2 * i + 8]) | (i32::from(pcb[2 * i + 9]) << 8);
        }

        // Additionally, check that the void-extent
        if bsd.dim_z == 1 {
            // 2D void-extent
            if read_bits(2, 10, pcb) != 3 {
                scb.block_type = SYM_BTYPE_ERROR;
                return;
            }
            // Low values span 3 bytes so need two read_bits calls
            let vx_low_s = read_bits(8, 12, pcb) | (read_bits(5, 12 + 8, pcb) << 8);
            let vx_high_s = read_bits(13, 25, pcb);
            let vx_low_t = read_bits(8, 38, pcb) | (read_bits(5, 38 + 8, pcb) << 8);
            let vx_high_t = read_bits(13, 51, pcb);
            let all_ones = vx_low_s == 0x1FFF
                && vx_high_s == 0x1FFF
                && vx_low_t == 0x1FFF
                && vx_high_t == 0x1FFF;
            if (vx_low_s >= vx_high_s || vx_low_t >= vx_high_t) && !all_ones {
                scb.block_type = SYM_BTYPE_ERROR;
            }
        } else {
            // 3D void-extent
            let v = |o: u32| read_bits(9, o, pcb);
            let (ls, hs, lt, ht, lr, hr) = (v(10), v(19), v(28), v(37), v(46), v(55));
            let all_ones = [ls, hs, lt, ht, lr, hr].iter().all(|&x| x == 0x1FF);
            if (ls >= hs || lt >= ht || lr >= hr) && !all_ones {
                scb.block_type = SYM_BTYPE_ERROR;
            }
        }
        return;
    }

    if bsd.block_mode_packed_index[block_mode] == BLOCK_BAD_BLOCK_MODE {
        scb.block_type = SYM_BTYPE_ERROR;
        return;
    }

    let bm = bsd.get_block_mode(block_mode);
    let di = bsd.get_decimation_info(bm.decimation_mode as usize);

    let weight_count = di.weight_count as usize;
    let weight_quant_method = bm.quant_mode as usize;
    let is_dual_plane = bm.is_dual_plane;
    let real_weight_count = if is_dual_plane {
        2 * weight_count
    } else {
        weight_count
    };

    let partition_count = read_bits(2, 11, pcb) as usize + 1;

    scb.block_mode = block_mode as u16;
    scb.partition_count = partition_count as u8;

    let mut bswapped = [0u8; 16];
    for (i, b) in bswapped.iter_mut().enumerate() {
        *b = bitrev8(pcb[15 - i]);
    }

    let bits_for_weights =
        super::get_ise_sequence_bitcount(real_weight_count as u32, weight_quant_method) as i32;
    let mut below_weights_pos = 128 - bits_for_weights;

    let mut indices = [0u8; 64];
    decode_ise(
        weight_quant_method,
        real_weight_count,
        &bswapped,
        &mut indices,
        0,
    );
    let Some(map) = WEIGHT_UNSCRAMBLE_AND_UNQUANT.get(weight_quant_method) else {
        scb.block_type = SYM_BTYPE_ERROR;
        return;
    };
    let unquant = |v: u8| map.get(v as usize).copied().unwrap_or(0);
    if is_dual_plane {
        for i in 0..weight_count {
            scb.weights[i] = unquant(indices[2 * i]);
            scb.weights[i + WEIGHTS_PLANE2_OFFSET] = unquant(indices[2 * i + 1]);
        }
    } else {
        for i in 0..weight_count {
            scb.weights[i] = unquant(indices[i]);
        }
    }

    if is_dual_plane && partition_count == 4 {
        scb.block_type = SYM_BTYPE_ERROR;
        return;
    }

    scb.color_formats_matched = 0;

    // Determine the format of each endpoint pair
    let mut color_formats = [0u32; BLOCK_MAX_PARTITIONS];
    let mut encoded_type_highpart_size: i32 = 0;
    if partition_count == 1 {
        color_formats[0] = read_bits(4, 13, pcb);
        scb.partition_index = 0;
    } else {
        encoded_type_highpart_size = (3 * partition_count as i32) - 4;
        below_weights_pos -= encoded_type_highpart_size;
        let encoded_type = read_bits(6, 13 + PARTITION_INDEX_BITS, pcb)
            | (read_bits(
                encoded_type_highpart_size as u32,
                below_weights_pos.max(0) as u32,
                pcb,
            ) << 6);
        let mut baseclass = encoded_type & 0x3;
        if baseclass == 0 {
            for f in color_formats.iter_mut().take(partition_count) {
                *f = (encoded_type >> 2) & 0xF;
            }
            below_weights_pos += encoded_type_highpart_size;
            scb.color_formats_matched = 1;
            encoded_type_highpart_size = 0;
        } else {
            let mut bitpos = 2;
            baseclass -= 1;
            for f in color_formats.iter_mut().take(partition_count) {
                *f = (((encoded_type >> bitpos) & 1) + baseclass) << 2;
                bitpos += 1;
            }
            for f in color_formats.iter_mut().take(partition_count) {
                *f |= (encoded_type >> bitpos) & 3;
                bitpos += 2;
            }
        }
        scb.partition_index = read_bits(10, 13, pcb) as u16;

        // Reject blocks selecting a partitioning that is not present in this block size
        // descriptor
        if bsd.partitioning_packed_index[partition_count - 2][scb.partition_index as usize]
            == BLOCK_BAD_PARTITIONING
        {
            scb.block_type = SYM_BTYPE_ERROR;
            return;
        }
    }

    for i in 0..partition_count {
        scb.color_formats[i] = color_formats[i] as u8;
    }

    // Determine number of color endpoint integers
    let color_integer_count: usize = color_formats[..partition_count]
        .iter()
        .map(|f| ((f >> 2) as usize + 1) * 2)
        .sum();
    if color_integer_count > 18 {
        scb.block_type = SYM_BTYPE_ERROR;
        return;
    }

    // Determine the color endpoint format to use
    const COLOR_BITS_ARR: [i32; 5] = [
        -1,
        115 - 4,
        113 - 4 - PARTITION_INDEX_BITS as i32,
        113 - 4 - PARTITION_INDEX_BITS as i32,
        113 - 4 - PARTITION_INDEX_BITS as i32,
    ];
    let mut color_bits =
        COLOR_BITS_ARR[partition_count] - bits_for_weights - encoded_type_highpart_size;
    if is_dual_plane {
        color_bits -= 2;
    }
    let color_bits = color_bits.max(0) as usize;

    let color_quant_level = QUANT_MODE_TABLE[color_integer_count >> 1][color_bits.min(127)];
    if color_quant_level < QUANT_6 as i8 {
        scb.block_type = SYM_BTYPE_ERROR;
        return;
    }

    // Unpack the integer color values and assign to endpoints
    scb.quant_mode = color_quant_level as u32;

    let mut values_to_decode = [0u8; 32];
    decode_ise(
        color_quant_level as usize,
        color_integer_count,
        pcb,
        &mut values_to_decode,
        if partition_count == 1 {
            17
        } else {
            19 + PARTITION_INDEX_BITS
        },
    );

    let unpack_table = COLOR_SCRAMBLED_PQUANT_TO_UQUANT[(scb.quant_mode - QUANT_6) as usize];
    let mut valuecount_to_decode = 0;
    for i in 0..partition_count {
        let vals = 2 * (color_formats[i] >> 2) as usize + 2;
        for j in 0..vals {
            scb.color_values[i][j] = unpack_table
                .get(values_to_decode[j + valuecount_to_decode] as usize)
                .copied()
                .unwrap_or(0);
        }
        valuecount_to_decode += vals;
    }

    // Fetch component for second-plane in the case of dual plane of weights
    scb.plane2_component = -1;
    if is_dual_plane {
        scb.plane2_component = read_bits(2, (below_weights_pos - 2).max(0) as u32, pcb) as i8;
    }
}
}

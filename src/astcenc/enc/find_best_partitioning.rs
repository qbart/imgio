// SPDX-FileCopyrightText: 2011-2026 Arm Limited
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from astc-encoder and modified; see NOTICE.md.
//! Port of `astcenc_find_best_partitioning.cpp`: k-means clustering of the block texels, the
//! partitioning ordering by coverage mismatch and the candidate selection by line errors.

#![allow(clippy::needless_range_loop)]

use super::averages_and_directions::*;
use super::internal::*;
use super::vecmath::*;

isa_fn! {
/// `kmeans_init`.
fn kmeans_init(
    blk: &ImageBlock,
    texel_count: usize,
    partition_count: usize,
    cluster_centers: &mut [Vfloat4; 4],
) {
    let mut clusters_selected = 0usize;
    let mut distances = [0.0f32; BLOCK_MAX_TEXELS];

    // Pick a random sample as first cluster center; 145897 from random.org
    let mut sample = 145897 % texel_count;
    let mut center_color = blk.texel(sample);
    cluster_centers[clusters_selected] = center_color;
    clusters_selected += 1;

    // Compute the distance to the first cluster center
    let mut distance_sum = 0.0f32;
    for i in 0..texel_count {
        let diff = blk.texel(i) - center_color;
        let distance = (diff * diff).dot_s(blk.channel_weight);
        distance_sum += distance;
        distances[i] = distance;
    }

    // More numbers from random.org for weighted-random center selection
    const CLUSTER_CUTOFFS: [f32; 9] = [
        0.626220, 0.932770, 0.275454, 0.318558, 0.240113, 0.009190, 0.347661, 0.731960, 0.156391,
    ];
    let mut cutoff = (clusters_selected - 1) + 3 * (partition_count - 2);

    // Pick the remaining samples as needed
    loop {
        // Pick the next center in a weighted-random fashion.
        let mut summa = 0.0f32;
        let distance_cutoff = distance_sum * CLUSTER_CUTOFFS[cutoff.min(8)];
        cutoff += 1;
        sample = 0;
        while sample < texel_count {
            summa += distances[sample];
            if summa >= distance_cutoff {
                break;
            }
            sample += 1;
        }
        // Clamp to a valid range and store the selected cluster center
        sample = sample.min(texel_count - 1);
        center_color = blk.texel(sample);
        cluster_centers[clusters_selected] = center_color;
        clusters_selected += 1;
        if clusters_selected >= partition_count {
            break;
        }
        // Compute the distance to the new cluster center, keep the min distance
        distance_sum = 0.0;
        for i in 0..texel_count {
            let diff = blk.texel(i) - center_color;
            let mut distance = (diff * diff).dot_s(blk.channel_weight);
            distance = super::mathlib::min(distance, distances[i]);
            distance_sum += distance;
            distances[i] = distance;
        }
    }
}
}

isa_fn! {
/// `kmeans_assign`.
fn kmeans_assign(
    blk: &ImageBlock,
    texel_count: usize,
    partition_count: usize,
    cluster_centers: &[Vfloat4; 4],
    partition_of_texel: &mut [u8; BLOCK_MAX_TEXELS],
) {
    let mut partition_texel_count = [0u8; BLOCK_MAX_PARTITIONS];
    // Find the best partition for every texel
    for i in 0..texel_count {
        let mut best_distance = f32::MAX;
        let mut best_partition = 0usize;
        let color = blk.texel(i);
        for j in 0..partition_count {
            let diff = color - cluster_centers[j];
            let distance = (diff * diff).dot_s(blk.channel_weight);
            if distance < best_distance {
                best_distance = distance;
                best_partition = j;
            }
        }
        partition_of_texel[i] = best_partition as u8;
        partition_texel_count[best_partition] =
            partition_texel_count[best_partition].wrapping_add(1);
    }

    // It is possible to get a situation where a partition ends up without any texels. In this
    // case, assign texel N to partition N. This is silly, but ensures that every partition
    // retains at least one texel. Reassigning a texel in this manner may cause another
    // partition to go empty, so if we actually did a reassignment, run the whole loop over again.
    loop {
        let mut problem_case = false;
        for i in 0..partition_count {
            if partition_texel_count[i] == 0 {
                let p = partition_of_texel[i] as usize;
                partition_texel_count[p] = partition_texel_count[p].wrapping_sub(1);
                partition_texel_count[i] = partition_texel_count[i].wrapping_add(1);
                partition_of_texel[i] = i as u8;
                problem_case = true;
            }
        }
        if !problem_case {
            break;
        }
    }
}
}

isa_fn! {
/// `kmeans_update`.
fn kmeans_update(
    blk: &ImageBlock,
    texel_count: usize,
    partition_count: usize,
    cluster_centers: &mut [Vfloat4; 4],
    partition_of_texel: &[u8; BLOCK_MAX_TEXELS],
) {
    let mut color_sum = [Vfloat4::zero(); BLOCK_MAX_PARTITIONS];
    let mut partition_texel_count = [0u8; BLOCK_MAX_PARTITIONS];
    // Find the center-of-gravity in each cluster
    for i in 0..texel_count {
        let partition = partition_of_texel[i] as usize & 3;
        color_sum[partition] += blk.texel(i);
        partition_texel_count[partition] = partition_texel_count[partition].wrapping_add(1);
    }
    // Set the center of gravity to be the new cluster center
    for i in 0..partition_count {
        let scale = 1.0 / partition_texel_count[i] as f32;
        cluster_centers[i] = color_sum[i] * scale;
    }
}
}

isa_fn! {
/// `partition_mismatch2`: the second masks are complements of the first within the sample,
/// so only the first XOR is counted; swapping the labels gives `texel_count - mismatch`.
fn partition_mismatch2(a: &[u64], b: &[u64; 2], texel_count: u32) -> u8 {
    let mismatch = (a[0] ^ b[0]).count_ones();
    mismatch.min(texel_count.wrapping_sub(mismatch)) as u8
}
}

isa_fn! {
/// `partition_mismatch3`: OR of the XORs of the first two masks counts each mismatched texel
/// once; the minimum over the six label permutations.
fn partition_mismatch3(a: &[u64], b: &[u64; 3]) -> u8 {
    let (p00, p01, p02) = (a[0] ^ b[0], a[0] ^ b[1], a[0] ^ b[2]);
    let (p10, p11, p12) = (a[1] ^ b[0], a[1] ^ b[1], a[1] ^ b[2]);
    let v0 = (p00 | p11).count_ones().min((p00 | p12).count_ones());
    let v1 = (p01 | p10).count_ones().min((p01 | p12).count_ones());
    let v2 = (p02 | p10).count_ones().min((p02 | p11).count_ones());
    v0.min(v1).min(v2) as u8
}
}

isa_fn! {
/// `partition_mismatch4`.
fn partition_mismatch4(a: &[u64], b: &[u64; 4]) -> u8 {
    let p = |i: usize, j: usize| popcount(a[i] ^ b[j]);
    let mx23 = (p(2, 2) + p(3, 3)).min(p(2, 3) + p(3, 2));
    let mx13 = (p(2, 1) + p(3, 3)).min(p(2, 3) + p(3, 1));
    let mx12 = (p(2, 1) + p(3, 2)).min(p(2, 2) + p(3, 1));
    let mx03 = (p(2, 0) + p(3, 3)).min(p(2, 3) + p(3, 0));
    let mx02 = (p(2, 0) + p(3, 2)).min(p(2, 2) + p(3, 0));
    let mx01 = (p(2, 1) + p(3, 0)).min(p(2, 0) + p(3, 1));
    let v0 = p(0, 0) + (p(1, 1) + mx23).min(p(1, 2) + mx13).min(p(1, 3) + mx12);
    let v1 = p(0, 1) + (p(1, 0) + mx23).min(p(1, 2) + mx03).min(p(1, 3) + mx02);
    let v2 = p(0, 2) + (p(1, 1) + mx03).min(p(1, 0) + mx13).min(p(1, 3) + mx01);
    let v3 = p(0, 3) + (p(1, 1) + mx02).min(p(1, 2) + mx01).min(p(1, 0) + mx12);
    (v0.min(v1).min(v2.min(v3)) / 2) as u8
}
}

isa_fn! {
/// `count_partition_mismatch_bits`.
fn count_partition_mismatch_bits(
    bsd: &BlockSizeDescriptor,
    partition_count: usize,
    bitmaps: &[u64; 4],
    mismatch_counts: &mut [u8; BLOCK_MAX_PARTITIONINGS],
) {
    let active_count = bsd.partitioning_count_selected[partition_count - 1] as usize;
    let texel_count = (bsd.texel_count as u32).min(BLOCK_MAX_KMEANS_TEXELS as u32);
    for i in 0..active_count {
        mismatch_counts[i] = match partition_count {
            2 => partition_mismatch2(bitmaps, &bsd.coverage_bitmaps_2[i], texel_count),
            3 => partition_mismatch3(bitmaps, &bsd.coverage_bitmaps_3[i]),
            _ => partition_mismatch4(bitmaps, &bsd.coverage_bitmaps_4[i]),
        };
    }
}
}

isa_fn! {
/// `get_partition_ordering_by_mismatch_bits` (counting sort by mismatch count).
fn get_partition_ordering_by_mismatch_bits(
    texel_count: usize,
    partitioning_count: usize,
    mismatch_count: &[u8; BLOCK_MAX_PARTITIONINGS],
    partition_ordering: &mut [u16; BLOCK_MAX_PARTITIONINGS],
) -> usize {
    let mut mscount = [0u16; BLOCK_MAX_KMEANS_TEXELS];
    // Create the histogram of mismatch counts
    for i in 0..partitioning_count {
        let m = (mismatch_count[i] as usize).min(BLOCK_MAX_KMEANS_TEXELS - 1);
        mscount[m] = mscount[m].wrapping_add(1);
    }
    // Create a running sum from the histogram array
    // Cells store previous values only; i.e. exclude self after sum
    let mut sum = 0u16;
    for i in 0..texel_count.min(BLOCK_MAX_KMEANS_TEXELS) {
        let cnt = mscount[i];
        mscount[i] = sum;
        sum = sum.wrapping_add(cnt);
    }
    // Use the running sum as the index, incrementing after read to allow sequential entries
    // with the same count
    for i in 0..partitioning_count {
        let m = (mismatch_count[i] as usize).min(BLOCK_MAX_KMEANS_TEXELS - 1);
        let idx = mscount[m] as usize;
        mscount[m] = mscount[m].wrapping_add(1);
        if let Some(o) = partition_ordering.get_mut(idx) {
            *o = i as u16;
        }
    }
    partitioning_count
}
}

isa_fn! {
/// `compute_kmeans_partition_ordering`.
fn compute_kmeans_partition_ordering(
    bsd: &BlockSizeDescriptor,
    blk: &ImageBlock,
    partition_count: usize,
    partition_ordering: &mut [u16; BLOCK_MAX_PARTITIONINGS],
) -> usize {
    let mut cluster_centers = [Vfloat4::zero(); 4];
    let mut texel_partitions = [0u8; BLOCK_MAX_TEXELS];
    let texel_count = bsd.texel_count as usize;

    // Use three passes of k-means clustering to partition the block data
    for i in 0..3 {
        if i == 0 {
            kmeans_init(blk, texel_count, partition_count, &mut cluster_centers);
        } else {
            kmeans_update(
                blk,
                texel_count,
                partition_count,
                &mut cluster_centers,
                &texel_partitions,
            );
        }
        kmeans_assign(
            blk,
            texel_count,
            partition_count,
            &cluster_centers,
            &mut texel_partitions,
        );
    }

    // Construct the block bitmaps of texel assignments to each partition
    let mut bitmaps = [0u64; BLOCK_MAX_PARTITIONS];
    let texels_to_process = texel_count.min(BLOCK_MAX_KMEANS_TEXELS);
    for i in 0..texels_to_process {
        let idx = bsd.kmeans_texels[i] as usize;
        bitmaps[texel_partitions[idx] as usize & 3] |= 1u64 << i;
    }

    // Count the mismatch between the block and the format's partition tables
    let mut mismatch_counts = [0u8; BLOCK_MAX_PARTITIONINGS];
    count_partition_mismatch_bits(bsd, partition_count, &bitmaps, &mut mismatch_counts);

    // Sort the partitions based on the number of mismatched bits
    get_partition_ordering_by_mismatch_bits(
        texels_to_process,
        bsd.partitioning_count_selected[partition_count - 1] as usize,
        &mismatch_counts,
        partition_ordering,
    )
}
}

isa_fn! {
/// `insert_result`.
fn insert_result(
    max_values: usize,
    this_error: f32,
    this_partition: u32,
    best_errors: &mut [f32],
    best_partitions: &mut [u32],
) {
    // Don't bother searching if the current worst error beats the new error
    if this_error >= best_errors[max_values - 1] {
        return;
    }
    // Else insert into the list in error-order
    for i in 0..max_values {
        // Existing result is better - move on ...
        if this_error > best_errors[i] {
            continue;
        }
        // Move existing results down one
        let mut j = max_values - 1;
        while j > i {
            best_errors[j] = best_errors[j - 1];
            best_partitions[j] = best_partitions[j - 1];
            j -= 1;
        }
        // Insert new result
        best_errors[i] = this_error;
        best_partitions[i] = this_partition;
        break;
    }
}
}

isa_fn! {
/// `find_best_partition_candidates`: returns the number of candidates written to
/// `best_partitions` (partition indices, not packed indices).
pub fn find_best_partition_candidates(
    bsd: &BlockSizeDescriptor,
    blk: &ImageBlock,
    partition_count: u32,
    partition_search_limit: u32,
    best_partitions: &mut [u32; TUNE_MAX_PARTITIONING_CANDIDATES],
    requested_candidates: u32,
) -> u32 {
    // Constant used to estimate quantization error for a given partitioning; the optimal value
    // for this depends on bitrate. These values have been determined empirically.
    let texels_per_block = bsd.texel_count as u32;
    let mut weight_imprecision_estim = 0.055f32;
    if texels_per_block <= 20 {
        weight_imprecision_estim = 0.03;
    } else if texels_per_block <= 31 {
        weight_imprecision_estim = 0.04;
    } else if texels_per_block <= 41 {
        weight_imprecision_estim = 0.05;
    }
    weight_imprecision_estim *= weight_imprecision_estim;

    let pc = partition_count as usize;
    let mut partition_sequence = [0u16; BLOCK_MAX_PARTITIONINGS];
    let sequence_len =
        compute_kmeans_partition_ordering(bsd, blk, pc, &mut partition_sequence) as u32;
    let partition_search_limit = partition_search_limit.min(sequence_len) as usize;
    let requested_candidates = (partition_search_limit as u32)
        .min(requested_candidates)
        .min(TUNE_MAX_PARTITIONING_CANDIDATES as u32) as usize;

    let uses_alpha = !blk.is_constant_channel(3);

    // Partitioning errors assuming uncorrelated-chrominance endpoints
    let mut uncor_best_errors = [0.0f32; TUNE_MAX_PARTITIONING_CANDIDATES];
    let mut uncor_best_partitions = [0u32; TUNE_MAX_PARTITIONING_CANDIDATES];
    // Partitioning errors assuming same-chrominance endpoints
    let mut samec_best_errors = [0.0f32; TUNE_MAX_PARTITIONING_CANDIDATES];
    let mut samec_best_partitions = [0u32; TUNE_MAX_PARTITIONING_CANDIDATES];
    for i in 0..requested_candidates {
        uncor_best_errors[i] = ERROR_CALC_DEFAULT;
        samec_best_errors[i] = ERROR_CALC_DEFAULT;
    }

    // The AVX2 build (`ASTCENC_HAS_NATIVE_QUAD_OPS`) scores two candidates per vector, one per
    // quad. The candidate list is padded to the batch width by repeating the last candidate;
    // padded results are not ranked.
    if partition_search_limit > 0 {
        let padded_limit = partition_search_limit.next_multiple_of(SIMD_BATCH_WIDTH);
        for i in partition_search_limit..padded_limit {
            partition_sequence[i] = partition_sequence[partition_search_limit - 1];
        }
    }
    let mut i = 0;
    while i < partition_search_limit {
        let candidates = [
            bsd.get_raw_partition_info(partition_count, partition_sequence[i] as u32),
            bsd.get_raw_partition_info(partition_count, partition_sequence[i + 1] as u32),
        ];
        let mut uncor_errors = [0.0f32; 2];
        let mut samec_errors = [0.0f32; 2];
        let w = weight_imprecision_estim;
        match (uses_alpha, blk.grayscale) {
            (true, true) => compute_partition_errors_luminance_alpha_batch(
                candidates,
                blk,
                w,
                &mut uncor_errors,
                &mut samec_errors,
            ),
            (true, false) => compute_partition_errors_rgba_batch(
                candidates,
                blk,
                w,
                &mut uncor_errors,
                &mut samec_errors,
            ),
            (false, true) => compute_partition_errors_luminance_batch(
                candidates,
                blk,
                w,
                &mut uncor_errors,
                &mut samec_errors,
            ),
            (false, false) => compute_partition_errors_rgb_batch(
                candidates,
                blk,
                w,
                &mut uncor_errors,
                &mut samec_errors,
            ),
        }

        // Keep candidate insertion order and exclude padded results from the rankings
        let active_count = SIMD_BATCH_WIDTH.min(partition_search_limit - i);
        for c in 0..active_count {
            let partition = partition_sequence[i + c] as u32;
            insert_result(
                requested_candidates,
                uncor_errors[c],
                partition,
                &mut uncor_best_errors,
                &mut uncor_best_partitions,
            );
            insert_result(
                requested_candidates,
                samec_errors[c],
                partition,
                &mut samec_best_errors,
                &mut samec_best_partitions,
            );
        }
        i += SIMD_BATCH_WIDTH;
    }

    let mut interleave = [0u32; 2 * TUNE_MAX_PARTITIONING_CANDIDATES];
    for i in 0..requested_candidates {
        interleave[2 * i] = bsd
            .get_raw_partition_info(partition_count, uncor_best_partitions[i])
            .partition_index as u32;
        interleave[2 * i + 1] = bsd
            .get_raw_partition_info(partition_count, samec_best_partitions[i])
            .partition_index as u32;
    }

    let mut bitmasks = [0u64; 1024 / 64];
    let mut emitted = 0usize;
    for i in 0..requested_candidates * 2 {
        let partition = interleave[i];
        let word = (partition / 64) as usize & 15;
        let bit = partition % 64;
        let written = bitmasks[word] & (1u64 << bit) != 0;
        if !written {
            best_partitions[emitted] = partition;
            bitmasks[word] |= 1u64 << bit;
            emitted += 1;
            if emitted == requested_candidates {
                break;
            }
        }
    }
    emitted as u32
}
}

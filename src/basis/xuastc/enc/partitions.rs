// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Port of the tail of `basisu_astc_ldr_common.cpp` (from `partition_pattern_vec`):
//! partition pattern vectors, the LSH partition pattern map and `partitions_data`.
//!
//! `vp_tree` is not ported: the encoder builds `partitions_data` with `init_vp_tree=false`
//! (`BASISU_USE_LSH2/3` = 1), so `vp_tree::find_nearest` is only reachable from code compiled
//! out by those defines.
//!
//! Also holds the precomputed ASTC texel partition table (`g_texel_partitions`).
#![allow(clippy::needless_range_loop)]

use super::rand::BasisuRand;
use crate::astc::helpers::{self as ah, MAX_BLOCK_PIXELS, NUM_PARTITION_PATTERNS};
use std::collections::HashSet;
use std::sync::OnceLock;

/// `NUM_PART3_MAPPINGS`.
pub(crate) const NUM_PART3_MAPPINGS: usize = 6;

/// `g_part3_mapping`.
pub(crate) const PART3_MAPPING: [[u8; 3]; NUM_PART3_MAPPINGS] = [
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
    [0, 2, 1],
    [1, 0, 2],
    [2, 1, 0],
];

/// `g_texel_partitions[seed][y][x]`: low nibble = small blocks (< 31 texels, only valid for
/// x <= 6, y <= 5), high nibble = large blocks; each nibble = 2-subset | (3-subset << 2).
pub(crate) fn texel_partitions() -> &'static [[u8; 144]] {
    static T: OnceLock<Vec<[u8; 144]>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = vec![[0u8; 144]; NUM_PARTITION_PATTERNS as usize];
        for (seed, pat) in t.iter_mut().enumerate() {
            for y in 0..12u32 {
                for x in 0..12u32 {
                    let mut k = 0u32;
                    if x <= 6 && y <= 5 {
                        let v2 = ah::compute_texel_partition(seed as u32, x, y, 0, 2, true);
                        let v3 = ah::compute_texel_partition(seed as u32, x, y, 0, 3, true);
                        k |= v2 | (v3 << 2);
                    }
                    let v2 = ah::compute_texel_partition(seed as u32, x, y, 0, 2, false);
                    let v3 = ah::compute_texel_partition(seed as u32, x, y, 0, 3, false);
                    k |= (v2 | (v3 << 2)) << 4;
                    pat[(x + y * 12) as usize] = k as u8;
                }
            }
        }
        t
    })
}

/// `astc_helpers::get_precomputed_texel_partition` (2 or 3 subsets only).
#[inline]
pub(crate) fn get_precomputed_texel_partition(
    block_width: u32,
    block_height: u32,
    seed: u32,
    x: u32,
    y: u32,
    subsets: u32,
) -> u32 {
    let v = u32::from(texel_partitions()[(seed & 1023) as usize][(x + y * 12) as usize]);
    let mut shift = if subsets == 3 { 2 } else { 0 };
    shift += u32::from(block_width * block_height >= 31) * 4;
    (v >> shift) & 3
}

/// `partition_pattern_vec`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PartitionPatternVec {
    pub width: u32,
    pub height: u32,
    pub parts: [u8; MAX_BLOCK_PIXELS],
}

impl Default for PartitionPatternVec {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            parts: [0; MAX_BLOCK_PIXELS],
        }
    }
}

/// Compares only the used part of `parts` (as the C++ `operator==`).
impl PartialEq for PartitionPatternVec {
    fn eq(&self, rhs: &Self) -> bool {
        self.width == rhs.width
            && self.height == rhs.height
            && self.parts[..self.get_total()] == rhs.parts[..rhs.get_total()]
    }
}

impl Eq for PartitionPatternVec {}

impl std::hash::Hash for PartitionPatternVec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.width.hash(state);
        self.height.hash(state);
        self.parts[..self.get_total()].hash(state);
    }
}

#[inline]
fn squarei(v: i32) -> i32 {
    v * v
}

impl PartitionPatternVec {
    /// `partition_pattern_vec(width, height, pParts)`.
    pub fn new(width: u32, height: u32, parts: Option<&[u8]>) -> Self {
        let mut v = Self::default();
        v.init(width, height, parts);
        v
    }

    /// `init(width, height, pParts)`.
    pub fn init(&mut self, width: u32, height: u32, parts: Option<&[u8]>) {
        self.width = width;
        self.height = height;
        if let Some(p) = parts {
            let n = self.get_total();
            self.parts[..n].copy_from_slice(&p[..n]);
        }
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    #[inline]
    pub fn get_total(&self) -> usize {
        ((self.width * self.height) as usize).min(MAX_BLOCK_PIXELS)
    }

    /// `operator()(x, y)`.
    #[inline]
    pub fn at(&self, x: u32, y: u32) -> u8 {
        self.parts[(x + y * self.width) as usize]
    }

    /// `get_squared_distance(other)`.
    pub fn get_squared_distance(&self, other: &Self) -> i32 {
        let total = self.get_total();
        let d = |i: usize| squarei(self.parts[i] as i32 - other.parts[i] as i32);
        let (mut t0, mut t1, mut t2, mut t3) = (0, 0, 0, 0);
        let mut idx = 0;
        while idx + 4 <= total {
            t0 += d(idx);
            t1 += d(idx + 1);
            t2 += d(idx + 2);
            t3 += d(idx + 3);
            idx += 4;
        }
        while idx < total {
            t0 += d(idx);
            idx += 1;
        }
        t0 + t1 + t2 + t3
    }

    /// `get_squared_distance_2subsets(other)`.
    pub fn get_squared_distance_2subsets(&self, other: &Self) -> i32 {
        let total = self.get_total();
        let d = |i: usize| (self.parts[i] ^ other.parts[i]) as i32;
        let (mut t0, mut t1, mut t2, mut t3) = (0, 0, 0, 0);
        let mut idx = 0;
        while idx + 4 <= total {
            t0 += d(idx);
            t1 += d(idx + 1);
            t2 += d(idx + 2);
            t3 += d(idx + 3);
            idx += 4;
        }
        while idx < total {
            t0 += squarei(d(idx));
            idx += 1;
        }
        t0 + t1 + t2 + t3
    }

    /// `get_squared_distance(other, num_subsets, other_perm_index)`.
    pub fn get_squared_distance_perm(
        &self,
        other: &Self,
        num_subsets: u32,
        other_perm_index: u32,
    ) -> i32 {
        let mut total_dist = 0;
        for i in 0..self.get_total() {
            let mut p = other.parts[i] as usize;
            if num_subsets == 2 {
                p ^= other_perm_index as usize;
            } else {
                p = PART3_MAPPING[other_perm_index as usize % NUM_PART3_MAPPINGS][p % 3] as usize;
            }
            total_dist += squarei(self.parts[i] as i32 - p as i32);
        }
        total_dist
    }

    /// `get_distance(other)`.
    pub fn get_distance(&self, other: &Self) -> f32 {
        (self.get_squared_distance(other) as f32).sqrt()
    }

    /// `get_permuted2(permute_index)`.
    pub fn get_permuted2(&self, permute_index: u32) -> Self {
        let mut res = Self::new(self.width, self.height, None);
        for i in 0..self.get_total() {
            res.parts[i] = self.parts[i] ^ permute_index as u8;
        }
        res
    }

    /// `get_permuted3(permute_index)`.
    pub fn get_permuted3(&self, permute_index: u32) -> Self {
        let mut res = Self::new(self.width, self.height, None);
        let map = &PART3_MAPPING[permute_index as usize % NUM_PART3_MAPPINGS];
        for i in 0..self.get_total() {
            res.parts[i] = map[(self.parts[i] % 3) as usize];
        }
        res
    }

    /// `get_canonicalized()`.
    pub fn get_canonicalized(&self) -> Self {
        let mut res = Self::new(self.width, self.height, None);
        let mut new_labels = [-1i32; 4];
        let mut next_index = 0;
        for i in 0..self.get_total() {
            let p = (self.parts[i] & 3) as usize;
            if new_labels[p] == -1 {
                new_labels[p] = next_index;
                next_index += 1;
            }
            res.parts[i] = new_labels[p] as u8;
        }
        res
    }
}

const LSH_MAX_SAMPLES: usize = 10;
const LSH_MAX_HASH_TABS: usize = 16;

/// `partition_lsh_map`.
#[derive(Clone, Debug, Default)]
pub(crate) struct PartitionLshMap {
    num_subsets: u32,
    num_samples: u32,
    num_hash_tabs: u32,
    sample_indices: [[u8; LSH_MAX_SAMPLES]; LSH_MAX_HASH_TABS],
    hash_tabs: Vec<Vec<Vec<u32>>>,
}

impl PartitionLshMap {
    /// `is_valid()`.
    pub fn is_valid(&self) -> bool {
        self.num_subsets != 0
    }

    /// `init(num_subsets, n, pUnique_pats)`.
    pub fn init(&mut self, num_subsets: u32, unique_pats: &[PartitionPatternVec]) {
        let Some(first) = unique_pats.first() else {
            return;
        };
        let total_texels = first.width * first.height;
        self.num_subsets = num_subsets;
        self.num_samples = if num_subsets == 2 { 6 } else { 5 };
        self.num_hash_tabs = 12;
        let hash_key_shift = if num_subsets == 2 { 1 } else { 2 };
        let hash_tab_size = 1usize << (self.num_samples * hash_key_shift);

        let mut rnd = BasisuRand::with_seed(33155);
        self.sample_indices = [[0; LSH_MAX_SAMPLES]; LSH_MAX_HASH_TABS];
        self.hash_tabs = vec![Vec::new(); LSH_MAX_HASH_TABS];

        for h in 0..self.num_hash_tabs as usize {
            for s in 0..self.num_samples as usize {
                for _t in 0..4 {
                    let rnd_index = rnd.irand(0, total_texels as i32 - 1) as u8;
                    self.sample_indices[h][s] = rnd_index;
                    let mut b = 0;
                    while b < s {
                        if rnd_index == self.sample_indices[h][b] {
                            break;
                        }
                        b += 1;
                    }
                    if b == s {
                        break;
                    }
                }
            }
            self.hash_tabs[h] = vec![Vec::new(); hash_tab_size];
        }

        let max_perm_index = if num_subsets == 2 { 1 } else { 5 };
        for (unique_pat_index, pat) in unique_pats.iter().enumerate() {
            for perm_index in 0..=max_perm_index {
                let pv = if num_subsets == 2 {
                    pat.get_permuted2(perm_index)
                } else {
                    pat.get_permuted3(perm_index)
                };
                for hti in 0..self.num_hash_tabs as usize {
                    let mut hash_key = 0usize;
                    for si in 0..self.num_samples as usize {
                        hash_key <<= hash_key_shift;
                        hash_key |= pv.parts[self.sample_indices[hti][si] as usize] as usize;
                    }
                    if let Some(b) = self.hash_tabs[hti].get_mut(hash_key) {
                        b.push(unique_pat_index as u32);
                    }
                }
            }
        }
    }

    /// `find(desired_pat, pResults, max_results, filter_flag)`: writes up to `max_results`
    /// candidate unique pattern indices into `results` (sorted, deduplicated) and returns the
    /// count.
    pub fn find(
        &self,
        desired_pat: &PartitionPatternVec,
        results: &mut [u32],
        max_results: u32,
        filter_flag: bool,
    ) -> u32 {
        let max_results = (max_results as usize).min(results.len());
        let mut total = 0usize;
        let hash_key_shift = if self.num_subsets == 2 { 1 } else { 2 };
        for hti in 0..self.num_hash_tabs as usize {
            let mut hash_key = 0usize;
            for si in 0..self.num_samples as usize {
                hash_key <<= hash_key_shift;
                hash_key |= desired_pat.parts[self.sample_indices[hti][si] as usize] as usize;
            }
            let h_tab: &[u32] = self
                .hash_tabs
                .get(hti)
                .and_then(|t| t.get(hash_key))
                .map_or(&[], |v| v.as_slice());
            let room_remaining = max_results - total;
            if room_remaining == 0 {
                break;
            }
            let num_to_copy = h_tab.len().min(room_remaining);
            results[total..total + num_to_copy].copy_from_slice(&h_tab[..num_to_copy]);
            total += num_to_copy;
            if total >= max_results {
                break;
            }
        }
        if total == 0 {
            return 0;
        }
        // Integer keys: any sort gives the same result as std::sort.
        results[..total].sort_unstable();
        let num_unique = if filter_flag {
            unique_duplicates_only(&mut results[..total])
        } else {
            let mut out = 1;
            for i in 1..total {
                if results[i] != results[out - 1] {
                    results[out] = results[i];
                    out += 1;
                }
            }
            out
        };
        num_unique as u32
    }
}

/// `unique_duplicates_only`.
fn unique_duplicates_only(results: &mut [u32]) -> usize {
    let total = results.len();
    if total < 2 {
        return 0;
    }
    let (mut i, mut out) = (0, 0);
    while i < total {
        let mut j = i + 1;
        while j < total && results[j] == results[i] {
            j += 1;
        }
        if j - i >= 2 {
            results[out] = results[i];
            out += 1;
        }
        i = j;
    }
    out
}

/// `partitions_data`.
#[derive(Clone, Debug)]
pub(crate) struct PartitionsData {
    pub width: u32,
    pub height: u32,
    pub num_partitions: u32,
    /// Indexed by unique index, NOT the 10-bit ASTC seed.
    pub partition_pats: Vec<PartitionPatternVec>,
    /// `m_partition_pat_histograms[unique].m_hist`.
    pub partition_pat_histograms: Vec<[u8; 4]>,
    pub part_seed_to_unique_index: [i16; NUM_PARTITION_PATTERNS as usize],
    pub unique_index_to_part_seed: [i16; NUM_PARTITION_PATTERNS as usize],
    pub total_unique_patterns: u32,
    pub part_lhs_map: PartitionLshMap,
}

impl Default for PartitionsData {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            num_partitions: 0,
            partition_pats: Vec::new(),
            partition_pat_histograms: Vec::new(),
            part_seed_to_unique_index: [-1; NUM_PARTITION_PATTERNS as usize],
            unique_index_to_part_seed: [-1; NUM_PARTITION_PATTERNS as usize],
            total_unique_patterns: 0,
            part_lhs_map: PartitionLshMap::default(),
        }
    }
}

impl PartitionsData {
    /// `init(num_partitions, block_width, block_height, init_vp_tree=false, init_lsh)`.
    pub fn new(num_partitions: u32, block_width: u32, block_height: u32, init_lsh: bool) -> Self {
        let mut d = Self {
            width: block_width,
            height: block_height,
            num_partitions,
            ..Default::default()
        };
        let mut part_hash: HashSet<PartitionPatternVec> = HashSet::with_capacity(1024);
        for seed_index in 0..NUM_PARTITION_PATTERNS {
            let mut pat = PartitionPatternVec::new(block_width, block_height, None);
            let mut part_hist = [0u32; 4];
            for y in 0..block_height {
                for x in 0..block_width {
                    let p = get_precomputed_texel_partition(
                        block_width,
                        block_height,
                        seed_index,
                        x,
                        y,
                        num_partitions,
                    );
                    pat.parts[(x + y * block_width) as usize] = p as u8;
                    part_hist[p as usize] += 1;
                }
            }
            if part_hist
                .iter()
                .take(num_partitions as usize)
                .any(|&h| h == 0)
            {
                continue;
            }
            let std_pat = pat.get_canonicalized();
            if !part_hash.insert(std_pat) {
                continue;
            }
            d.partition_pat_histograms.push(part_hist.map(|h| h as u8));
            let u = d.total_unique_patterns;
            d.part_seed_to_unique_index[seed_index as usize] = u as i16;
            d.unique_index_to_part_seed[u as usize] = seed_index as i16;
            d.partition_pats.push(pat);
            d.total_unique_patterns += 1;
        }
        if init_lsh {
            d.part_lhs_map.init(num_partitions, &d.partition_pats);
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn irand_matches_libstdcxx() {
        // From a g++ 13 driver: uniform_int_distribution<int>(0, n - 1) over mt19937(33155).
        let expect: [(i32, u64); 5] = [
            (16, 4040397301264271241),
            (25, 13887852604326815178),
            (36, 2628858020304750182),
            (64, 11226819702709491450),
            (144, 11934826807512804516),
        ];
        for (n, hash) in expect {
            let mut r = BasisuRand::with_seed(33155);
            let mut h = 0u64;
            for _ in 0..300 {
                h = h.wrapping_mul(31).wrapping_add(r.irand(0, n - 1) as u64);
            }
            assert_eq!(h, hash, "n={n}");
        }
    }

    #[test]
    fn unique_counts_match_decoder_tables() {
        for (i, b) in crate::basis::file::BASIS_ASTC_ORDER.iter().enumerate() {
            let (w, h) = b.dim();
            for parts in 2..=3 {
                let d = PartitionsData::new(parts, w, h, false);
                assert_eq!(
                    d.total_unique_patterns,
                    crate::basis::xuastc::total_unique_patterns(i, parts)
                );
                for u in 0..d.total_unique_patterns {
                    assert_eq!(
                        d.unique_index_to_part_seed[u as usize] as u16,
                        crate::basis::xuastc::part_seed(i, parts, u).unwrap_or(0xFFFF)
                    );
                }
            }
        }
    }
}

// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from Basis Universal and modified; see NOTICE.md.
//! Deterministic container ports whose iteration / pop order is observable in the encoder
//! output: `basisu::hash_map` (`transcoder/basisu_containers.h`: open addressing, Fibonacci
//! hashing, backwards linear probing, iteration in slot order) and libstdc++'s
//! `std::priority_queue` (`std::push_heap` / `std::pop_heap`, `bits/stl_heap.h`).

/// Key hashing / equality as the C++ key type defines them (`operator size_t()` and
/// `operator==`).
pub(crate) trait BuKey {
    /// `operator size_t()` (64-bit `size_t`).
    fn bu_hash(&self) -> u64;
    /// `operator==`.
    fn bu_eq(&self, other: &Self) -> bool;
}

const MIN_HASH_SIZE: usize = 4;
const SIZE_T_BITS: u32 = 64;

/// `basisu::hash_map<K, V>` without `erase` (unused by the encoder). Entries live in a dense
/// vector (stable indices, insertion order); `slots` is the open addressing table (entry
/// index + 1, 0 = empty). Slot placement, growth and iteration order match the C++ exactly.
pub(crate) struct BuHashMap<K, V> {
    slots: Vec<u32>,
    entries: Vec<(K, V)>,
    grow_threshold: usize,
    hash_shift: u32,
}

impl<K: BuKey, V> Default for BuHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: BuKey, V> BuHashMap<K, V> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            entries: Vec::new(),
            grow_threshold: 0,
            hash_shift: SIZE_T_BITS,
        }
    }

    /// `get_table_size`.
    pub fn table_size(&self) -> usize {
        self.slots.len()
    }

    /// `size`.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `clear`: frees the table.
    pub fn clear(&mut self) {
        self.slots = Vec::new();
        self.entries.clear();
        self.hash_shift = SIZE_T_BITS;
        self.grow_threshold = 0;
    }

    /// `reset`: empties the container, keeps the table size.
    pub fn reset(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        self.slots.iter_mut().for_each(|s| *s = 0);
        self.entries.clear();
    }

    /// `reserve`.
    pub fn reserve(&mut self, new_capacity: usize) {
        if new_capacity == 0 {
            return;
        }
        let new_hash_size = (new_capacity * 2).next_power_of_two().max(MIN_HASH_SIZE);
        if new_hash_size <= self.slots.len() {
            return;
        }
        self.rehash(new_hash_size);
    }

    /// `swap`.
    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(self, other);
    }

    fn hash_key(&self, k: &K) -> usize {
        (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(k.bu_hash()) >> self.hash_shift) as usize
    }

    fn try_grow(&mut self) {
        let n = (self.slots.len() * 2)
            .next_power_of_two()
            .max(MIN_HASH_SIZE);
        self.rehash(n);
    }

    /// `rehash`: re-inserts the entries in old slot order (`move_into`).
    fn rehash(&mut self, new_hash_size: usize) {
        if new_hash_size == self.slots.len() || new_hash_size < self.entries.len() {
            return;
        }
        let old = std::mem::replace(&mut self.slots, vec![0; new_hash_size]);
        self.hash_shift = SIZE_T_BITS - new_hash_size.trailing_zeros();
        let mask = new_hash_size - 1;
        for &s in old.iter().filter(|&&s| s != 0) {
            let mut index = self.hash_key(&self.entries[(s - 1) as usize].0);
            while self.slots[index] != 0 {
                index = index.wrapping_sub(1) & mask;
            }
            self.slots[index] = s;
        }
        self.grow_threshold = new_hash_size >> 1;
    }

    /// `insert_no_grow`: `Ok((entry, inserted))`, `Err(v)` if the table must grow.
    fn insert_no_grow(&mut self, k: K, v: V) -> Result<(usize, bool), (K, V)> {
        if self.slots.is_empty() {
            return Err((k, v));
        }
        let mask = self.slots.len() - 1;
        let mut index = self.hash_key(&k);
        let s = self.slots[index];
        if s != 0 {
            if self.entries[(s - 1) as usize].0.bu_eq(&k) {
                return Ok(((s - 1) as usize, false));
            }
            let orig_index = index;
            loop {
                index = index.wrapping_sub(1) & mask;
                if index == orig_index {
                    return Err((k, v));
                }
                let s = self.slots[index];
                if s == 0 {
                    break;
                }
                if self.entries[(s - 1) as usize].0.bu_eq(&k) {
                    return Ok(((s - 1) as usize, false));
                }
            }
        }
        if self.entries.len() >= self.grow_threshold {
            return Err((k, v));
        }
        self.entries.push((k, v));
        self.slots[index] = self.entries.len() as u32;
        Ok((self.entries.len() - 1, true))
    }

    /// `insert`: returns (entry index, true if newly inserted). An existing key keeps its value.
    pub fn insert(&mut self, k: K, v: V) -> (usize, bool) {
        match self.insert_no_grow(k, v) {
            Ok(r) => r,
            Err((k, v)) => {
                self.try_grow();
                match self.insert_no_grow(k, v) {
                    Ok(r) => r,
                    // Unreachable: a grown table always has room.
                    Err(_) => (usize::MAX, false),
                }
            }
        }
    }

    /// `find`: entry index.
    pub fn find(&self, k: &K) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut index = self.hash_key(k);
        let s = self.slots[index];
        if s == 0 {
            return None;
        }
        if self.entries[(s - 1) as usize].0.bu_eq(k) {
            return Some((s - 1) as usize);
        }
        let orig_index = index;
        loop {
            index = index.wrapping_sub(1) & mask;
            if index == orig_index {
                return None;
            }
            let s = self.slots[index];
            if s == 0 {
                return None;
            }
            if self.entries[(s - 1) as usize].0.bu_eq(k) {
                return Some((s - 1) as usize);
            }
        }
    }

    /// Entry indices in iteration (slot) order.
    pub fn iter_order(&self, out: &mut Vec<u32>) {
        out.clear();
        out.extend(self.slots.iter().filter(|&&s| s != 0).map(|&s| s - 1));
    }

    pub fn entry(&self, i: usize) -> &(K, V) {
        &self.entries[i]
    }

    pub fn entry_mut(&mut self, i: usize) -> &mut (K, V) {
        &mut self.entries[i]
    }
}

/// `std::priority_queue<T>` (libstdc++, `std::less<T>`, max-heap) with the exact
/// `push_heap` / `pop_heap` element movements.
pub(crate) struct StdPriorityQueue<T> {
    c: Vec<T>,
}

impl<T: Copy> Default for StdPriorityQueue<T> {
    fn default() -> Self {
        Self { c: Vec::new() }
    }
}

impl<T: Copy> StdPriorityQueue<T> {
    pub fn len(&self) -> usize {
        self.c.len()
    }

    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }

    pub fn clear(&mut self) {
        self.c.clear();
    }

    pub fn top(&self) -> Option<&T> {
        self.c.first()
    }

    /// `push`: `push_back` + `std::push_heap`.
    pub fn push<F: Fn(&T, &T) -> bool>(&mut self, x: T, less: F) {
        self.c.push(x);
        let n = self.c.len();
        push_heap_hole(&mut self.c, n - 1, 0, x, &less);
    }

    /// `pop`: `std::pop_heap` + `pop_back`.
    pub fn pop<F: Fn(&T, &T) -> bool>(&mut self, less: F) {
        let n = self.c.len();
        if n > 1 {
            // __pop_heap(first, last - 1, last - 1)
            let last = n - 1;
            let value = self.c[last];
            self.c[last] = self.c[0];
            adjust_heap(&mut self.c, 0, last, value, &less);
        }
        self.c.pop();
    }
}

/// libstdc++ `std::__push_heap`.
fn push_heap_hole<T: Copy, F: Fn(&T, &T) -> bool>(
    v: &mut [T],
    mut hole: usize,
    top: usize,
    value: T,
    less: &F,
) {
    while hole > top {
        let parent = (hole - 1) / 2;
        if !less(&v[parent], &value) {
            break;
        }
        v[hole] = v[parent];
        hole = parent;
    }
    v[hole] = value;
}

/// libstdc++ `std::__adjust_heap`.
fn adjust_heap<T: Copy, F: Fn(&T, &T) -> bool>(
    v: &mut [T],
    mut hole: usize,
    len: usize,
    value: T,
    less: &F,
) {
    let top = hole;
    let mut second = hole;
    while len >= 1 && second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if less(&v[second], &v[second - 1]) {
            second -= 1;
        }
        v[hole] = v[second];
        hole = second;
    }
    if (len & 1) == 0 && len >= 2 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[hole] = v[second - 1];
        hole = second - 1;
    }
    push_heap_hole(v, hole, top, value, less);
}

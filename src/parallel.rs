//! Minimal scoped data-parallel helper on `std::thread` (no extra dependencies).
//!
//! Work is handed out in groups of units from a shared iterator; the calling thread
//! works too, so a failed thread spawn (e.g. on wasm) only means less parallelism.

use std::sync::Mutex;

/// Below this many pixels the work runs on the calling thread.
pub(crate) const MIN_PARALLEL_PIXELS: usize = 64 * 1024;

fn worker_count() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Splits `dst` into units of `unit_bytes` (the last may be shorter) and calls
/// `f(unit_index, unit)` for each, in parallel when `pixels` (the work size) is large enough.
pub(crate) fn for_each_unit<F>(dst: &mut [u8], unit_bytes: usize, pixels: usize, f: F)
where
    F: Fn(usize, &mut [u8]) + Sync,
{
    if unit_bytes == 0 || dst.is_empty() {
        return;
    }
    let units = dst.len().div_ceil(unit_bytes);
    let workers = worker_count().min(units);
    if workers <= 1 || pixels < MIN_PARALLEL_PIXELS {
        for (i, u) in dst.chunks_mut(unit_bytes).enumerate() {
            f(i, u);
        }
        return;
    }
    // A few groups per worker for load balancing.
    let per_group = units.div_ceil(workers * 4).max(1);
    let groups = Mutex::new(dst.chunks_mut(unit_bytes * per_group).enumerate());
    let run = || {
        loop {
            let next = match groups.lock() {
                Ok(mut it) => it.next(),
                Err(_) => None,
            };
            let Some((g, chunk)) = next else { break };
            for (i, u) in chunk.chunks_mut(unit_bytes).enumerate() {
                f(g * per_group + i, u);
            }
        }
    };
    std::thread::scope(|s| {
        for _ in 1..workers {
            // A spawn failure is fine: the remaining threads drain the queue.
            let _ = std::thread::Builder::new().spawn_scoped(s, run);
        }
        run();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_every_unit() {
        let mut buf = vec![0u8; 1_000_003];
        for_each_unit(&mut buf, 1000, MIN_PARALLEL_PIXELS, |i, u| {
            u.fill((i % 251) as u8)
        });
        for (i, b) in buf.iter().enumerate() {
            assert_eq!(*b, ((i / 1000) % 251) as u8);
        }
    }
}

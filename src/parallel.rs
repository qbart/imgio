//! Minimal scoped data-parallel helper on `std::thread` (no extra dependencies).
//!
//! Work is handed out in groups of units from a shared iterator; the calling thread
//! works too, so a failed thread spawn (e.g. on wasm) only means less parallelism.

use std::cell::Cell;
use std::sync::Mutex;

/// Below this many pixels the work runs on the calling thread.
pub(crate) const MIN_PARALLEL_PIXELS: usize = 64 * 1024;

thread_local! {
    /// Set while running an item of [`try_for_each_item`]: nested work stays on its thread.
    static NESTED: Cell<bool> = const { Cell::new(false) };
}

fn worker_count() -> usize {
    if NESTED.with(Cell::get) {
        return 1;
    }
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Calls `f` on every item, items in parallel (nested [`for_each_unit`] calls run inline);
/// returns the first error (other items still run).
pub(crate) fn try_for_each_item<T, F>(items: &mut [T], f: F) -> crate::Result<()>
where
    T: Send,
    F: Fn(&mut T) -> crate::Result<()> + Sync,
{
    let workers = worker_count().min(items.len());
    let queue = Mutex::new(items.iter_mut());
    let first_error = Mutex::new(None);
    let run = || {
        let was = NESTED.with(|n| n.replace(true));
        loop {
            let next = match queue.lock() {
                Ok(mut it) => it.next(),
                Err(_) => None,
            };
            let Some(item) = next else { break };
            if let Err(e) = f(item)
                && let Ok(mut slot) = first_error.lock()
            {
                slot.get_or_insert(e);
            }
        }
        NESTED.with(|n| n.set(was));
    };
    std::thread::scope(|s| {
        for _ in 1..workers {
            let _ = std::thread::Builder::new().spawn_scoped(s, run);
        }
        run();
    });
    match first_error.into_inner() {
        Ok(Some(e)) => Err(e),
        _ => Ok(()),
    }
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

/// Like [`for_each_unit`] for fallible work: returns the first error (other units still run).
pub(crate) fn try_for_each_unit<F>(
    dst: &mut [u8],
    unit_bytes: usize,
    pixels: usize,
    f: F,
) -> crate::Result<()>
where
    F: Fn(usize, &mut [u8]) -> crate::Result<()> + Sync,
{
    let first_error = Mutex::new(None);
    for_each_unit(dst, unit_bytes, pixels, |i, u| {
        if let Err(e) = f(i, u)
            && let Ok(mut slot) = first_error.lock()
        {
            slot.get_or_insert(e);
        }
    });
    match first_error.into_inner() {
        Ok(Some(e)) => Err(e),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_run_once_and_errors_propagate() {
        let mut items = [0u32; 100];
        assert!(
            try_for_each_item(&mut items, |v| {
                *v += 1;
                Ok(())
            })
            .is_ok()
        );
        assert!(items.iter().all(|&v| v == 1));
        let r = try_for_each_item(&mut items, |v| {
            if *v == 1 {
                Err(crate::Error::InvalidData)
            } else {
                Ok(())
            }
        });
        assert_eq!(r, Err(crate::Error::InvalidData));
    }

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

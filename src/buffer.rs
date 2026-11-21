//! Owned byte buffers backed by vas.

use crate::{Error, Result};

/// Growable byte buffer returned by allocating APIs. Derefs to `[u8]`.
pub type Buffer = vas::Vector<u8, vas::storage::Dynamic>;

/// A buffer of `len` zero bytes.
pub(crate) fn zeroed(len: usize) -> Result<Buffer> {
    let mut b = Buffer::default();
    b.resize(len, 0).map_err(|_| Error::OutOfMemory)?;
    Ok(b)
}

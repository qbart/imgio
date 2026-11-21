//! imgio: pure Rust image and GPU texture codecs.
//!
//! Every fallible operation returns [`Result`]; the library never panics.

mod error;
mod format;

pub use error::{Error, Result};
pub use format::Format;

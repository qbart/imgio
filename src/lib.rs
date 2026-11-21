//! imgio: pure Rust image and GPU texture codecs.
//!
//! Every fallible operation returns [`Result`]; the library never panics.

pub mod bc;
mod buffer;
mod error;
mod format;
pub mod image;
mod parallel;

pub use buffer::Buffer;
pub use error::{Error, Result};
pub use format::Format;
pub use image::Surface;

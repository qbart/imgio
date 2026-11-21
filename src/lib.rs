//! imgio: pure Rust image and GPU texture codecs.
//!
//! Every fallible operation returns [`Result`]; the library never panics.

mod astc;
pub mod basis;
mod bc;
mod block;
mod buffer;
pub mod convert;
mod decode;
mod error;
mod etc;
mod format;
pub mod half;
pub mod image;
pub mod ktx2;
mod parallel;

pub use buffer::Buffer;
pub use decode::{can_decode, decode, decode_to_buffer};
pub use error::{Error, Result};
pub use format::{AstcBlock, AstcKind, Format};
pub use image::Surface;

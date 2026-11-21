//! imgio: pure Rust image and GPU texture codecs.
//!
//! Every fallible operation returns [`Result`]; the library never panics. See the README for
//! an overview of every format, encoder and decoder.

// The README examples compile and run as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod astc;
pub mod astc_file;
#[allow(dead_code, unused_imports)]
pub(crate) mod astcenc;
mod atc;
pub mod basis;
mod bc;
mod bc6h_signed;
mod bc7e;
mod block;
mod buffer;
pub mod convert;
pub mod dds;
mod decode;
mod encode;
mod error;
mod etc;
mod etc1hq;
mod etc2modes;
pub mod file;
mod format;
mod fxt1;
pub mod gpu;
pub mod half;
pub mod image;
pub mod ktx1;
pub mod ktx2;
mod legacy_enc;
pub mod mip;
mod parallel;
mod pvrtc;
pub mod texture;

pub use buffer::Buffer;
pub use decode::{can_decode, decode, decode_to_buffer};
pub use encode::{EncodeOptions, Quality, can_encode, encode, encode_to_buffer};
pub use error::{Error, Result};
pub use format::{AstcBlock, AstcKind, Format};
pub use image::Surface;

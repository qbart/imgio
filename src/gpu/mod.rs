//! GPU texture upload.
//!
//! [`TextureSource`] abstracts the uploaded images: KTX2 files with plain payloads (Basis
//! Universal payloads must be transcoded first), DDS files and caller-held [`TextureData`]
//! (encoded or transcoded data). [`gl_format`] maps an imgio [`crate::Format`] to its OpenGL
//! formats.
//!
//! The uploads are opt-in bindings, the only non-Rust code imgio talks to:
//!
//! - feature `vulkan`: [`vulkan`], port of KTX-Software `vkloader.c` (via ktx-rs) on `ash`.
//! - feature `opengl`: [`gl`], port of KTX-Software `glloader.c` (via ktx-rs).
//!
//! Neither feature links a GPU library: the application's Vulkan / GL commands are looked up
//! at run time. Both are unavailable on wasm.

#[cfg(all(feature = "opengl", not(target_family = "wasm")))]
pub mod gl;
mod gl_format;
mod texture;
#[cfg(all(feature = "vulkan", not(target_family = "wasm")))]
pub mod vulkan;

pub use gl_format::{GlFormat, gl_format};
pub use texture::{LevelData, TextureData, TextureSource};

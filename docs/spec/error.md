# Error handling

- Every fallible API returns `imgio::Result<T>` = `Result<T, imgio::Error>`.
- No panics: `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!` are denied by clippy.
- `Error` is `#[non_exhaustive]`; variants carry only cheap context (sizes).

## Audit

- 2026-11-21: initial `Error` (InvalidData, InvalidArgument, BufferTooSmall, Unsupported).
- 2026-11-21: `Gpu(i32)`: failed Vulkan (`VkResult`, negative) or OpenGL (error enum) call.

# ImgIO for Rust language

imgio is a pure Rust library to convert images to GPU texture formats like BC1-BC7, UASTC, ASTC, RGBA (including HDR variants) etc., and in the future other basic formats like PNG, TGA, JPG etc.

The library should have an encode/decode interface that works on byte slices/arrays - whatever is more performant - so it can later be chained into a file write. If encode/decode requires options, it does not have to follow the same interface, as PNG vs BC7 encode options are vastly different.

If possible, encode/decode should be multithreaded.
No C/C++ bindings allowed, never. Pure Rust, otherwise it breaks the main goal of the library.

Especially focus on a full implementation of Basis Universal integration and transcoding for all types.
A Vulkan/OpenGL opt-in feature flag can be used as the only binding that allows the transcoding interface.

Use https://github.com/qbart/vas as the container library and ask me if anything is missing.

https://github.com/qbart/ktx-rs is my port of the KTX library and already contains a lot of groundwork that might be copied and modified to imgio's needs.

Reference for image encoding as well:
https://github.com/binomialLLC/basis_universal

## Rules

1. Keep the copy of ktx-rs in `./ref/ktx-rs` and keep `./ref` in gitignore. Also keep basis_universal at `./ref/basisu` to have the reference of encoders - pick basisu or ktx-rs as reference based on needs.
2. Use vas for containers so memory can be better managed.
3. Do not use `.unwrap` / `.expect`. The library should create its own error handling approach (`err, ok := Operation` like in Golang is preferred to simplify the API; runtime exceptions are not allowed).
4. Make sure to copy the test data like images to the library, version them in the repository and explicitly mark the license for the images like in ktx-rs.
5. Create images for test purposes so we can maintain them.
6. Maintain implementation audit logs at `./docs/spec/*.md`. First write the spec there, but be concise.
7. Maintain ROADMAP.md that tracks the progress of the whole project.
8. Only use the main branch.
9. `unsafe` is denied except in the GPU loaders (`vkloader`, `glloader`) and astcenc SIMD (ktx-rs `src/astcenc/enc/sse.rs` kernels, the instruction set dispatch in `src/astcenc/mod.rs`), or in places where it is necessary or speed matters.
10. Maintain a Makefile with phony rules like test, build, bench and crate publish.
11. Maintain legal notices about external use of libraries and test data.

## Loop iteration

1. Analyze what is missing from the required features.
2. Implement the API partially, provide atomic changes.
3. Commit after each change, use semantic git commits, keep the messages short, skip the annoying Co-Author lines.
4. Verify tests.
5. Fix what is needed until tests are passing.
6. Push changes.
7. Notify about the changes done in a max 5 sentence summary.

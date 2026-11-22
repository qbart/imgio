.PHONY: cli install build test test-gpu bench lint fmt check msrv package publish publish-dry ref

build:
	cargo build --release

# The `imgio` command line tool with every file format.
cli:
	cargo build --release --all-features --bin imgio

install:
	cargo install --path . --all-features --bin imgio

# Every image file format is a cargo feature: test them all (GPU tests skip without a device).
test:
	cargo test --release --all-features

# GPU upload tests (Vulkan via lavapipe or any device, GL via surfaceless EGL); they skip
# without a device / context.
test-gpu:
	cargo test --release --features vulkan,opengl --test gpu_vulkan --test gpu_gl
	cargo test --release --features vulkan,opengl --lib gpu

bench:
	cargo bench

lint:
	cargo clippy --all-targets --all-features -- -D warnings
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

fmt:
	cargo fmt --all

check:
	cargo fmt --all --check
	cargo clippy --all-targets --all-features -- -D warnings
	cargo test --release --all-features -q
	cargo build --release --no-default-features

# Minimum supported Rust version (Cargo.toml `rust-version`).
# `jpeg2000` (hayro-jpeg2000) needs Rust 1.92 and `avif` (avif-parse) 1.90.
msrv:
	cargo +1.88 build --release --features png,jpeg,gif,webp,exr,tiff,jxl,vdb,vulkan,opengl

# Files that go into the published crate.
package:
	cargo package --list

publish-dry: check
	cargo publish --dry-run

publish: check
	cargo publish

# Reference sources (gitignored): ktx-rs, Basis Universal and vas.
ref:
	mkdir -p ref
	test -d ref/ktx-rs || git clone --depth 1 https://github.com/qbart/ktx-rs ref/ktx-rs
	test -d ref/basisu || git clone --depth 1 https://github.com/BinomialLLC/basis_universal ref/basisu
	test -d ref/vas || git clone --depth 1 https://github.com/qbart/vas ref/vas
	test -d ref/astcenc || git clone --depth 1 https://github.com/ARM-software/astc-encoder ref/astcenc

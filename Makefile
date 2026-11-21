.PHONY: build test test-gpu bench lint fmt check publish publish-dry ref

build:
	cargo build --release

test:
	cargo test --release

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
	cargo test --release -q

publish-dry:
	cargo publish --dry-run

publish:
	cargo publish

# Reference sources (gitignored): ktx-rs, Basis Universal and vas.
ref:
	mkdir -p ref
	test -d ref/ktx-rs || git clone --depth 1 https://github.com/qbart/ktx-rs ref/ktx-rs
	test -d ref/basisu || git clone --depth 1 https://github.com/BinomialLLC/basis_universal ref/basisu
	test -d ref/vas || git clone --depth 1 https://github.com/qbart/vas ref/vas

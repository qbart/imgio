.PHONY: build test bench lint fmt check publish publish-dry ref

build:
	cargo build --release

test:
	cargo test --release

bench:
	cargo bench

lint:
	cargo clippy --all-targets --all-features -- -D warnings

fmt:
	cargo fmt --all

check: fmt lint test

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

//! Golden tests of the ported Basis Universal encoder (from ktx-rs `tests/basisu_*.rs`): every
//! case is compared with the output of the C++ original (`testdata/basisu_enc/*.txt`).

mod astc_hdr_enc;
mod backend;
mod bc7enc;
mod comp;
mod etc;
mod frontend;
mod hdr6x6_enc;
mod hdr6x6_parts;
mod huffman_enc;
mod uastc_enc;
mod uastc_hdr_4x4_enc;
mod uastc_rdo;
mod vq;

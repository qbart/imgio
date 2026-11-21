// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0
// Offline golden generator for imgio's XUBC7 (XBC7) encoder port (dev tool, not part of the
// crate): encodes raw RGBA8 images (testdata/images/raw/NAME_WxH.rgba) with the C++ Basis
// Universal v2.50 `xbc7::pack_image`, configured exactly like `basis_compressor::
// encode_slices_to_xubc7` (the basisu CLI `-xubc7` path), and prints
// "image params-id size fnv1a64" of every encoded slice stream.
// Derived images (also built by the Rust test): NAME:opaque (alpha forced to 255), and for the
// first file NAME:tiny (6x5 top-left crop) and NAME:tall (4x vertical tiling, 2 encoder stripes).
// Build: see build.sh (links the basisu_encoder static library; -ffp-contract=off, no fast-math).
#include "basisu_enc.h"
#include "basisu_xbc7_encode.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace basisu;

static uint64_t fnv(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

// Mirrors basis_compressor_params (-1 quality = CLI default), plus an optional exact stripe
// count stored straight into pack_options::m_num_stripes (low-level API, bypasses the clamp).
struct pset {
    const char* id;
    int quality, effort, rdo, stripes;
    int bc7e;   // -1 = bc7f, else bc7e_scalar level
    bool srgb;  // weights 9,11,1,11 + perceptual, else 1,1,1,1 linear
    int exact_stripes;
};

static const pset g_sets[] = {
    { "default", -1, 3, 0, 8, -1, true, 0 },
    { "q80", 80, 3, 0, 8, -1, true, 0 },
    { "q50e0", 50, 0, 0, 8, -1, true, 0 },
    { "q20e1", 20, 1, 0, 8, -1, true, 0 },
    { "q90e10", 90, 10, 0, 8, -1, true, 0 },
    { "q70e5r50", 70, 5, 50, 8, -1, true, 0 },
    { "q30e2r100lin", 30, 2, 100, 8, -1, false, 0 },
    { "bc7e2", -1, 3, 0, 8, 2, true, 0 },
    { "bc7e0q60lin", 60, 3, 0, 8, 0, false, 0 },
    { "bc7e4q85e6r20", 85, 6, 20, 8, 4, true, 0 },
    { "q75e4s3", 75, 4, 0, 8, -1, true, 3 },
    { "q100r30s4lin", -1, 3, 30, 8, -1, false, 4 },
};

static bool encode(const char* name, const image& img, const pset& s, job_pool& pool) {
    xbc7::pack_options o;
    o.m_dct_q = (s.quality < 0) ? 100 : clamp<int>(s.quality, 1, 100);
    o.set_rdo_level(s.rdo);
    const uint32_t w[4] = { s.srgb ? 9u : 1u, s.srgb ? 11u : 1u, 1u, s.srgb ? 11u : 1u };
    for (int i = 0; i < 4; i++) o.m_weights[i] = w[i];
    if ((o.m_dct_q >= 90) && (s.effort >= 3))
        o.m_optimize_weights_after_bc7f = true;
    if (s.effort <= 1)
        o.m_bc7_pack_flags = basist::bc7f::cPackBC7FlagDefault;
    else if (s.effort == 10)
        o.m_bc7_pack_flags = basist::bc7f::cPackBC7FlagDefaultNonAnalytical;
    o.m_effort_level = clamp<uint32_t>(s.effort, 0, 10);
    o.m_bc7_encoder = (s.bc7e >= 0) ? xbc7::bc7_encoder_type::cBC7E_Scalar : xbc7::bc7_encoder_type::cBC7F;
    o.m_bc7e_scalar_level = (s.bc7e >= 0) ? (uint32_t)s.bc7e : xbc7::DEFAULT_BC7E_SCALAR_LEVEL;
    o.m_perceptual = s.srgb;
    o.m_pJob_pool = &pool;
    o.set_num_stripes_for_image(img, s.stripes);
    if (s.exact_stripes)
        o.m_num_stripes = s.exact_stripes;

    uint8_vec comp;
    vector2D<basist::bc7u::log_bc7_block> coded;
    if (!xbc7::pack_image(img, o, comp, coded)) {
        printf("%s %s fail\n", name, s.id);
        return false;
    }
    printf("%s %s %u %016llx\n", name, s.id, comp.size_u32(), (unsigned long long)fnv(comp.data(), comp.size()));
    return true;
}

static void encode_all(const std::string& name, const image& img, job_pool& pool) {
    for (const pset& s : g_sets)
        encode(name.c_str(), img, s, pool);
}

int main(int argc, char** argv) {
    basisu_encoder_init();
    job_pool pool(1);
    for (int a = 1; a < argc; a++) {
        FILE* f = fopen(argv[a], "rb");
        if (!f) return 1;
        std::vector<uint8_t> d;
        uint8_t buf[4096]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) d.insert(d.end(), buf, buf + n);
        fclose(f);
        const char* name = strrchr(argv[a], '/'); name = name ? name + 1 : argv[a];
        unsigned w = 0, h = 0;
        const char* dims = strrchr(name, '_');
        if (!dims || sscanf(dims + 1, "%ux%u", &w, &h) != 2 || d.size() != (size_t)w * h * 4) return 2;

        image img(w, h);
        for (unsigned y = 0; y < h; y++)
            for (unsigned x = 0; x < w; x++) {
                const uint8_t* p = &d[(y * w + x) * 4];
                img(x, y).set(p[0], p[1], p[2], p[3]);
            }
        encode_all(name, img, pool);

        if (img.has_alpha()) {
            image op(img);
            for (unsigned y = 0; y < h; y++)
                for (unsigned x = 0; x < w; x++)
                    op(x, y).a = 255;
            encode_all(std::string(name) + ":opaque", op, pool);
        }

        if (a == 1) {
            image tiny(6, 5);
            for (unsigned y = 0; y < 5; y++)
                for (unsigned x = 0; x < 6; x++)
                    tiny(x, y) = img(x, y);
            encode_all(std::string(name) + ":tiny", tiny, pool);

            image tall(w, h * 4);
            for (unsigned y = 0; y < h * 4; y++)
                for (unsigned x = 0; x < w; x++)
                    tall(x, y) = img(x, y % h);
            encode_all(std::string(name) + ":tall", tall, pool);
        }
    }
    return 0;
}

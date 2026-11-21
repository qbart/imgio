// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0
// Offline golden generator for imgio's bc7e port (dev tool, not part of the crate): encodes raw
// RGBA8 images (testdata/images/raw/NAME_WxH.rgba) to BC7 with the C++ Basis Universal
// bc7e_scalar encoder, edge blocks reading clamped pixels like imgio::encode, and prints
// "file preset perceptual fnv1a64" (hash over the row-major 16-byte blocks).
// Build: see build.sh (x86-64, SSE2 scalar float math, no -ffast-math).
#include "basisu_bc7e_scalar.h"
#include <cstdio>
#include <cstring>
#include <vector>

using namespace bc7e_scalar;

static uint64_t fnv(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

typedef void (*init_fn)(bc7e_compress_block_params*, bool);

int main(int argc, char** argv) {
    bc7e_compress_block_init();
    const struct { const char* name; init_fn fn; } presets[] = {
        { "ultrafast", bc7e_compress_block_params_init_ultrafast },
        { "veryfast", bc7e_compress_block_params_init_veryfast },
        { "fast", bc7e_compress_block_params_init_fast },
        { "basic", bc7e_compress_block_params_init_basic },
        { "slow", bc7e_compress_block_params_init_slow },
        { "veryslow", bc7e_compress_block_params_init_veryslow },
        { "slowest", bc7e_compress_block_params_init_slowest },
    };
    for (int a = 1; a < argc; a++) {
        FILE* f = fopen(argv[a], "rb");
        if (!f) return 1;
        std::vector<uint8_t> img;
        uint8_t buf[4096]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) img.insert(img.end(), buf, buf + n);
        fclose(f);
        const char* name = strrchr(argv[a], '/'); name = name ? name + 1 : argv[a];
        unsigned w = 0, h = 0;
        const char* dims = strrchr(name, '_');
        if (!dims || sscanf(dims + 1, "%ux%u", &w, &h) != 2 || img.size() != (size_t)w * h * 4) return 2;
        const unsigned bx = (w + 3) / 4, by = (h + 3) / 4, nb = bx * by;
        std::vector<uint32_t> px(nb * 16);
        for (unsigned y = 0; y < by; y++)
            for (unsigned x = 0; x < bx; x++)
                for (unsigned i = 0; i < 16; i++) {
                    unsigned sx = x * 4 + (i & 3), sy = y * 4 + (i >> 2);
                    if (sx >= w) sx = w - 1;
                    if (sy >= h) sy = h - 1;
                    memcpy(&px[(y * bx + x) * 16 + i], &img[(sy * w + sx) * 4], 4);
                }
        for (const auto& p : presets)
            for (int perceptual = 0; perceptual < 2; perceptual++) {
                bc7e_compress_block_params params;
                memset(&params, 0, sizeof(params));
                p.fn(&params, perceptual != 0);
                std::vector<uint64_t> out(nb * 2);
                bc7e_compress_blocks(nb, out.data(), px.data(), &params);
                printf("%s %s %d %016llx\n", name, p.name, perceptual,
                       (unsigned long long)fnv((const uint8_t*)out.data(), out.size() * 8));
            }
    }
    return 0;
}

// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0
// Offline golden generator for imgio's block encoders (dev tool, not part of the crate): encodes
// raw RGBA8 images (testdata/images/raw/NAME_WxH.rgba) block by block with the C++ Basis
// Universal real-time encoders, edge blocks reading clamped pixels like imgio::encode, and prints
// "file target quality fnv1a64". Targets: bc1, bc3, bc4, bc5 (quality = encode_bc1 flags).
// Build: see build.sh.
#include "basisu_transcoder.h"
#include <cstdio>
#include <cstring>
#include <vector>

namespace basist {
void encode_bc1(void* pDst, const uint8_t* pPixels, uint32_t flags);
void encode_bc4(void* pDst, const uint8_t* pPixels, uint32_t stride);
}

static uint64_t fnv(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

int main(int argc, char** argv) {
    basist::basisu_transcoder_init();
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
        const unsigned bx = (w + 3) / 4, by = (h + 3) / 4;
        for (const char* t : { "bc1", "bc3", "bc4", "bc5" }) {
            const bool is_bc1 = !strcmp(t, "bc1"), is_bc3 = !strcmp(t, "bc3");
            const unsigned bytes = (is_bc1 || !strcmp(t, "bc4")) ? 8 : 16;
            for (uint32_t flags = 0; flags < ((is_bc1 || is_bc3) ? 3u : 1u); flags++) {
                std::vector<uint8_t> out(bx * by * bytes);
                for (unsigned y = 0; y < by; y++)
                    for (unsigned x = 0; x < bx; x++) {
                        uint8_t px[64];
                        for (unsigned i = 0; i < 16; i++) {
                            unsigned sx = x * 4 + (i & 3), sy = y * 4 + (i >> 2);
                            if (sx >= w) sx = w - 1;
                            if (sy >= h) sy = h - 1;
                            memcpy(px + i * 4, &img[(sy * w + sx) * 4], 4);
                        }
                        uint8_t* o = &out[(y * bx + x) * bytes];
                        if (is_bc1) basist::encode_bc1(o, px, flags);
                        else if (is_bc3) { basist::encode_bc4(o, px + 3, 4); basist::encode_bc1(o + 8, px, flags); }
                        else if (!strcmp(t, "bc4")) basist::encode_bc4(o, px, 4);
                        else { basist::encode_bc4(o, px, 4); basist::encode_bc4(o + 8, px + 1, 4); }
                    }
                printf("%s %s %u %016llx\n", name, t, flags, (unsigned long long)fnv(out.data(), out.size()));
            }
        }
    }
    return 0;
}

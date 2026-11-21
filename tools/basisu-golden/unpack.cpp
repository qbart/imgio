// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0
// Offline golden generator for imgio's PVRTC1/2, ATC and FXT1 decoders (dev tool, not part of the
// crate). For every image-0 level of the given .basis files and every target among PVRTC1
// RGB/RGBA (8, 9), ATC RGB/RGBA (11, 12), FXT1 (17), PVRTC2 RGB/RGBA (18, 19) that the file
// supports, transcodes with the C++ transcoder, unpacks with basisu `gpu_image::unpack` (the
// encoder library) and prints "file format level fnv1a64(rgba8 w*h)" or "file format level fail".
// Build: see build.sh (needs ../../ref/basisu and its basisu_encoder static library).
#include "basisu_enc.h"
#include "basisu_gpu_texture.h"
#include "basisu_transcoder.h"
#include <cstdio>
#include <cstring>
#include <vector>

using namespace basist;

static uint64_t fnv(const uint8_t* p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

int main(int argc, char** argv) {
    basisu::basisu_encoder_init();
    basisu_transcoder_init();
    const int fmts[] = { 8, 9, 11, 12, 17, 18, 19 };
    for (int a = 1; a < argc; a++) {
        FILE* f = fopen(argv[a], "rb");
        if (!f) return 1;
        std::vector<uint8_t> d;
        uint8_t buf[4096]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) d.insert(d.end(), buf, buf + n);
        fclose(f);
        const char* name = strrchr(argv[a], '/'); name = name ? name + 1 : argv[a];
        basisu_transcoder t;
        if (!t.validate_header(d.data(), (uint32_t)d.size()) || !t.start_transcoding(d.data(), (uint32_t)d.size())) return 2;
        basisu_file_info fi;
        t.get_file_info(d.data(), (uint32_t)d.size(), fi);
        for (int fmt : fmts) {
            auto tf = (transcoder_texture_format)fmt;
            if (!basis_is_format_supported(tf, fi.m_tex_format)) continue;
            uint32_t levels = t.get_total_image_levels(d.data(), (uint32_t)d.size(), 0);
            for (uint32_t l = 0; l < levels; l++) {
                uint32_t w, h, blocks;
                t.get_image_level_desc(d.data(), (uint32_t)d.size(), 0, l, w, h, blocks);
                basisu::gpu_image gi(basis_get_basisu_texture_format(tf), w, h);
                bool ok = t.transcode_image_level(d.data(), (uint32_t)d.size(), 0, l, gi.get_ptr(),
                    gi.get_total_blocks(), tf, 0);
                basisu::image img;
                ok = ok && gi.unpack(img, false);
                if (!ok) { printf("%s %d %u fail\n", name, fmt, l); continue; }
                std::vector<uint8_t> px(w * h * 4);
                for (uint32_t y = 0; y < h; y++)
                    for (uint32_t x = 0; x < w; x++)
                        memcpy(&px[(y * w + x) * 4], &img(x, y), 4);
                // basisu unpack_atc leaves alpha of palette entries 1/2 (mode 0) uninitialized;
                // ATC RGB is opaque.
                if (fmt == 11)
                    for (size_t i = 3; i < px.size(); i += 4) px[i] = 255;
                printf("%s %d %u %016llx\n", name, fmt, l, (unsigned long long)fnv(px.data(), px.size()));
            }
        }
    }
    return 0;
}

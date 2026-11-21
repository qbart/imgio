// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0
// Offline golden generator (dev tool, not part of the crate): transcodes every image level of
// the given .basis / .ktx2 files to every supported transcoder_texture_format (0..40) with the
// C++ Basis Universal transcoder and prints "file format level size fnv1a64" (.basis, image 0)
// or "file format level layer face size fnv1a64" (.ktx2, every layer and face).
// Build: see build.sh (needs ../../ref/basisu).
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
    basisu_transcoder_init();
    for (int a = 1; a < argc; a++) {
        FILE* f = fopen(argv[a], "rb");
        if (!f) return 1;
        std::vector<uint8_t> d;
        uint8_t buf[4096]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) d.insert(d.end(), buf, buf + n);
        fclose(f);
        const char* name = strrchr(argv[a], '/'); name = name ? name + 1 : argv[a];
        if (strstr(name, ".ktx2")) {
            ktx2_transcoder k;
            if (!k.init(d.data(), (uint32_t)d.size()) || !k.start_transcoding()) return 3;
            for (int fmt = 0; fmt <= 40; fmt++) {
                auto tf = (transcoder_texture_format)fmt;
                if (!basis_is_format_supported(tf, k.get_basis_tex_format())) continue;
                for (uint32_t l = 0; l < k.get_levels(); l++)
                for (uint32_t layer = 0; layer < (k.get_layers() ? k.get_layers() : 1); layer++)
                for (uint32_t face = 0; face < k.get_faces(); face++) {
                    ktx2_image_level_info li;
                    k.get_image_level_info(li, l, layer, face);
                    uint32_t w = li.m_orig_width, h = li.m_orig_height, size, units;
                    if (basis_transcoder_format_is_uncompressed(tf)) {
                        units = w * h;
                        size = units * basis_get_uncompressed_bytes_per_pixel(tf);
                    } else {
                        uint32_t bw = basis_get_block_width(tf), bh = basis_get_block_height(tf);
                        units = ((w + bw - 1) / bw) * ((h + bh - 1) / bh);
                        size = units * basis_get_bytes_per_block_or_pixel(tf);
                    }
                    std::vector<uint8_t> out(size);
                    if (k.transcode_image_level(l, layer, face, out.data(), units, tf, 0))
                        printf("%s %d %u %u %u %u %016llx\n", name, fmt, l, layer, face, size, (unsigned long long)fnv(out.data(), size));
                    else printf("%s %d %u %u %u fail\n", name, fmt, l, layer, face);
                }
            }
            continue;
        }
        basisu_transcoder t;
        if (!t.validate_header(d.data(), (uint32_t)d.size()) || !t.start_transcoding(d.data(), (uint32_t)d.size())) return 2;
        basisu_file_info fi;
        t.get_file_info(d.data(), (uint32_t)d.size(), fi);
        for (int fmt = 0; fmt <= 40; fmt++) {
            auto tf = (transcoder_texture_format)fmt;
            if (!basis_is_format_supported(tf, fi.m_tex_format)) continue;
            uint32_t levels = t.get_total_image_levels(d.data(), (uint32_t)d.size(), 0);
            for (uint32_t l = 0; l < levels; l++) {
                uint32_t w, h, blocks;
                t.get_image_level_desc(d.data(), (uint32_t)d.size(), 0, l, w, h, blocks);
                uint32_t size;
                if (basis_transcoder_format_is_uncompressed(tf)) {
                    size = w * h * basis_get_uncompressed_bytes_per_pixel(tf);
                } else {
                    uint32_t bw = basis_get_block_width(tf), bh = basis_get_block_height(tf);
                    size = ((w + bw - 1) / bw) * ((h + bh - 1) / bh) * basis_get_bytes_per_block_or_pixel(tf);
                }
                std::vector<uint8_t> out(size);
                uint32_t units = basis_transcoder_format_is_uncompressed(tf) ? w * h : size / basis_get_bytes_per_block_or_pixel(tf);
                bool ok = t.transcode_image_level(d.data(), (uint32_t)d.size(), 0, l, out.data(), units, tf, 0);
                if (ok) printf("%s %d %u %u %016llx\n", name, fmt, l, size, (unsigned long long)fnv(out.data(), size));
                else printf("%s %d %u fail\n", name, fmt, l);
            }
        }
    }
    return 0;
}

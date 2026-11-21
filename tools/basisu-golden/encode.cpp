// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0
// Offline golden generator for imgio's block encoders (dev tool, not part of the crate): encodes
// raw RGBA8 images (testdata/images/raw/NAME_WxH.rgba) block by block with the C++ Basis
// Universal real-time encoders, edge blocks reading clamped pixels like imgio::encode, and prints
// "file target quality fnv1a64". Targets: bc1, bc3, bc4, bc5 (quality = encode_bc1 flags), bc7
// (bc7f presets 0..4 = fastest, fast, default, partially / non analytical). Raw RGBA16F images
// (NAME_WxH.rgba16f) are encoded to bc6h with imgio's 5 fast_bc6h_params presets. etc1
// (etc1f), etc2rgba / eacr11 / eacrg11 (quality 0 = pack_eac, 1 = pack_eac_high_quality).
// pvrtc1rgb / pvrtc1rgba (encode_pvrtc1) on the top-left power-of-two crop (largest w, h).
// Build: see build.sh.
// The transcoder source is included (not linked) to reach its static EAC packers.
#include "basisu_transcoder.cpp"
#include <cstdio>
#include <cstring>
#include <vector>


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
        if (!dims || sscanf(dims + 1, "%ux%u", &w, &h) != 2) return 2;
        if (!strstr(name, ".rgba16f") && img.size() != (size_t)w * h * 4) return 2;
        const unsigned bx = (w + 3) / 4, by = (h + 3) / 4;
        if (strstr(name, ".rgba16f")) {
            if (img.size() != (size_t)w * h * 8) return 3;
            const uint16_t* hp = reinterpret_cast<const uint16_t*>(img.data());
            // {diff endpoint modes, 2-subset patterns, hq ls, brute force weights}
            const unsigned presets[5][4] = { { 0, 0, 0, 0 }, { 2, 0, 0, 0 }, { 2, 1, 1, 0 }, { 4, 8, 1, 0 }, { 9, 32, 1, 1 } };
            for (unsigned q = 0; q < 5; q++) {
                basist::astc_6x6_hdr::fast_bc6h_params p;
                p.m_num_diff_endpoint_modes_to_try = presets[q][0];
                p.m_max_2subset_pats_to_try = presets[q][1];
                p.m_hq_ls = presets[q][2] != 0;
                p.m_brute_force_weight4_assignment = presets[q][3] != 0;
                std::vector<uint8_t> out(bx * by * 16);
                for (unsigned y = 0; y < by; y++)
                    for (unsigned x = 0; x < bx; x++) {
                        basist::half_float px[48];
                        for (unsigned i = 0; i < 16; i++) {
                            unsigned sx = x * 4 + (i & 3), sy = y * 4 + (i >> 2);
                            if (sx >= w) sx = w - 1;
                            if (sy >= h) sy = h - 1;
                            for (unsigned c = 0; c < 3; c++) px[i * 3 + c] = hp[(sy * w + sx) * 4 + c];
                        }
                        basist::astc_6x6_hdr::fast_encode_bc6h(px, reinterpret_cast<basist::bc6h_block*>(&out[(y * bx + x) * 16]), p);
                    }
                printf("%s bc6h %u %016llx\n", name, q, (unsigned long long)fnv(out.data(), out.size()));
            }
            continue;
        }
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
        for (const char* t : { "etc1", "etc2rgba", "eacr11", "eacrg11" }) {
            const bool is_etc1 = !strcmp(t, "etc1"), is_r = !strcmp(t, "eacr11");
            const unsigned bytes = (is_etc1 || is_r) ? 8 : 16;
            for (unsigned hq = 0; hq < (is_etc1 ? 1u : 2u); hq++) {
                std::vector<uint8_t> out(bx * by * bytes);
                basist::etc1f::pack_etc1_state st;
                for (unsigned y = 0; y < by; y++)
                    for (unsigned x = 0; x < bx; x++) {
                        basist::color_rgba px[16];
                        for (unsigned i = 0; i < 16; i++) {
                            unsigned sx = x * 4 + (i & 3), sy = y * 4 + (i >> 2);
                            if (sx >= w) sx = w - 1;
                            if (sy >= h) sy = h - 1;
                            memcpy(&px[i], &img[(sy * w + sx) * 4], 4);
                        }
                        uint8_t* o = &out[(y * bx + x) * bytes];
                        const uint8_t* p8 = &px[0].r;
                        auto eac = [&](uint8_t* d, unsigned c) {
                            if (hq) basist::pack_eac_high_quality(*reinterpret_cast<basist::eac_block*>(d), p8 + c, 4);
                            else basist::pack_eac(*reinterpret_cast<basist::eac_block*>(d), p8 + c, 4);
                        };
                        if (is_etc1) basist::etc1f::pack_etc1(o, px, st);
                        else if (is_r) eac(o, 0);
                        else if (!strcmp(t, "eacrg11")) { eac(o, 0); eac(o + 8, 1); }
                        else { eac(o, 3); basist::etc1f::pack_etc1(o + 8, px, st); }
                    }
                printf("%s %s %u %016llx\n", name, t, hq, (unsigned long long)fnv(out.data(), out.size()));
            }
        }
        {
            unsigned pw = 1, ph = 1;
            while (pw * 2 <= w) pw *= 2;
            while (ph * 2 <= h) ph *= 2;
            basisu::vector2D<basist::color32> timg(pw, ph);
            for (unsigned y = 0; y < ph; y++)
                for (unsigned x = 0; x < pw; x++)
                    memcpy(&timg(x, y), &img[(y * w + x) * 4], 4);
            for (unsigned rgba = 0; rgba < 2; rgba++) {
                std::vector<uint8_t> out((pw / 4) * (ph / 4) * 8);
                basist::encode_pvrtc1(rgba ? basist::block_format::cPVRTC1_4_RGBA : basist::block_format::cPVRTC1_4_RGB,
                    out.data(), timg, pw / 4, ph / 4, false);
                printf("%s %s 0 %016llx\n", name, rgba ? "pvrtc1rgba" : "pvrtc1rgb", (unsigned long long)fnv(out.data(), out.size()));
            }
        }
        const uint32_t bc7_presets[5] = { basist::bc7f::cPackBC7FlagDefaultFastest, basist::bc7f::cPackBC7FlagDefaultFast,
            basist::bc7f::cPackBC7FlagDefault, basist::bc7f::cPackBC7FlagDefaultPartiallyAnalytical, basist::bc7f::cPackBC7FlagDefaultNonAnalytical };
        for (unsigned q = 0; q < 5; q++) {
            std::vector<uint8_t> out(bx * by * 16);
            for (unsigned y = 0; y < by; y++)
                for (unsigned x = 0; x < bx; x++) {
                    basist::color_rgba px[16];
                    for (unsigned i = 0; i < 16; i++) {
                        unsigned sx = x * 4 + (i & 3), sy = y * 4 + (i >> 2);
                        if (sx >= w) sx = w - 1;
                        if (sy >= h) sy = h - 1;
                        memcpy(&px[i], &img[(sy * w + sx) * 4], 4);
                    }
                    basist::bc7f::fast_pack_bc7_auto_rgba(&out[(y * bx + x) * 16], px, bc7_presets[q]);
                }
            printf("%s bc7 %u %016llx\n", name, q, (unsigned long long)fnv(out.data(), out.size()));
        }
    }
    return 0;
}

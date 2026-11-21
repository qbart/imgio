// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Offline golden generator for src/basis/xuastc/enc/{block_helpers,lowlevel}.rs: exercises the
// block level full ("basisu") ASTC LDR encoder of basisu v2.50 (encoder/basisu_astc_ldr_encode.cpp
// up to scoped_ldr_astc_lowlevel_block_encoder) on deterministic images and prints one line per
// test group: "name<TAB>fnv1a64<TAB>count".
// The .cpp is included directly because the tested functions are static (see build.sh). The Rust
// test (src/basis/xuastc/enc/tests_block.rs) regenerates the same inputs with the same PRNG.
#include "basisu_astc_ldr_encode.cpp"
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

using namespace basisu;
using namespace basisu::astc_ldr;

struct lcg
{
	uint32_t s;
	explicit lcg(uint32_t seed) : s(seed) {}
	uint32_t next() { s = s * 1664525u + 1013904223u; return s >> 8; }
	uint32_t range(uint32_t n) { return next() % n; }
};

struct group
{
	uint64_t h = 0xcbf29ce484222325ull;
	uint32_t count = 0;
	void bytes(const void* p, size_t n)
	{
		const uint8_t* b = (const uint8_t*)p;
		for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
	}
	void u32(uint32_t v) { bytes(&v, 4); }
	void u64(uint64_t v) { bytes(&v, 8); }
	void f32(float v) { uint32_t u; memcpy(&u, &v, 4); u32(u); }
	void blk(const astc_helpers::log_astc_block& b)
	{
		const uint8_t hd[11] = { (uint8_t)b.m_error_flag, (uint8_t)b.m_solid_color_flag_ldr, (uint8_t)b.m_solid_color_flag_hdr, b.m_user_mode,
			b.m_grid_width, b.m_grid_height, (uint8_t)b.m_dual_plane, b.m_weight_ise_range, b.m_endpoint_ise_range, b.m_color_component_selector,
			b.m_num_partitions };
		bytes(hd, 11);
		bytes(&b.m_partition_id, 2);
		bytes(b.m_color_endpoint_modes, 4);
		bytes(b.m_weights, 64);
		bytes(b.m_endpoints, 18);
	}
	void sblk(const log_surrogate_astc_blk& b, uint32_t n)
	{
		u32((uint32_t)b.m_grid_width); u32((uint32_t)b.m_grid_height); u32(b.m_cem_index); u32((uint32_t)b.m_ccs_index);
		u32(b.m_num_endpoint_levels); u32(b.m_num_weight_levels); u32(b.m_num_parts); u32(b.m_seed_index);
		for (uint32_t p = 0; p < 4; p++)
		{
			for (uint32_t c = 0; c < 4; c++) { f32(b.m_endpoints[p][0][c]); f32(b.m_endpoints[p][1][c]); }
			f32(b.m_scales[p]);
		}
		for (uint32_t i = 0; i < n; i++) { f32(b.m_weights0[i]); f32(b.m_weights1[i]); }
	}
};

static std::map<std::string, group> g_groups;
static group& G(const std::string& n) { group& g = g_groups[n]; g.count++; return g; }

static const uint32_t BS[10][2] = { {4,4},{5,4},{6,5},{6,6},{8,5},{8,8},{10,6},{10,10},{12,10},{12,12} };

static uint8_t clamp8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

// Deterministic test image of 2x2 blocks (integer only, mirrored in the Rust test).
static void make_image(uint32_t c, uint32_t w, uint32_t h, image& img)
{
	lcg r(c * 7919u + 31u);
	const uint32_t kind = c % 8;
	const uint32_t alpha_mode = (c / 8) % 3;
	int base[4], base2[4], base3[4];
	for (int i = 0; i < 4; i++) base[i] = (int)r.range(256);
	for (int i = 0; i < 4; i++) base2[i] = (int)r.range(256);
	for (int i = 0; i < 4; i++) base3[i] = (int)r.range(256);
	const int dx = (int)r.range(41) - 20, dy = (int)r.range(41) - 20;
	img.resize(w, h);
	for (uint32_t y = 0; y < h; y++)
	{
		for (uint32_t x = 0; x < w; x++)
		{
			const int xi = (int)x, yi = (int)y, wi = (int)w, hi = (int)h;
			int v[4];
			for (int ch = 0; ch < 4; ch++)
			{
				switch (kind)
				{
				case 0: v[ch] = base[ch] + (dx * xi * (ch + 1)) / 4 + (dy * yi) / 3; break;
				case 1: v[ch] = ((xi * 3 + (yi * dx) / 10) > (wi * 3) / 2) ? base[ch] : base2[ch]; break;
				case 2: v[ch] = base[ch] + (int)r.range(61) - 30; break;
				case 3: v[ch] = base[ch]; break;
				case 4:
				{
					const int t = ((xi * 2 + yi) * 3) / (wi * 2 + hi);
					v[ch] = ((t == 0) ? base[ch] : ((t == 1) ? base2[ch] : base3[ch])) + (int)r.range(9) - 4;
					break;
				}
				case 5: v[ch] = base[ch] + (dx * xi) / 3 + (int)r.range(17) - 8; break;
				case 6: v[ch] = base[0] + (dx * xi) / 2 + (dy * yi) / 2; break;
				default:
				{
					const int t = (xi * 64) / (wi - 1);
					v[ch] = (base[ch] * (64 - t) + base2[ch] * t) / 64;
					break;
				}
				}
			}
			if (alpha_mode == 0)
				v[3] = 255;
			else if (alpha_mode == 2)
				v[3] = 255 - (xi * 255) / (wi - 1);
			img(x, y).set(clamp8(v[0]), clamp8(v[1]), clamp8(v[2]), clamp8(v[3]));
		}
	}
}

static void sobel_xy(const image& img, image& xy)
{
	image sx, sy;
	compute_sobel(img, sx, &g_sobel_x[0][0]);
	compute_sobel(img, sy, &g_sobel_y[0][0]);
	xy.resize(img.get_width(), img.get_height());
	for (uint32_t y = 0; y < img.get_height(); y++)
		for (uint32_t x = 0; x < img.get_width(); x++)
		{
			const color_rgba& a = sx(x, y);
			const color_rgba& b = sy(x, y);
			xy(x, y).set(iabs((int)a.r - 128) + iabs((int)b.r - 128), iabs((int)a.g - 128) + iabs((int)b.g - 128),
				iabs((int)a.b - 128) + iabs((int)b.b - 128), iabs((int)a.a - 128) + iabs((int)b.a - 128));
		}
}

struct size_ctx
{
	basisu::vector<basist::astc_ldr_t::trial_mode> tms;
	basist::astc_ldr_t::grouped_trial_modes grouped;
	partitions_data* p2 = nullptr;
	partitions_data* p3 = nullptr;
	basist::astc_ldr_t::dct2f dct;
};

int main(int argc, char** argv)
{
	basisu_encoder_init(false, false);
	const uint32_t num_cases = (argc > 1) ? (uint32_t)atoi(argv[1]) : 60;

	size_ctx ctx[10];
	for (uint32_t s = 0; s < 10; s++)
	{
		const uint32_t bw = BS[s][0], bh = BS[s][1];
		basist::astc_ldr_t::create_encoder_trial_modes_table(bw, bh, ctx[s].tms, ctx[s].grouped, false, false);
		ctx[s].p2 = new partitions_data();
		ctx[s].p2->init(2, bw, bh, false, true);
		ctx[s].p3 = new partitions_data();
		ctx[s].p3->init(3, bw, bh, false, true);
		ctx[s].dct.init(bh, bw);
	}

	ldr_astc_lowlevel_block_encoder* pEnc = new ldr_astc_lowlevel_block_encoder();

	for (uint32_t c = 0; c < num_cases; c++)
	{
		const uint32_t si = c % 10;
		const uint32_t bw = BS[si][0], bh = BS[si][1], n = bw * bh;
		size_ctx& sc = ctx[si];

		image img, sob;
		make_image(c, bw * 2, bh * 2, img);
		sobel_xy(img, sob);
		{
			group& g = G("sobel");
			g.bytes(sob.get_ptr(), sob.get_width() * sob.get_height() * 4);
		}

		const uint32_t bx = c & 1, by = (c >> 1) & 1;
		color_rgba px[144];
		for (uint32_t y = 0; y < bh; y++)
			for (uint32_t x = 0; x < bw; x++)
				px[x + y * bw] = img(bx * bw + x, by * bh + y);

		pixel_stats_t ps;
		ps.clear();
		ps.init(n, px);

		cem_encode_params cep;
		const bool srgb = ((c / 3) & 1) != 0;
		cep.m_decode_mode_srgb = srgb;
		if (srgb)
		{
			cep.m_comp_weights[0] = 9; cep.m_comp_weights[1] = 11; cep.m_comp_weights[2] = 1; cep.m_comp_weights[3] = 11;
		}

		lcg r(c * 104729u + 7u);

		// Partition estimation
		int parts2[512], parts3[512];
		int num2 = (int)minimum<uint32_t>(32u << (c % 3), sc.p2->m_total_unique_patterns);
		int num3 = (int)minimum<uint32_t>(32u << (c % 3), sc.p3->m_total_unique_patterns);
		{
			const bool ok2 = estimate_partition2(bw, bh, ps, parts2, num2, sc.p2, false);
			group& g = G("est_part2");
			g.u32(ok2); g.u32(ok2 ? num2 : 0);
			if (ok2) g.bytes(parts2, num2 * 4);
			if (!ok2) num2 = 0;

			const bool ok3 = estimate_partition3(bw, bh, ps, parts3, num3, sc.p3, false);
			group& g3 = G("est_part3");
			g3.u32(ok3); g3.u32(ok3 ? num3 : 0);
			if (ok3) g3.bytes(parts3, num3 * 4);
			if (!ok3) num3 = 0;

			int bf[1024];
			int nbf = 16;
			const bool okb = estimate_partition2(bw, bh, ps, bf, nbf, sc.p2, true);
			group& gb = G("est_part2_bf");
			gb.u32(okb); gb.u32(nbf); gb.bytes(bf, nbf * 4);
			nbf = 16;
			const bool okb3 = estimate_partition3(bw, bh, ps, bf, nbf, sc.p3, true);
			group& gb3 = G("est_part3_bf");
			gb3.u32(okb3);
			if (okb3) { gb3.u32(nbf); gb3.bytes(bf, nbf * 4); }

			group& ge = G("est_sse");
			const uint32_t cems[4] = { 8, 6, 12, 10 };
			for (int i = 0; i < minimum(num2, 4); i++)
				for (uint32_t k = 0; k < 4; k++)
					ge.f32(estimate_partition_sse(cems[k], ps, bw, bh, 2, &sc.p2->m_partition_pats[parts2[i]]));
			for (int i = 0; i < minimum(num3, 4); i++)
				for (uint32_t k = 0; k < 4; k++)
					ge.f32(estimate_partition_sse(cems[k], ps, bw, bh, 3, &sc.p3->m_partition_pats[parts3[i]]));
		}

		// Trial encodes on random trial modes
		for (uint32_t t = 0; t < 6; t++)
		{
			const uint32_t tm_index = r.range(sc.tms.size_u32());
			const basist::astc_ldr_t::trial_mode& tm = sc.tms[tm_index];
			const uint32_t pick = r.next();
			const bool gd = (pick & 1) == 0, pw = (pick & 2) == 0, qcd = (pick & 4) == 0;
			const bool bc = (pick & 8) == 0;
			const uint32_t sflags = (pick >> 4) & 1;
			const float eo = ((pick >> 5) & 1) ? .1f : 0.0f;

			const partitions_data* pd = (tm.m_num_parts == 3) ? sc.p3 : sc.p2;
			const int num_est = (tm.m_num_parts == 3) ? num3 : num2;
			if ((tm.m_num_parts > 1) && (!num_est))
			{
				G("trial_skip").u32(tm_index);
				continue;
			}
			uint32_t unique = 0, seed = 0;
			const partition_pattern_vec* pPat = nullptr;
			if (tm.m_num_parts > 1)
			{
				unique = ((tm.m_num_parts == 3) ? parts3 : parts2)[(pick >> 8) % (uint32_t)num_est];
				seed = pd->m_unique_index_to_part_seed[unique];
				pPat = &pd->m_partition_pats[unique];
			}

			// surrogate
			{
				log_surrogate_astc_blk lb;
				lb.clear();
				float e;
				if (tm.m_num_parts == 1)
					e = encode_surrogate_trial(bw, bh, ps, tm.m_cem, tm.m_ccs_index, tm.m_endpoint_ise_range, tm.m_weight_ise_range,
						tm.m_grid_width, tm.m_grid_height, lb, cep, sflags);
				else
					e = encode_surrogate_trial_subsets(bw, bh, ps, tm.m_cem, tm.m_num_parts, seed, pPat, tm.m_endpoint_ise_range, tm.m_weight_ise_range,
						tm.m_grid_width, tm.m_grid_height, lb, cep, sflags);
				group& g = G("surrogate");
				g.u32(tm_index); g.f32(e); g.sblk(lb, tm.m_grid_width * tm.m_grid_height);

				vec4F dec[144];
				lb.decode(bw, bh, dec, pPat);
				group& gd2 = G("surrogate_decode");
				for (uint32_t i = 0; i < n; i++) for (uint32_t k = 0; k < 4; k++) gd2.f32(dec[i][k]);
			}

			// full trial encode
			astc_helpers::log_astc_block lb;
			lb.clear();
			bool tdef = false;
			bool ok;
			if (tm.m_num_parts == 1)
				ok = encode_trial(bw, bh, ps, tm.m_cem, tm.m_ccs_index >= 0, tm.m_ccs_index, tm.m_endpoint_ise_range, tm.m_weight_ise_range,
					tm.m_grid_width, tm.m_grid_height, eo, lb, cep, gd, pw, qcd, bc, &tdef);
			else
				ok = encode_trial_subsets(bw, bh, ps, tm.m_cem, tm.m_num_parts, seed, pPat, tm.m_endpoint_ise_range, tm.m_weight_ise_range,
					tm.m_grid_width, tm.m_grid_height, eo, lb, cep, false, gd, pw, qcd, bc, &tdef);
			{
				group& g = G("trial");
				g.u32(tm_index); g.u32(ok); g.u32(tdef);
				if (ok) { g.blk(lb); g.u64(eval_error(bw, bh, lb, ps, cep)); }
			}
			if (!ok)
				continue;

			// polish on an unpolished encode
			{
				astc_helpers::log_astc_block lb2;
				lb2.clear();
				bool ok2;
				if (tm.m_num_parts == 1)
					ok2 = encode_trial(bw, bh, ps, tm.m_cem, tm.m_ccs_index >= 0, tm.m_ccs_index, tm.m_endpoint_ise_range, tm.m_weight_ise_range,
						tm.m_grid_width, tm.m_grid_height, 0.0f, lb2, cep, false, false, false, bc, nullptr);
				else
					ok2 = encode_trial_subsets(bw, bh, ps, tm.m_cem, tm.m_num_parts, seed, pPat, tm.m_endpoint_ise_range, tm.m_weight_ise_range,
						tm.m_grid_width, tm.m_grid_height, 0.0f, lb2, cep, false, false, false, false, bc, nullptr);
				group& g = G("polish");
				g.u32(ok2);
				if (ok2)
				{
					bool improved = false;
					polish_block_weights(bw, bh, ps, lb2, cep, pPat, improved, true, true, (pick & 16) == 0);
					g.u32(improved); g.blk(lb2); g.u64(eval_error(bw, bh, lb2, ps, cep));
				}
			}

			if (tm.m_num_parts == 1)
			{
				astc_helpers::log_astc_block lb3(lb);
				bool tdef3 = false;
				encode_trial_refine_only(bw, bh, ps, lb3, cep, bc, &tdef3);
				group& g = G("refine_only");
				g.u32(tdef3); g.blk(lb3);
			}
			else
			{
				astc_helpers::log_astc_block lb3(lb);
				bool tdef3 = false;
				const bool ok3 = encode_trial_subsets(bw, bh, ps, tm.m_cem, tm.m_num_parts, seed, pPat, tm.m_endpoint_ise_range, tm.m_weight_ise_range,
					tm.m_grid_width, tm.m_grid_height, eo, lb3, cep, true, gd, pw, qcd, bc, &tdef3);
				group& g = G("refine_only_subsets");
				g.u32(ok3); g.u32(tdef3); g.blk(lb3);
			}
		}

		// Full block encode
		{
			ldr_astc_lowlevel_block_encoder_params p;
			p.m_bx = bx; p.m_by = by;
			p.m_block_width = bw; p.m_block_height = bh; p.m_total_block_pixels = n;
			p.m_pOrig_img_sobel_xy_t = &sob;
			p.m_pPart_data_p2 = sc.p2; p.m_pPart_data_p3 = sc.p3;
			p.m_pEnc_params = &cep;
			p.m_num_trial_modes = sc.tms.size_u32();
			p.m_pTrial_modes = sc.tms.data();
			p.m_pGrouped_trial_modes = &sc.grouped;
			p.m_pDCT2F = &sc.dct;
			p.m_alpha_cems = ps.m_has_alpha;
			p.m_use_alpha_or_opaque_modes = ps.m_has_alpha;
			p.m_use_lum_direct_modes = (c % 4) == 0;
			p.m_use_base_scale_modes = (c % 3) != 2;
			p.m_use_direct_modes = ((c % 17) != 16) || !p.m_use_base_scale_modes;
			p.m_subsets_enabled = (c % 5) != 3;
			p.m_use_dual_planes = (c % 5) != 4;
			p.m_grid_hv_filtering = (c % 7) == 3;
			p.m_filter_horizontally_flag = (c & 1) != 0;
			p.m_use_small_grids_only = (c % 11) == 5;
			p.m_early_stop_wpsnr = ((c % 6) == 1) ? 45.0f : 0.0f;
			p.m_early_stop2_wpsnr = ((c % 6) == 2) ? 40.0f : 0.0f;
			p.m_bucket_pruning_passes = (c % 9) != 8;
			p.m_use_superbuckets = (c % 13) != 12;
			p.m_final_encode_always_try_rgb_direct = (c % 10) == 7;
			p.m_final_encode_try_base_ofs = (c % 8) != 6;
			p.m_use_fast_part_est_stage2 = (c % 12) != 11;
			if ((c % 4) == 1) p.m_dp_active_chans[1] = false;

			basisu::vector<encode_block_output> out_blocks;
			encode_block_stats stats;
			const bool ok = pEnc->full_encode(p, ps, out_blocks, c & 3, stats);
			group& g = G("full_encode");
			g.u32(ok);
			g.u32(out_blocks.size_u32());
			for (uint32_t i = 0; i < out_blocks.size(); i++)
			{
				g.u32((uint32_t)(int)out_blocks[i].m_trial_mode_index);
				g.u32(out_blocks[i].m_blur_id);
				g.blk(out_blocks[i].m_log_blk);
				g.u64(out_blocks[i].m_sse);
			}
			if (getenv("XU_BLOCK_DEBUG"))
			{
				for (uint32_t i = 0; i < out_blocks.size(); i++)
				{
					group gg; gg.blk(out_blocks[i].m_log_blk);
					fprintf(stderr, "%u %u tm=%d sse=%llu blk=%016llx\n", c, i, out_blocks[i].m_trial_mode_index, (unsigned long long)out_blocks[i].m_sse, (unsigned long long)gg.h);
					if (getenv("XU_BLOCK_DEBUG2") && (c == (uint32_t)atoi(getenv("XU_BLOCK_DEBUG2"))))
					{
						const auto& b = out_blocks[i].m_log_blk;
						fprintf(stderr, "  cem %u parts %u pid %u er %u wr %u g %ux%u dp %u ccs %u\n  E:", b.m_color_endpoint_modes[0], b.m_num_partitions, b.m_partition_id, b.m_endpoint_ise_range, b.m_weight_ise_range, b.m_grid_width, b.m_grid_height, b.m_dual_plane, b.m_color_component_selector);
						for (int k = 0; k < 18; k++) fprintf(stderr, " %u", b.m_endpoints[k]);
						fprintf(stderr, "\n  W:");
						for (int k = 0; k < 64; k++) fprintf(stderr, " %u", b.m_weights[k]);
						fprintf(stderr, "\n");
					}
				}
			}
			group& gs = G("full_encode_stats");
			gs.u32(stats.m_total_superbuckets_created); gs.u32(stats.m_total_buckets_created); gs.u32(stats.m_total_surrogate_encodes);
			gs.u32(stats.m_total_shortlist_candidates); gs.u32(stats.m_total_full_encodes);
			gs.f32(pEnc->m_sobel_energy); gs.f32(pEnc->m_max_std_dev); gs.u32(pEnc->m_block_complexity_index);
		}
	}

	for (auto& kv : g_groups)
		printf("%s\t%016llx\t%u\n", kv.first.c_str(), (unsigned long long)kv.second.h, kv.second.count);
	return 0;
}

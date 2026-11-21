// SPDX-FileCopyrightText: 2019-2026 Binomial LLC
// SPDX-License-Identifier: Apache-2.0
// Offline golden generator for src/basis/xuastc/enc/common*.rs: exercises the public API of
// basisu v2.50 encoder/basisu_astc_ldr_common.h (namespace astc_ldr) on deterministic blocks and
// prints one line per test group: "name<TAB>fnv1a64<TAB>count".
// Links the reference libbasisu_encoder.a (see build.sh). The Rust test
// (src/basis/xuastc/enc/tests_common.rs) regenerates the same inputs with the same PRNG.
#include "basisu_enc.h"
#include "basisu_astc_ldr_common.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

using namespace basisu;

namespace basisu { namespace astc_ldr { color_rgba blue_contract_dec(int enc_r, int enc_g, int enc_b, int enc_a); } }

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
};

static std::map<std::string, group> g_groups;
static group& G(const std::string& n) { return g_groups[n]; }

static const uint32_t SIZES[] = { 16, 144, 36, 64, 20, 25, 30, 48, 80, 100, 120, 1, 2, 5, 9, 12 };

static uint8_t clamp8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

// Deterministic test block (integer only, mirrored in the Rust test).
static uint32_t make_block(uint32_t b, color_rgba* px)
{
	lcg r(b * 7919u + 17u);
	const uint32_t n = SIZES[b % 16];
	const uint32_t kind = (b / 16) % 10;
	int base[4], base2[4];
	for (int c = 0; c < 4; c++) base[c] = (int)r.range(256);
	for (int c = 0; c < 4; c++) base2[c] = (int)r.range(256);
	const bool alpha = (r.next() & 1) != 0;
	const int dx = (int)r.range(41) - 20, dy = (int)r.range(41) - 20;
	for (uint32_t i = 0; i < n; i++)
	{
		const int x = (int)(i % 12), y = (int)(i / 12);
		int v[4];
		for (int c = 0; c < 4; c++)
		{
			switch (kind)
			{
			case 0: v[c] = (int)r.range(256); break;
			case 1: v[c] = base[c] + (int)r.range(9) - 4; break;
			case 2: v[c] = base[c] + (x * dx + y * dy) * (c + 1) / 4 + (int)r.range(5) - 2; break;
			case 3: v[c] = base[c]; break;
			case 4: v[c] = (r.next() & 1) ? base[c] : base2[c]; break;
			case 5: { int t = (int)((i * 255) / n); v[c] = base[c] + ((base2[c] - base[c]) * t) / 255; break; }
			case 6: { int g = base[0] + x * dx / 2 + y * dy / 2; v[c] = (c == 3) ? base[3] : g; break; }
			case 7: v[c] = (int)r.range(2) * 255; break;
			case 8: v[c] = (c == 3) ? (int)r.range(256) : base[c] + (int)r.range(33) - 16; break;
			default: v[c] = ((x + y) & 1) ? base[c] : base2[c] + (int)r.range(3); break;
			}
		}
		if (!alpha && kind != 8)
			v[3] = 255;
		px[i] = color_rgba(clamp8(v[0]), clamp8(v[1]), clamp8(v[2]), clamp8(v[3]));
	}
	return n;
}

static void hash_stats(group& g, const astc_ldr::pixel_stats_t& ps)
{
	g.u32(ps.m_num_pixels);
	g.bytes(&ps.m_min, 4); g.bytes(&ps.m_max, 4);
	for (int c = 0; c < 4; c++) { g.f32(ps.m_min_f[c]); g.f32(ps.m_max_f[c]); g.f32(ps.m_mean_f[c]); g.f32(ps.m_mean_rel_axis4[c]); }
	for (int c = 0; c < 3; c++) { g.f32(ps.m_mean_rel_axis3[c]); g.f32(ps.m_zero_rel_axis3[c]); }
	g.u32(ps.m_has_alpha);
	for (int c = 0; c < 4; c++)
	{
		const auto& s = ps.m_rgba_stats[c];
		g.u32(s.m_n); g.f32(s.m_total); g.f32(s.m_avg); g.f32(s.m_var); g.f32(s.m_std_dev);
		g.f32(s.m_min); g.f32(s.m_max); g.f32(s.m_range);
	}
	for (uint32_t i = 0; i < ps.m_num_pixels; i++)
		for (int c = 0; c < 4; c++)
			g.f32(ps.m_pixels_f[i][c]);
	g.count++;
}

static const uint32_t LDR_CEMS[] = { 0, 4, 6, 8, 9, 10, 12, 13 };

static int pick_ccs(lcg& r, uint32_t cem)
{
	// valid dual plane channels per CEM (-1 = single plane)
	switch (cem)
	{
	case 0: return -1;
	case 4: return (r.next() & 1) ? 3 : -1;
	case 6: case 8: case 9: return (int)r.range(4) - 1;
	default: return (int)r.range(5) - 1;
	}
}

static uint32_t pick_weight_range(lcg& r)
{
	uint32_t v = r.range(13);
	return (v == 12) ? astc_helpers::BISE_64_LEVELS : v;
}

static uint32_t num_weight_levels(uint32_t wr)
{
	return (wr == astc_helpers::BISE_64_LEVELS) ? 65 : astc_helpers::get_ise_levels(wr);
}

int main()
{
	basisu_encoder_init(false, false);

	const uint32_t NUM_BLOCKS = 480;
	static color_rgba blocks[NUM_BLOCKS][144];
	static uint32_t block_n[NUM_BLOCKS];
	static astc_ldr::pixel_stats_t stats[NUM_BLOCKS];

	for (uint32_t b = 0; b < NUM_BLOCKS; b++)
	{
		block_n[b] = make_block(b, blocks[b]);
		stats[b].clear();
		stats[b].init(block_n[b], blocks[b]);
		hash_stats(G("pixel_stats"), stats[b]);
	}

	// blue contraction, weight ISE deltas
	{
		lcg r(1);
		group& g = G("blue_contract");
		for (uint32_t i = 0; i < 2000; i++)
		{
			color_rgba c;
			for (int k = 0; k < 4; k++) c[k] = (uint8_t)r.range(256);
			int eb = (int)r.range(256);
			bool clamped = false;
			color_rgba e = astc_ldr::blue_contract_enc(c, clamped, eb);
			color_rgba d = astc_ldr::blue_contract_dec(e.r, e.g, e.b, e.a);
			g.bytes(&e, 4); g.bytes(&d, 4); g.u32(clamped); g.count++;
		}
		group& gw = G("apply_delta_weight");
		for (uint32_t wr = 0; wr <= 11; wr++)
			for (uint32_t v = 0; v < astc_helpers::get_ise_levels(wr); v++)
				for (int d = -3; d <= 3; d++)
				{
					gw.u32((uint32_t)astc_ldr::apply_delta_to_bise_weight_val(wr, (int)v, d));
					gw.count++;
				}
	}

	// get_colors / get_colors_raw_weights / decode_endpoints
	{
		lcg r(2);
		group& gc = G("get_colors");
		group& gd = G("decode_endpoints");
		for (uint32_t i = 0; i < 3000; i++)
		{
			const uint32_t cem = LDR_CEMS[r.range(8)];
			const uint32_t er = 4 + r.range(17);
			const uint32_t levels = astc_helpers::get_ise_levels(er);
			uint8_t ev[8];
			for (int k = 0; k < 8; k++) ev[k] = (uint8_t)r.range(levels);
			const bool srgb = (r.next() & 1) != 0;
			color_rgba l, h;
			float scale = -1.0f;
			astc_ldr::decode_endpoints(cem, ev, er, l, h, &scale);
			gd.bytes(&l, 4); gd.bytes(&h, 4); gd.f32(scale); gd.count++;

			color_rgba cols[65];
			memset(cols, 0, sizeof(cols));
			const uint32_t wr = r.range(12);
			uint32_t n = astc_ldr::get_colors(cem, ev, er, wr, cols, srgb);
			gc.u32(n); gc.bytes(cols, n * 4);
			n = astc_ldr::get_colors_raw_weights(cem, ev, er, cols, srgb);
			gc.u32(n); gc.bytes(cols, n * 4);
			color_rgba l8, h8;
			for (int k = 0; k < 3; k++) l8[k] = (uint8_t)r.range(256);
			l8.a = (r.next() & 1) ? 255 : (uint8_t)r.range(256);
			for (int k = 0; k < 3; k++) h8[k] = (uint8_t)r.range(256);
			h8.a = (l8.a == 255) ? 255 : (uint8_t)r.range(256);
			n = astc_ldr::get_colors(l8, h8, wr, cols, srgb);
			gc.u32(n); gc.bytes(cols, n * 4);
			gc.count++;

			uint8_t ev20[8];
			for (int k = 0; k < 8; k++) ev20[k] = (uint8_t)r.range(256);
			astc_ldr::decode_endpoints_ise20(cem, ev20, l, h);
			gd.bytes(&l, 4); gd.bytes(&h, 4);
		}
	}

	// eval_solution / eval_solution_dp
	{
		lcg r(3);
		group& gs = G("eval_solution");
		group& gp = G("eval_solution_dp");
		for (uint32_t i = 0; i < 3000; i++)
		{
			const uint32_t b = r.range(NUM_BLOCKS);
			const auto& ps = stats[b];
			const uint32_t cem = LDR_CEMS[r.range(8)];
			const uint32_t er = 4 + r.range(17);
			const uint32_t levels = astc_helpers::get_ise_levels(er);
			uint8_t ev[8];
			for (int k = 0; k < 8; k++) ev[k] = (uint8_t)r.range(levels);
			const uint32_t wr = pick_weight_range(r);
			astc_ldr::cem_encode_params p;
			p.m_decode_mode_srgb = (r.next() & 1) != 0;
			if (r.next() & 1) { p.m_comp_weights[0] = 9; p.m_comp_weights[1] = 11; p.m_comp_weights[2] = 1; p.m_comp_weights[3] = 11; }
			p.m_use_exhaustive_weight_eval = (r.range(4) == 0);
			uint8_t forced0[144], forced1[144];
			const uint32_t nwl = num_weight_levels(wr);
			for (int k = 0; k < 144; k++) { forced0[k] = (uint8_t)r.range(nwl); forced1[k] = (uint8_t)r.range(nwl); }
			const uint32_t fmode = r.range(4);
			if (fmode == 1) p.m_pForced_weight_vals0 = forced0;
			if (fmode == 2) { p.m_pForced_weight_vals0 = forced0; p.m_pForced_weight_vals1 = forced1; }
			if (fmode == 3) p.m_pForced_weight_vals1 = forced1;

			uint8_t w0[144], w1[144];
			memset(w0, 0xCD, 144); memset(w1, 0xCD, 144);
			uint64_t e = astc_ldr::eval_solution(ps, cem, ev, er, w0, wr, p);
			gs.u64(e); gs.bytes(w0, 144); gs.count++;

			const uint32_t ccs = r.range(4);
			memset(w0, 0xCD, 144); memset(w1, 0xCD, 144);
			e = astc_ldr::eval_solution_dp(ps, cem, ccs, ev, er, w0, w1, wr, p);
			gp.u64(e); gp.bytes(w0, 144); gp.bytes(w1, 144); gp.count++;
		}
	}

	// cem_encode_pixels
	{
		lcg r(4);
		for (uint32_t ci = 0; ci < 8; ci++)
		{
			const uint32_t cem = LDR_CEMS[ci];
			char name[64];
			snprintf(name, sizeof(name), "cem_encode_pixels_%u", cem);
			group& g = G(name);
			snprintf(name, sizeof(name), "cem_encode_pixels_forced_%u", cem);
			group& gf = G(name);
			for (uint32_t i = 0; i < 4000; i++)
			{
				const uint32_t b = r.range(NUM_BLOCKS);
				const auto& ps = stats[b];
				const int ccs = pick_ccs(r, cem);
				const uint32_t er = 4 + r.range(17);
				const uint32_t wr = pick_weight_range(r);
				astc_ldr::cem_encode_params p;
				p.m_decode_mode_srgb = (r.next() & 1) != 0;
				if (r.next() & 1) { p.m_comp_weights[0] = 9; p.m_comp_weights[1] = 11; p.m_comp_weights[2] = 1; p.m_comp_weights[3] = 11; }
				p.m_use_exhaustive_weight_eval = (r.range(4) == 0);
				static const uint32_t LS[] = { 0, 1, 3, 5 };
				p.m_max_ls_passes = LS[r.range(4)];
				static const uint32_t RP[] = { 0, 0, 1, 5, 17, 20 };
				p.m_total_weight_refine_passes = RP[r.range(6)];
				p.m_worst_weight_nudging_flag = (r.next() & 1) != 0;
				p.m_endpoint_refinement_flag = (r.next() & 1) != 0;
				const bool use_bc = (r.next() & 1) != 0;
				const uint32_t cm = r.range(8);
				const uint64_t cur_err = (cm == 0) ? (uint64_t)r.range(2000) : UINT64_MAX;
				const bool forced = (r.range(5) == 0);
				uint8_t forced0[144], forced1[144];
				const uint32_t nwl = num_weight_levels(wr);
				for (int k = 0; k < 144; k++) { forced0[k] = (uint8_t)r.range(nwl); forced1[k] = (uint8_t)r.range(nwl); }
				if (forced)
				{
					p.m_pForced_weight_vals0 = forced0;
					if (ccs >= 0) p.m_pForced_weight_vals1 = forced1;
				}

				uint8_t ev[8], w0[144], w1[144];
				memset(ev, 0xCD, 8); memset(w0, 0xCD, 144); memset(w1, 0xCD, 144);
				if (cm == 0)
				{
					// The encoders start refinement from the caller's buffers when not improved: keep them valid.
					const uint32_t el = astc_helpers::get_ise_levels(er);
					for (int k = 0; k < 8; k++) ev[k] = (uint8_t)(forced1[k] % el);
					memcpy(w0, forced0, 144); memcpy(w1, forced1, 144);
				}
				bool try_direct = true;
				uint64_t e = astc_ldr::cem_encode_pixels(cem, ccs, ps, p, er, wr, ev, w0, (ccs >= 0) ? w1 : nullptr, cur_err, use_bc, &try_direct);
				group& gg = forced ? gf : g;
				gg.u64(e); gg.bytes(ev, 8); gg.bytes(w0, 144); gg.bytes(w1, 144); gg.u32(try_direct); gg.count++;
			}
		}
	}

	// cem_surrogate_encode_pixels
	{
		lcg r(5);
		static const uint32_t SCEMS[] = { 0, 4, 6, 8, 10, 12, 9 };
		for (uint32_t ci = 0; ci < 7; ci++)
		{
			const uint32_t cem = SCEMS[ci];
			char name[64];
			snprintf(name, sizeof(name), "cem_surrogate_encode_pixels_%u", cem);
			group& g = G(name);
			for (uint32_t i = 0; i < 3000; i++)
			{
				const uint32_t b = r.range(NUM_BLOCKS);
				const auto& ps = stats[b];
				int ccs = pick_ccs(r, cem);
				if ((cem == 4) && (ccs >= 0)) ccs = 3;
				const uint32_t er = 4 + r.range(17);
				const uint32_t wr = pick_weight_range(r);
				astc_ldr::cem_encode_params p;
				if (r.next() & 1) { p.m_comp_weights[0] = 9; p.m_comp_weights[1] = 11; p.m_comp_weights[2] = 1; p.m_comp_weights[3] = 11; }
				const uint32_t flags = r.range(4);
				vec4F l(0.0f), h(0.0f);
				float s = 0.0f;
				float w0[144], w1[144];
				for (int k = 0; k < 144; k++) { w0[k] = -1.0f; w1[k] = -1.0f; }
				float e = astc_ldr::cem_surrogate_encode_pixels(cem, ccs, ps, p, er, wr, l, h, s, w0, w1, flags);
				g.f32(e); g.f32(s);
				for (int c = 0; c < 4; c++) { g.f32(l[c]); g.f32(h[c]); }
				for (int k = 0; k < 144; k++) { g.f32(w0[k]); g.f32(w1[k]); }
				g.count++;
			}
		}

		group& gq = G("surrogate_quant");
		for (uint32_t i = 0; i < 2000; i++)
		{
			const float e = (float)r.range(1000001) * (1.0f / 1000000.0f);
			const uint32_t levels = astc_helpers::get_ise_levels(4 + r.range(17));
			const uint32_t flags = r.range(2);
			gq.f32(astc_ldr::surrogate_quant_endpoint_val(e, levels, flags));
			vec4F v(e, 1.0f - e, e * .5f, e * e);
			vec4F q = astc_ldr::surrogate_quant_endpoint(v, levels, flags);
			for (int c = 0; c < 4; c++) gq.f32(q[c]);
			gq.count++;
		}
	}

	for (const auto& kv : g_groups)
		printf("%s\t%016llx\t%u\n", kv.first.c_str(), (unsigned long long)kv.second.h, kv.second.count);
	return 0;
}

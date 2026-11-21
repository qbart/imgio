// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0
// Cross-check harness for imgio's XUBC7 internals vs the C++ transcoder: random BC7 blocks
// through bc7f::fast_pack_astc, bc7u logical pack/unpack, endpoint DPCM, the XBC7 weight DCT
// and the weight predictors. Prints FNV-1a hashes that src/basis/xubc7/decoder.rs tests
// against. Build like dump.cpp (see build.sh), e.g.:
//   g++ -O2 -std=c++17 -I../../ref/basisu/transcoder -I../../ref/basisu/zstd \
//     -DBASISD_SUPPORT_KTX2_ZSTD=1 xubc7_check.cpp ../../ref/basisu/transcoder/basisu_transcoder.cpp \
//     dump.zstd.o -o xubc7_check
#include "basisu_transcoder.h"
#include "basisu_transcoder_internal.h"
#include "basisu_xbc7_decoder.h"
#include <cstdio>

using namespace basist;

static uint64_t g_rng;
static uint64_t rnd()
{
	uint64_t x = g_rng;
	x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
	g_rng = x;
	return x * 0x2545F4914F6CDD1Dull;
}

struct fnv
{
	uint64_t h = 0xcbf29ce484222325ull;
	void b(uint8_t v) { h ^= v; h *= 0x100000001b3ull; }
	void bytes(const void* p, size_t n) { for (size_t i = 0; i < n; i++) b(((const uint8_t*)p)[i]); }
};

static void rand_block(uint8_t blk[16], uint32_t mode)
{
	for (int i = 0; i < 16; i++) blk[i] = (uint8_t)rnd();
	blk[0] = (uint8_t)((blk[0] & ~((2u << mode) - 1)) | (1u << mode));
}

static void hash_log(fnv& h, const bc7u::log_bc7_block& l)
{
	h.bytes(&l.m_endpoints[0][0][0], sizeof(l.m_endpoints));
	h.bytes(&l.m_weights[0][0], sizeof(l.m_weights));
	h.bytes(l.m_pbits, sizeof(l.m_pbits));
}

int main()
{
	basisu_transcoder_init();
	const uint32_t N = 100000;

	// 1. fast_pack_astc per BC7 mode.
	for (uint32_t mode = 0; mode < 8; mode++)
	{
		g_rng = 0x1234567 + mode;
		fnv h;
		for (uint32_t i = 0; i < N; i++)
		{
			uint8_t blk[16], astc[16];
			rand_block(blk, mode);
			bc7_block b;
			memcpy(&b, blk, 16);
			bool ok = bc7f::fast_pack_astc(astc, b, bc7f::cPackBC7FlagDefault);
			h.b(ok);
			h.bytes(astc, 16);
		}
		printf("astc %u %016llx\n", mode, (unsigned long long)h.h);
	}

	// 2. logical unpack / pack / pixels.
	{
		g_rng = 0xabcdef;
		fnv h;
		for (uint32_t i = 0; i < N; i++)
		{
			uint8_t blk[16], out[16];
			rand_block(blk, i & 7);
			bc7u::log_bc7_block l;
			bc7u::unpack_bc7(blk, l);
			bc7u::pack_bc7(l, out);
			color_rgba px[16];
			bc7u::unpack_bc7(l, px);
			h.bytes(out, 16);
			h.bytes(px, 64);
			color_rgba t;
			bc7u::unpack_bc7_texel(l, t, i & 3, (i >> 2) & 3);
			h.bytes(&t, 4);
		}
		printf("log %016llx\n", (unsigned long long)h.h);
	}

	// 3. endpoint DPCM decode.
	{
		g_rng = 0x5eed;
		fnv h;
		for (uint32_t i = 0; i < N; i++)
		{
			uint8_t a[16], b[16];
			rand_block(a, (uint32_t)(rnd() & 7));
			rand_block(b, (uint32_t)(rnd() & 7));
			bc7u::log_bc7_block pred, cur;
			bc7u::unpack_bc7(a, pred);
			bc7u::unpack_bc7(b, cur);
			uint32_t ps = (uint32_t)(rnd() % pred.m_num_partitions);
			for (uint32_t s = 0; s < cur.m_num_partitions; s++)
			{
				uint8_t res[8], pb[2];
				for (int k = 0; k < 8; k++) res[k] = (uint8_t)rnd();
				pb[0] = (uint8_t)(rnd() & 1); pb[1] = (uint8_t)(rnd() & 1);
				uint32_t nr, np;
				bc7u::endpoint_dpcm(true, pred, ps, cur, s, res, nr, pb, np);
			}
			hash_log(h, cur);
		}
		printf("dpcm %016llx\n", (unsigned long long)h.h);
	}

	// 4. weight grid DCT inverse.
	{
		g_rng = 0xdc7;
		fnv h;
		xbc7::xbc7_weight_grid_dct_fixed dct;
		dct.init();
		xbc7::fxvec work;
		for (uint32_t i = 0; i < N; i++)
		{
			uint8_t a[16];
			rand_block(a, (uint32_t)(rnd() & 7));
			bc7u::log_bc7_block l;
			bc7u::unpack_bc7(a, l);
			uint32_t p = (uint32_t)(rnd() % l.m_num_planes);
			int preds[16];
			for (int k = 0; k < 16; k++) preds[k] = (int)(rnd() % 65);
			bool use_preds = (rnd() & 1) != 0;
			uint32_t q = 1 + (uint32_t)(rnd() % 100);
			xbc7::dct_syms syms;
			syms.clear();
			syms.m_dc = (int16_t)((int)(rnd() % 511) - 255);
			uint32_t zig = 1;
			while (zig < 16)
			{
				if ((rnd() % 4) == 0)
				{
					xbc7::coeff cf; cf.m_num_zeros = (int16_t)(16 - zig); cf.m_coeff = INT16_MAX;
					syms.m_ac_vals.push_back(cf);
					break;
				}
				uint32_t run = (uint32_t)(rnd() % (16 - zig));
				int mag = 1 + (int)(rnd() % 255);
				xbc7::coeff cf; cf.m_num_zeros = (int16_t)run; cf.m_coeff = (int16_t)((rnd() & 1) ? -mag : mag);
				syms.m_ac_vals.push_back(cf);
				zig += run + 1;
			}
			bool ok = dct.inverse(fixed16_16::from_int(q), p, use_preds ? preds : nullptr, syms, l, work);
			h.b(ok);
			h.bytes(&l.m_weights[0][0], sizeof(l.m_weights));
		}
		printf("dct %016llx\n", (unsigned long long)h.h);
	}

	// 5. weight predictors on a 6x6 grid.
	{
		g_rng = 0x9e7;
		fnv h;
		basisu::vector2D<bc7u::log_bc7_block> grid(6, 6);
		for (uint32_t i = 0; i < N / 10; i++)
		{
			for (uint32_t y = 0; y < 6; y++)
				for (uint32_t x = 0; x < 6; x++)
				{
					uint8_t a[16];
					rand_block(a, (uint32_t)(rnd() & 7));
					bc7u::unpack_bc7(a, grid(x, y));
				}
			for (uint32_t k = 0; k < 20; k++)
			{
				uint32_t bx = (uint32_t)(rnd() % 6), by = (uint32_t)(rnd() % 6);
				uint32_t cand = 1 + (uint32_t)(rnd() % (xbc7::cTotalCandidates - 1));
				uint32_t amp = (uint32_t)(rnd() & 3);
				uint32_t p = (uint32_t)(rnd() & 1);
				xbc7::tile_bounds tile = { 0, 0, 5, 5 };
				int out[16] = {};
				bool ok = xbc7::eval_weight_predictor(cand, amp, bx, by, 6, tile, grid, p, out);
				h.b(ok);
				if (ok) h.bytes(out, sizeof(out));
			}
		}
		printf("pred %016llx\n", (unsigned long long)h.h);
	}
	return 0;
}

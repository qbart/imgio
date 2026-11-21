// SPDX-FileCopyrightText: 2026 imgio contributors
// SPDX-License-Identifier: Apache-2.0
//
// Writes the OpenVDB test files of testdata/file/vdb/ with the C++ OpenVDB library and, for
// every scalar floating-point grid, the reference dense values of its active voxel bounding box
// as read back by OpenVDB (`<file>.<grid name>.f32`, little-endian f32, x fastest, then y,
// then z) plus a manifest (`expected.txt`). Offline dev tool (scripts/gen_file_testdata_jxlj2kvdb.py
// builds it); never linked into imgio.
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

using namespace openvdb;

static std::string g_dir;
static FILE* g_manifest = nullptr;

static float pattern(int x, int y, int z) {
    return 0.125f * float(x) - 0.375f * float(y) + 0.0625f * float(z * z) + 0.3f;
}

template <typename GridT>
static void dumpDense(const GridT& grid, const std::string& path, const CoordBBox& bb) {
    std::ofstream out(path, std::ios::binary);
    auto acc = grid.getConstAccessor();
    for (int z = bb.min().z(); z <= bb.max().z(); ++z)
        for (int y = bb.min().y(); y <= bb.max().y(); ++y)
            for (int x = bb.min().x(); x <= bb.max().x(); ++x) {
                float v = float(acc.getValue(Coord(x, y, z)));
                out.write(reinterpret_cast<const char*>(&v), 4);
            }
}

// Reads `file` back and records every grid in the manifest (OpenVDB lists them by name):
// grid <file> <name> <type> <half> <map> <box|empty> <bbox min max> <16 matrix values> <dense|->
static void record(const std::string& file) {
    io::File f(g_dir + "/" + file);
    f.open();
    GridPtrVecPtr grids = f.getGrids();
    for (GridBase::Ptr g : *grids) {
        CoordBBox bb = g->evalActiveVoxelBoundingBox();
        std::string dense = "-";
        bool empty = bb.empty();
        Vec3i lo = empty ? Vec3i(0) : bb.min().asVec3i();
        Vec3i hi = empty ? Vec3i(0) : bb.max().asVec3i();
        bool scalar = g->isType<FloatGrid>() || g->isType<DoubleGrid>();
        if (scalar && !empty && bb.volume() <= 64 * 1024) {
            dense = file + "." + g->getName() + ".f32";
            if (auto fg = gridPtrCast<FloatGrid>(g)) dumpDense(*fg, g_dir + "/" + dense, bb);
            if (auto dg = gridPtrCast<DoubleGrid>(g)) dumpDense(*dg, g_dir + "/" + dense, bb);
        }
        math::Mat4d m = g->transform().baseMap()->getAffineMap()->getMat4();
        std::fprintf(g_manifest, "grid %s %s %s %d %s %s %d %d %d %d %d %d", file.c_str(),
            g->getName().c_str(), g->type().c_str(), int(g->saveFloatAsHalf()),
            g->transform().baseMap()->type().c_str(), empty ? "empty" : "box", lo.x(), lo.y(), lo.z(),
            hi.x(), hi.y(), hi.z());
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) std::fprintf(g_manifest, " %.17g", m(r, c));
        std::fprintf(g_manifest, " %s\n", dense.c_str());
    }
    f.close();
}

static void write(const std::string& file, const GridPtrVec& grids, uint32_t compression) {
    io::File f(g_dir + "/" + file);
    f.setCompression(compression);
    f.write(grids);
    f.close();
    record(file);
}

static FloatGrid::Ptr scattered() {
    FloatGrid::Ptr g = FloatGrid::create(0.0f);
    g->setName("density");
    math::Transform::Ptr xf = math::Transform::createLinearTransform(0.5);
    xf->postTranslate(Vec3d(1.0, 2.0, 3.0));
    g->setTransform(xf);
    auto acc = g->getAccessor();
    // Negative coordinates, but a single root child (an uncompressed 32^3 internal node
    // alone is 140 KB).
    for (int z = -10; z <= -1; ++z)
        for (int y = -11; y <= -1; ++y)
            for (int x = -16; x <= -1; ++x)
                if ((x + 2 * y + 3 * z) % 3 != 0) acc.setValue(Coord(x, y, z), pattern(x, y, z));
    return g;
}

// One leaf per inactive-value case of the mask compression (io/Compression.h).
static FloatGrid::Ptr inactiveCases() {
    FloatGrid::Ptr g = FloatGrid::create(0.25f);
    g->setName("density");
    auto acc = g->getAccessor();
    for (int leaf = 0; leaf < 6; ++leaf) {
        Coord o((leaf % 4) * 8, (leaf / 4) * 8, 0);
        for (int i = 0; i < 6; ++i) acc.setValue(o + Coord(i, i % 3, i % 2), pattern(i, leaf, i));
        Coord a = o + Coord(7, 7, 7), b = o + Coord(6, 7, 7), c = o + Coord(5, 7, 7);
        switch (leaf) {
            case 0: break;                                   // background only
            case 1: acc.setValueOff(a, -0.25f); break;       // -background (+ background)
            case 2: acc.setValueOff(a, 7.5f); break;         // background + one value
            case 3:                                          // two non-background values
                for (int i = 0; i < 512; ++i) {
                    Coord p = o + Coord(i >> 6, (i >> 3) & 7, i & 7);
                    if (!acc.isValueOn(p)) acc.setValueOff(p, (i & 1) ? 1.5f : -2.0f);
                }
                break;
            case 4:                                          // one non-background value only
                for (int i = 0; i < 512; ++i) {
                    Coord p = o + Coord(i >> 6, (i >> 3) & 7, i & 7);
                    if (!acc.isValueOn(p)) acc.setValueOff(p, 4.0f);
                }
                break;
            default:                                         // more than two: all values
                acc.setValueOff(a, 9.0f);
                acc.setValueOff(b, -3.0f);
                acc.setValueOff(c, 0.5f);
        }
    }
    return g;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <out dir>\n", argv[0]);
        return 1;
    }
    initialize();
    g_dir = argv[1];
    g_manifest = std::fopen((g_dir + "/expected.txt").c_str(), "w");
    std::fprintf(g_manifest, "# generated by tools/vdb-golden/gen.cpp with OpenVDB %s\n",
        OPENVDB_LIBRARY_VERSION_STRING);

    write("float_none.vdb", {scattered()}, io::COMPRESS_NONE);
    write("float_zip.vdb", {scattered()}, io::COMPRESS_ZIP);
    write("float_active_mask.vdb", {inactiveCases()}, io::COMPRESS_ACTIVE_MASK);
    write("float_zip_mask.vdb", {inactiveCases()}, io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK);

    // Level set: zip is dropped for level sets, so this is active-mask only; inactive values
    // inside the sphere are -background.
    FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(3.0f, Vec3f(10.5f, 10.0f, 9.5f), 1.0f, 2.0f);
    ls->setName("surface");
    write("levelset.vdb", {ls}, io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK);

    // Half float storage, an active 8^3 tile and an inactive non-background tile.
    {
        FloatGrid::Ptr g = FloatGrid::create(0.0f);
        g->setName("smoke");
        g->setSaveFloatAsHalf(true);
        g->fill(CoordBBox(Coord(8, 8, 8), Coord(15, 15, 15)), 0.75f, true);
        g->fill(CoordBBox(Coord(16, 0, 0), Coord(23, 7, 7)), 3.0f, false);
        auto acc = g->getAccessor();
        for (int i = 0; i < 24; ++i) acc.setValue(Coord(i, (i * 5) % 16, (i * 3) % 16), 0.1f * float(i) - 1.0f);
        write("half_tiles.vdb", {g}, io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK);
    }

    // Several grid types, an affine transform, a root tile, an instanced grid and a sparse grid
    // whose bounding box is too large to densify.
    {
        Int32Grid::Ptr ids = Int32Grid::create(-1);
        ids->setName("ids");
        ids->fill(CoordBBox(Coord(4096, 0, 0), Coord(8191, 4095, 4095)), 7, true);
        ids->tree().setValue(Coord(1, 2, 3), 42);

        Vec3SGrid::Ptr vel = Vec3SGrid::create(Vec3s(0.0f));
        vel->setName("vel");
        for (int i = 0; i < 10; ++i) vel->tree().setValue(Coord(i, i, 2 * i), Vec3s(float(i), 1.0f, -float(i)));

        BoolGrid::Ptr flags = BoolGrid::create(false);
        flags->setName("flags");
        flags->tree().setValue(Coord(3, 4, 5), true);
        flags->tree().setValue(Coord(9, 9, 9), true);

        DoubleGrid::Ptr temp = DoubleGrid::create(20.0);
        temp->setName("temperature");
        math::Transform::Ptr xf = math::Transform::createLinearTransform(0.25);
        xf->preRotate(0.3, math::Z_AXIS);
        xf->postTranslate(Vec3d(-1.0, 0.5, 2.0));
        temp->setTransform(xf);
        for (int z = 0; z < 6; ++z)
            for (int y = 0; y < 5; ++y)
                for (int x = 0; x < 9; ++x)
                    if ((x * y + z) % 4 != 1) temp->tree().setValue(Coord(x, y, z), 20.0 + 0.1 * (x + 10 * y + 100 * z));

        FloatGrid::Ptr dens = scattered();
        FloatGrid::Ptr inst = dens->copy();  // shares the tree: written as an instance
        inst->setName("density_instance");
        inst->setTransform(math::Transform::createLinearTransform(2.0));

        FloatGrid::Ptr sparse = FloatGrid::create(0.0f);
        sparse->setName("sparse");
        sparse->tree().setValue(Coord(0, 0, 0), 1.0f);
        sparse->tree().setValue(Coord(1000, 1000, 1000), 2.0f);

        write("multi.vdb", {ids, vel, flags, temp, dens, inst, sparse},
            io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK);
    }

    // Blosc (the OpenVDB default): listed, but not densified by imgio.
    write("blosc.vdb", {scattered()}, io::COMPRESS_BLOSC | io::COMPRESS_ACTIVE_MASK);

    std::fclose(g_manifest);
    return 0;
}

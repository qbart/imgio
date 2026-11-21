//! OpenVDB reader vs the C++ OpenVDB 10 library (testdata/file/vdb, written and read back by
//! `tools/vdb-golden/gen.cpp` via `scripts/gen_file_testdata_jxlj2kvdb.py`). `expected.txt`:
//! `grid <file> <name> <type> <half> <map> <box|empty> <min xyz> <max xyz> <4x4 matrix> <dense>`
//! where `<dense>` holds OpenVDB's `getValue` over the active bounding box (f32, x fastest).

#[cfg(not(feature = "vdb"))]
#[test]
fn without_feature_unsupported() {
    let data = [0x20, 0x42, 0x44, 0x56, 0, 0, 0, 0];
    assert_eq!(
        imgio::file::vdb::decode(&data).err(),
        Some(imgio::Error::Unsupported)
    );
    assert_eq!(
        imgio::file::vdb::grids(&data).err(),
        Some(imgio::Error::Unsupported)
    );
}

#[cfg(feature = "vdb")]
mod with_feature {
    use imgio::file::vdb::{self, DecodeOptions, ValueType};
    use imgio::file::{self, FileFormat};
    use imgio::{Error, Format};
    use std::error::Error as StdError;

    type R = Result<(), Box<dyn StdError>>;

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/file/vdb/");
    const FILES: [&str; 8] = [
        "float_none.vdb",
        "float_zip.vdb",
        "float_active_mask.vdb",
        "float_zip_mask.vdb",
        "levelset.vdb",
        "half_tiles.vdb",
        "multi.vdb",
        "blosc.vdb",
    ];

    struct Entry {
        file: String,
        name: String,
        type_name: String,
        half: bool,
        map: String,
        bbox: Option<([i32; 3], [i32; 3])>,
        matrix: [[f64; 4]; 4],
        dense: Option<String>,
    }

    fn manifest() -> Result<Vec<Entry>, Box<dyn StdError>> {
        let text = std::fs::read_to_string(format!("{DIR}expected.txt"))?;
        let mut out = Vec::new();
        for line in text.lines().filter(|l| l.starts_with("grid ")) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 30 {
                return Err(format!("bad line {line}").into());
            }
            let i = |k: usize| f[k].parse::<i32>();
            let bbox = match f[6] {
                "box" => Some(([i(7)?, i(8)?, i(9)?], [i(10)?, i(11)?, i(12)?])),
                _ => None,
            };
            let mut matrix = [[0.0; 4]; 4];
            for (k, v) in matrix.iter_mut().flatten().enumerate() {
                *v = f[13 + k].parse()?;
            }
            out.push(Entry {
                file: f[1].into(),
                name: f[2].into(),
                type_name: f[3].into(),
                half: f[4] == "1",
                map: f[5].into(),
                bbox,
                matrix,
                dense: (f[29] != "-").then(|| f[29].to_string()),
            });
        }
        Ok(out)
    }

    /// File bytes (the uncompressed file is stored zlib-wrapped).
    fn load(name: &str) -> Result<Vec<u8>, Box<dyn StdError>> {
        match std::fs::read(format!("{DIR}{name}")) {
            Ok(d) => Ok(d),
            Err(_) => {
                let z = std::fs::read(format!("{DIR}{name}.zz"))?;
                miniz_oxide::inflate::decompress_to_vec_zlib(&z)
                    .map_err(|e| format!("{name}: {e:?}").into())
            }
        }
    }

    fn floats(b: &[u8]) -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn index(list: &[vdb::GridInfo], name: &str) -> Result<usize, Box<dyn StdError>> {
        list.iter()
            .position(|g| g.name == name)
            .ok_or_else(|| format!("no grid {name}").into())
    }

    #[test]
    fn lists_grids_like_openvdb() -> R {
        let all = manifest()?;
        for file in FILES {
            let data = load(file)?;
            assert_eq!(FileFormat::detect(&data), Some(FileFormat::Vdb), "{file}");
            let list = vdb::grids(&data)?;
            let entries: Vec<&Entry> = all.iter().filter(|e| e.file == file).collect();
            assert_eq!(list.len(), entries.len(), "{file}");
            for e in entries {
                let g = &list[index(&list, &e.name)?];
                let ctx = format!("{file}/{}", e.name);
                assert_eq!(g.type_name, e.type_name, "{ctx}");
                assert_eq!(g.half_float, e.half, "{ctx}");
                assert_eq!(g.active_bbox, e.bbox, "{ctx}");
                let t = g.transform.as_ref().ok_or("transform")?;
                assert_eq!(t.map, e.map, "{ctx}");
                assert_eq!(t.matrix, Some(e.matrix), "{ctx}");
            }
        }
        Ok(())
    }

    #[test]
    fn grid_details() -> R {
        let data = load("multi.vdb")?;
        let list = vdb::grids(&data)?;
        // File order, as written.
        let names: Vec<&str> = list.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "ids",
                "vel",
                "flags",
                "temperature",
                "density",
                "density_instance",
                "sparse"
            ]
        );
        let ty: Vec<ValueType> = list.iter().map(|g| g.value_type).collect();
        assert_eq!(
            ty,
            [
                ValueType::Int32,
                ValueType::Vec3Float,
                ValueType::Bool,
                ValueType::Double,
                ValueType::Float,
                ValueType::Float,
                ValueType::Float
            ]
        );
        let inst = &list[index(&list, "density_instance")?];
        assert_eq!(inst.instance_of.as_deref(), Some("density"));
        assert_eq!(
            list[0].compression,
            vdb::COMPRESS_ZIP | vdb::COMPRESS_ACTIVE_MASK
        );
        let ls = vdb::grids(&load("levelset.vdb")?)?;
        assert_eq!(ls[0].class, "level set");
        // Zip is never used for level sets.
        assert_eq!(ls[0].compression, vdb::COMPRESS_ACTIVE_MASK);
        let blosc = vdb::grids(&load("blosc.vdb")?)?;
        assert_eq!(
            blosc[0].compression & vdb::COMPRESS_BLOSC,
            vdb::COMPRESS_BLOSC
        );
        Ok(())
    }

    #[test]
    fn densify_matches_openvdb() -> R {
        let mut checked = 0;
        for e in manifest()? {
            let Some(dense) = &e.dense else { continue };
            let data = load(&e.file)?;
            let list = vdb::grids(&data)?;
            let gi = index(&list, &e.name)?;
            let ctx = format!("{}/{}", e.file, e.name);
            if e.file == "blosc.vdb" {
                assert_eq!(
                    vdb::densify(&data, gi, &DecodeOptions::default()).err(),
                    Some(Error::Unsupported),
                    "{ctx}"
                );
                continue;
            }
            let vol = vdb::densify(&data, gi, &DecodeOptions::default())?;
            let (min, max) = e.bbox.ok_or("bbox")?;
            assert_eq!(vol.origin, min, "{ctx}");
            assert_eq!(
                [vol.width, vol.height, vol.depth],
                [0, 1, 2].map(|k| (max[k] - min[k] + 1) as u32),
                "{ctx}"
            );
            let want = floats(&std::fs::read(format!("{DIR}{dense}"))?);
            let got = floats(&vol.data);
            assert_eq!(got.len(), want.len(), "{ctx}");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "{ctx} voxel {i}: {g} vs {w}");
            }
            checked += 1;
        }
        assert_eq!(checked, 9);
        Ok(())
    }

    #[test]
    fn decode_is_middle_slice_of_first_float_grid() -> R {
        let all = manifest()?;
        for file in FILES {
            let data = load(file)?;
            let list = vdb::grids(&data)?;
            let first = list
                .iter()
                .find(|g| matches!(g.value_type, ValueType::Float | ValueType::Double))
                .ok_or("float grid")?;
            if file == "blosc.vdb" {
                assert_eq!(vdb::decode(&data).err(), Some(Error::Unsupported));
                continue;
            }
            let e = all
                .iter()
                .find(|e| e.file == file && e.name == first.name)
                .ok_or("entry")?;
            let (min, max) = e.bbox.ok_or("bbox")?;
            let (w, h) = (
                (max[0] - min[0] + 1) as usize,
                (max[1] - min[1] + 1) as usize,
            );
            let z = ((max[2] - min[2]) / 2) as usize;
            let want = floats(&std::fs::read(format!(
                "{DIR}{}",
                e.dense.as_ref().ok_or("dense")?
            ))?);
            let img = file::decode(&data)?;
            assert_eq!(img.format, Format::R32Float);
            assert_eq!((img.width as usize, img.height as usize), (w, h), "{file}");
            assert_eq!(
                floats(&img.data),
                want[z * w * h..(z + 1) * w * h],
                "{file}"
            );
            let anim = file::decode_animation(&data)?;
            assert_eq!(anim.frames.len(), 1);
        }
        Ok(())
    }

    #[test]
    fn regions_tiles_and_limits() -> R {
        let data = load("multi.vdb")?;
        let list = vdb::grids(&data)?;
        let opts = DecodeOptions::default();
        // A region beyond the active voxels: background outside.
        let d = index(&list, "density")?;
        let vol = vdb::densify_region(&data, d, [-20, -12, -11], [2, 0, 1], &opts)?;
        assert_eq!((vol.width, vol.height, vol.depth), (23, 13, 13));
        let full = vdb::densify(&data, d, &opts)?;
        for (x, y, z) in [(-16, -11, -10), (-1, -1, -1), (-5, -7, -3)] {
            assert_eq!(vol.get(x, y, z), full.get(x, y, z));
        }
        assert_eq!(vol.get(2, 0, 1), Some(0.0));
        assert_eq!(vol.get(-20, -12, -11), Some(0.0));
        assert_eq!(vol.get(3, 0, 0), None);
        // The instance shares the tree.
        let inst = vdb::densify(&data, index(&list, "density_instance")?, &opts)?;
        assert_eq!(inst.data[..], full.data[..]);
        // Too many voxels for the limit, a small region works.
        let s = index(&list, "sparse")?;
        assert_eq!(list[s].active_bbox, Some(([0, 0, 0], [1000, 1000, 1000])));
        assert_eq!(
            vdb::densify(&data, s, &opts).err(),
            Some(Error::InvalidArgument)
        );
        let v = vdb::densify_region(&data, s, [999, 999, 999], [1000, 1000, 1000], &opts)?;
        assert_eq!(floats(&v.data), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
        // Non-scalar grids and bad indices.
        for name in ["ids", "vel", "flags"] {
            assert_eq!(
                vdb::densify(&data, index(&list, name)?, &opts).err(),
                Some(Error::Unsupported),
                "{name}"
            );
        }
        assert_eq!(
            vdb::densify(&data, 99, &opts).err(),
            Some(Error::InvalidArgument)
        );
        // Half tiles: active 8^3 tile and an inactive non-background tile.
        let data = load("half_tiles.vdb")?;
        let vol = vdb::densify(&data, 0, &opts)?;
        assert_eq!(vol.get(12, 12, 12), Some(0.75));
        assert_eq!(vol.get(20, 3, 3), Some(3.0));
        // Level set: -background inside, +background outside.
        let data = load("levelset.vdb")?;
        let vol = vdb::densify_region(&data, 0, [0, 0, 0], [20, 20, 20], &opts)?;
        assert_eq!(vol.get(10, 10, 10), Some(-2.0));
        assert_eq!(vol.get(0, 0, 0), Some(2.0));
        assert_eq!(
            file::encode(FileFormat::Vdb, &full_surface()?).err(),
            Some(Error::Unsupported)
        );
        Ok(())
    }

    fn full_surface() -> Result<imgio::Surface<'static>, Box<dyn StdError>> {
        static PIX: [u8; 16] = [0; 16];
        Ok(imgio::Surface::new(Format::Rgba8Unorm, 2, 2, &PIX)?)
    }

    #[test]
    fn corrupt_input_never_panics() -> R {
        let opts = DecodeOptions {
            max_voxels: 1 << 16,
        };
        let run = |d: &[u8]| {
            if let Ok(list) = vdb::grids(d) {
                for i in 0..list.len() {
                    let _ = vdb::densify(d, i, &opts);
                }
            }
            let _ = vdb::decode_with(d, &opts);
        };
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for file in FILES {
            let data = load(file)?;
            for k in 0..200 {
                let len = data.len() * k / 200;
                run(&data[..len]);
            }
            for _ in 0..300 {
                let mut d = data.clone();
                for _ in 0..1 + rnd() % 4 {
                    let i = (rnd() % d.len() as u64) as usize;
                    d[i] ^= 1 << (rnd() % 8);
                }
                run(&d);
            }
        }
        Ok(())
    }
}

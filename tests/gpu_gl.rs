// SPDX-FileCopyrightText: 2010-2026 Mark Callow, The Khronos Group Inc. and KTX-Software contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from ktx-rs `tests/glloader.rs`, `tests/glloader_es.rs`; see NOTICE.md.
//! OpenGL upload (`imgio::gpu::gl`, port of `glloader.c`) on Mesa llvmpipe through surfaceless
//! EGL: a desktop core profile context with commands from `eglGetProcAddress` (read back with
//! `glGetTexImage` / `glGetCompressedTexImage`) and a GL ES 3.0 context with commands from the
//! process symbols (read back through a framebuffer). Skipped without EGL.
#![cfg(all(feature = "opengl", not(target_family = "wasm")))]
#![allow(unsafe_code)]

mod gpu_common;

use std::ffi::{CStr, c_void};

use gpu_common::egl::Gl;
use gpu_common::{DIR, TestResult, level_data, levels, pad_rows, pattern, texture};
use imgio::dds::DdsFile;
use imgio::gpu::gl::{
    GL_INVALID_ENUM, GL_TEXTURE_2D, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_3D, GL_TEXTURE_CUBE_MAP,
    GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_CUBE_MAP_POSITIVE_X, GLenum, GLuint, GlContext,
    GlDecodeFallback, GlUploadOptions,
};
use imgio::gpu::{TextureData, TextureSource};
use imgio::ktx2::Ktx2File;
use imgio::{Error, Format};

const GL_TEXTURE_WIDTH: GLenum = 0x1000;
const GL_PACK_ALIGNMENT: GLenum = 0x0D05;
const GL_UNPACK_ALIGNMENT: GLenum = 0x0CF5;
const GL_RED: GLenum = 0x1903;
const GL_RGBA: GLenum = 0x1908;
const GL_UNSIGNED_BYTE: GLenum = 0x1401;
const GL_HALF_FLOAT: GLenum = 0x140B;
const GL_FRAMEBUFFER: GLenum = 0x8D40;
const GL_COLOR_ATTACHMENT0: GLenum = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: GLenum = 0x8CD5;

type GetTexImage = unsafe extern "system" fn(GLenum, i32, GLenum, GLenum, *mut c_void);
type GetCompressedTexImage = unsafe extern "system" fn(GLenum, i32, *mut c_void);
type PixelStorei = unsafe extern "system" fn(GLenum, i32);
type GetIntegerv = unsafe extern "system" fn(GLenum, *mut i32);
type GetTexLevelParameteriv = unsafe extern "system" fn(GLenum, i32, GLenum, *mut i32);
type BindTexture = unsafe extern "system" fn(GLenum, GLuint);
type GenTextures = unsafe extern "system" fn(i32, *mut GLuint);
type GenFramebuffers = unsafe extern "system" fn(i32, *mut GLuint);
type BindFramebuffer = unsafe extern "system" fn(GLenum, GLuint);
type FramebufferTexture2D = unsafe extern "system" fn(GLenum, GLenum, GLenum, GLuint, i32);
type CheckFramebufferStatus = unsafe extern "system" fn(GLenum) -> GLenum;
type ReadPixels = unsafe extern "system" fn(i32, i32, i32, i32, GLenum, GLenum, *mut c_void);

/// Desktop GL read-back commands.
struct Fns {
    get_tex_image: GetTexImage,
    get_compressed_tex_image: GetCompressedTexImage,
    get_integerv: GetIntegerv,
    get_tex_level_parameteriv: GetTexLevelParameteriv,
    bind_texture: BindTexture,
    gen_textures: GenTextures,
}

impl Fns {
    fn new(gl: &Gl) -> Result<Fns, String> {
        Ok(Fns {
            get_tex_image: gl.func("glGetTexImage")?,
            get_compressed_tex_image: gl.func("glGetCompressedTexImage")?,
            get_integerv: gl.func("glGetIntegerv")?,
            get_tex_level_parameteriv: gl.func("glGetTexLevelParameteriv")?,
            bind_texture: gl.func("glBindTexture")?,
            gen_textures: gl.func("glGenTextures")?,
        })
    }

    /// `glGetTexImage` of the bound texture.
    fn read(&self, target: GLenum, level: u32, format: GLenum, ty: GLenum, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        // SAFETY: `len` is the size of the level for `format` / `ty`.
        unsafe { (self.get_tex_image)(target, level as i32, format, ty, out.as_mut_ptr().cast()) };
        out
    }

    /// `glGetCompressedTexImage` of the bound texture.
    fn read_compressed(&self, target: GLenum, level: u32, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        // SAFETY: `len` is the compressed size of the level.
        unsafe { (self.get_compressed_tex_image)(target, level as i32, out.as_mut_ptr().cast()) };
        out
    }

    fn integer(&self, pname: GLenum) -> i32 {
        let mut v = 0;
        // SAFETY: one integer.
        unsafe { (self.get_integerv)(pname, &mut v) };
        v
    }
}

/// Loads the commands with `eglGetProcAddress`.
fn load(gl: &Gl) -> imgio::Result<GlContext> {
    let mut get = |name: &CStr| gl.proc_address(&name.to_string_lossy());
    // SAFETY: commands of the current context.
    unsafe { GlContext::load(&mut get) }
}

/// Uploads `src` and compares every level (all layers / faces) read back as `format` / `ty`,
/// or compressed when `format` is 0.
fn upload_and_compare(
    ctx: &mut GlContext,
    fns: &Fns,
    src: &dyn TextureSource,
    want_target: GLenum,
    (format, ty): (GLenum, GLenum),
) -> TestResult {
    // SAFETY: the context is current.
    let t = unsafe { ctx.upload(src, &GlUploadOptions::default()) }?;
    assert_eq!(t.target, want_target);
    assert!(!t.decoded);
    let want = levels(src)?;
    for (level, w) in want.iter().enumerate() {
        let level = level as u32;
        if t.target == GL_TEXTURE_CUBE_MAP {
            // One face target at a time.
            let n = w.len() / 6;
            for face in 0..6 {
                let target = GL_TEXTURE_CUBE_MAP_POSITIVE_X + face as GLenum;
                let got = if format == 0 {
                    fns.read_compressed(target, level, n)
                } else {
                    fns.read(target, level, format, ty, n)
                };
                assert!(got == w[face * n..][..n], "level {level} face {face}");
            }
        } else {
            let got = if format == 0 {
                fns.read_compressed(t.target, level, w.len())
            } else {
                fns.read(t.target, level, format, ty, w.len())
            };
            assert!(got == *w, "{:#x} level {level}", t.target);
        }
    }
    Ok(())
}

#[test]
fn gl_upload_desktop_core() -> TestResult {
    let Some(gl) = Gl::new(false) else {
        return Ok(());
    };
    let fns = Fns::new(&gl)?;
    let pixel_storei: PixelStorei = gl.func("glPixelStorei")?;
    // SAFETY: current context.
    unsafe { pixel_storei(GL_PACK_ALIGNMENT, 1) };
    let mut ctx = load(&gl)?;
    let alignment = fns.integer(GL_UNPACK_ALIGNMENT);
    let rgba = (GL_RGBA, GL_UNSIGNED_BYTE);

    // 2D with mips, cubemap, array, cube array, 3D.
    let f = Format::Rgba8Unorm;
    let cases = [
        ((16, 8, 1), 4, 1, false, GL_TEXTURE_2D),
        ((8, 8, 1), 2, 1, true, GL_TEXTURE_CUBE_MAP),
        ((8, 4, 1), 2, 3, false, GL_TEXTURE_2D_ARRAY),
        ((4, 4, 1), 3, 2, true, GL_TEXTURE_CUBE_MAP_ARRAY),
        ((4, 4, 4), 3, 1, false, GL_TEXTURE_3D),
    ];
    for (i, (dims, nlevels, layers, cube, target)) in cases.into_iter().enumerate() {
        let images = layers * if cube { 6 } else { 1 };
        let data = level_data(f, dims, nlevels, images, i as u32 * 10)?;
        let t = texture(f, dims, layers, cube, &data)?;
        upload_and_compare(&mut ctx, &fns, &t, target, rgba)?;
    }
    assert!(!ctx.is_es());
    // Uploads restore the unpack alignment.
    assert_eq!(fns.integer(GL_UNPACK_ALIGNMENT), alignment);

    // Files: compressed (BC7, BC1 array, BC3 cube) and uncompressed (RGBA16F cube, R8).
    for (name, target) in [
        ("bc7_mips", GL_TEXTURE_2D),
        ("bc1_array", GL_TEXTURE_2D_ARRAY),
    ] {
        let data = std::fs::read(format!("{DIR}ktx2/plain/{name}.ktx2"))?;
        let f = Ktx2File::parse(&data)?;
        upload_and_compare(&mut ctx, &fns, &f, target, (0, 0))?;
    }
    let data = std::fs::read(format!("{DIR}ktx2/plain/rgba16f_cube.ktx2"))?;
    let cube16 = Ktx2File::parse(&data)?;
    upload_and_compare(
        &mut ctx,
        &fns,
        &cube16,
        GL_TEXTURE_CUBE_MAP,
        (GL_RGBA, GL_HALF_FLOAT),
    )?;
    let data = std::fs::read(format!("{DIR}dds/bc3_cube_mips.dds"))?;
    let dds = DdsFile::parse(&data)?;
    upload_and_compare(&mut ctx, &fns, &dds, GL_TEXTURE_CUBE_MAP, (0, 0))?;
    let data = std::fs::read(format!("{DIR}dds/r8.dds"))?;
    let dds = DdsFile::parse(&data)?;
    upload_and_compare(
        &mut ctx,
        &fns,
        &dds,
        GL_TEXTURE_2D,
        (GL_RED, GL_UNSIGNED_BYTE),
    )?;

    // R8 5x3 with rows padded to 4: uploaded with unpack alignment 4, then restored.
    let tight = pattern(15, 5);
    let padded = pad_rows(&tight, 5);
    let mut t = TextureData::new(Format::R8Unorm, 5, 3).with_row_alignment(4);
    t.push_level(&padded)?;
    // SAFETY (all uploads): the context is current.
    let r = unsafe { ctx.upload(&t, &GlUploadOptions::default()) }?;
    assert_eq!(r.target, GL_TEXTURE_2D);
    assert_eq!(
        fns.read(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, 15),
        tight
    );
    assert_eq!(fns.integer(GL_UNPACK_ALIGNMENT), alignment);

    // ETC1 is not a desktop GL format: glCompressedTexImage2D fails with INVALID_ENUM and the
    // image is decoded in software (_ktxUnpackETC) and uploaded as RGBA8.
    let blocks = [pattern(32, 9)];
    let etc1 = texture(Format::Etc1Rgb, (8, 8, 1), 1, false, &blocks)?;
    let r = unsafe { ctx.upload(&etc1, &GlUploadOptions::default()) }?;
    assert!(r.decoded);
    let mut want = vec![0u8; 8 * 8 * 4];
    imgio::decode(Format::Etc1Rgb, &blocks[0], 8, 8, &mut want)?;
    assert_eq!(
        fns.read(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 256),
        want
    );
    // Without the fallback the GL error is returned.
    let never = GlUploadOptions {
        decode_fallback: GlDecodeFallback::Never,
        ..Default::default()
    };
    let r = unsafe { ctx.upload(&etc1, &never) };
    assert_eq!(r, Err(Error::Gpu(GL_INVALID_ENUM as i32)));

    // Compressed formats the context supports (ETC2 RGB8 is core in GL 4.3) are uploaded as is.
    let etc2 = texture(Format::Etc2Rgb, (8, 8, 1), 1, false, &blocks)?;
    let r = unsafe { ctx.upload(&etc2, &GlUploadOptions::default()) }?;
    assert!(!r.decoded);
    assert_eq!(fns.read_compressed(GL_TEXTURE_2D, 0, 32), blocks[0]);

    // PVRTC is rejected: error by default (ETC-only fallback), decoded with `Always`.
    let pvrtc = texture(Format::Pvrtc1Rgba, (8, 8, 1), 1, false, &blocks)?;
    let r = unsafe { ctx.upload(&pvrtc, &GlUploadOptions::default()) };
    assert_eq!(r, Err(Error::Gpu(GL_INVALID_ENUM as i32)));
    let always = GlUploadOptions {
        decode_fallback: GlDecodeFallback::Always,
        ..Default::default()
    };
    let r = unsafe { ctx.upload(&pvrtc, &always) }?;
    assert!(r.decoded);
    imgio::decode(Format::Pvrtc1Rgba, &blocks[0], 8, 8, &mut want)?;
    assert_eq!(
        fns.read(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 256),
        want
    );
    // An ATC array (no desktop GL support) is decoded layer by layer.
    let atc_data = [pattern(8 * 2, 4)];
    let atc = texture(Format::AtcRgb, (4, 4, 1), 2, false, &atc_data)?;
    let r = unsafe { ctx.upload(&atc, &always) }?;
    assert!(r.decoded);
    assert_eq!(r.target, GL_TEXTURE_2D_ARRAY);
    let mut want = vec![0u8; 2 * 64];
    for (i, w) in want.chunks_mut(64).enumerate() {
        imgio::decode(Format::AtcRgb, &atc_data[0][i * 8..][..8], 4, 4, w)?;
    }
    assert_eq!(
        fns.read(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, GL_UNSIGNED_BYTE, 128),
        want
    );

    // Generated mip levels (glGenerateMipmap).
    let base = [pattern(16 * 16 * 4, 3)];
    let t = texture(Format::Rgba8Unorm, (16, 16, 1), 1, false, &base)?;
    let generate = GlUploadOptions {
        generate_mipmaps: true,
        ..Default::default()
    };
    let r = unsafe { ctx.upload(&t, &generate) }?;
    let mut w = 0;
    unsafe { (fns.get_tex_level_parameteriv)(r.target, 4, GL_TEXTURE_WIDTH, &mut w) };
    assert_eq!(w, 1);

    // A texture name supplied by the application is used.
    let mut user = 0;
    unsafe { (fns.gen_textures)(1, &mut user) };
    let small = [pattern(64, 1)];
    let t = texture(Format::Rgba8Unorm, (4, 4, 1), 1, false, &small)?;
    let named = GlUploadOptions {
        texture: user,
        ..Default::default()
    };
    let r = unsafe { ctx.upload(&t, &named) }?;
    assert_eq!(r.name, user);
    unsafe { (fns.bind_texture)(GL_TEXTURE_2D, user) };
    assert_eq!(
        fns.read(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 64),
        small[0]
    );

    // Basis Universal payloads must be transcoded first.
    let data = std::fs::read(format!("{DIR}ktx2/etc1s_rgba.ktx2"))?;
    let basis = Ktx2File::parse(&data)?;
    let r = unsafe { ctx.upload(&basis, &GlUploadOptions::default()) };
    assert_eq!(r, Err(Error::Unsupported));
    Ok(())
}

/// GL ES 3.0 with the commands looked up in the process (`ktxLoadOpenGL(NULL)`): libGLESv2 is
/// loaded globally first, as when an application links it.
#[cfg(unix)]
#[test]
fn gl_upload_es3_process_symbols() -> TestResult {
    let Some(gl) = Gl::new(true) else {
        return Ok(());
    };
    // SAFETY: loading the system GL ES library.
    let _gles = unsafe {
        libloading::os::unix::Library::open(
            Some("libGLESv2.so.2"),
            libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
        )
    }?;
    // SAFETY: the ES context is current.
    let mut ctx = unsafe { GlContext::load_from_process() }?;

    let gen_framebuffers: GenFramebuffers = gl.func("glGenFramebuffers")?;
    let bind_framebuffer: BindFramebuffer = gl.func("glBindFramebuffer")?;
    let framebuffer_texture_2d: FramebufferTexture2D = gl.func("glFramebufferTexture2D")?;
    let check_status: CheckFramebufferStatus = gl.func("glCheckFramebufferStatus")?;
    let read_pixels: ReadPixels = gl.func("glReadPixels")?;
    let pixel_storei: PixelStorei = gl.func("glPixelStorei")?;
    let mut fbo = 0;
    // SAFETY (GL calls): the context is current.
    unsafe {
        gen_framebuffers(1, &mut fbo);
        bind_framebuffer(GL_FRAMEBUFFER, fbo);
        pixel_storei(GL_PACK_ALIGNMENT, 1);
    }
    let read = |tex: GLuint, target: GLenum, level: u32, w: u32, h: u32| -> Vec<u8> {
        let mut out = vec![0u8; (w * h * 4) as usize];
        unsafe {
            framebuffer_texture_2d(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                target,
                tex,
                level as i32,
            );
            assert_eq!(check_status(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
            read_pixels(
                0,
                0,
                w as i32,
                h as i32,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                out.as_mut_ptr().cast(),
            );
        }
        out
    };

    let f = Format::Rgba8Unorm;
    let data = level_data(f, (8, 4, 1), 3, 1, 1)?;
    let t = texture(f, (8, 4, 1), 1, false, &data)?;
    let r = unsafe { ctx.upload(&t, &GlUploadOptions::default()) }?;
    assert_eq!(r.target, GL_TEXTURE_2D);
    assert!(ctx.is_es());
    for (level, want) in data.iter().enumerate() {
        let (w, h) = ((8u32 >> level).max(1), (4u32 >> level).max(1));
        assert!(
            read(r.name, GL_TEXTURE_2D, level as u32, w, h) == *want,
            "level {level}"
        );
    }

    let data = level_data(f, (4, 4, 1), 1, 6, 2)?;
    let cube = texture(f, (4, 4, 1), 1, true, &data)?;
    let r = unsafe { ctx.upload(&cube, &GlUploadOptions::default()) }?;
    assert_eq!(r.target, GL_TEXTURE_CUBE_MAP);
    for face in 0..6 {
        let want = &data[0][face as usize * 64..][..64];
        assert!(
            read(r.name, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, 4, 4) == want,
            "face {face}"
        );
    }

    // ETC2 is core in ES 3.0: uploaded compressed (compressed textures cannot be attached to
    // a framebuffer, so only the upload is checked).
    let blocks = [pattern(32, 9)];
    let etc2 = texture(Format::Etc2Rgb, (8, 8, 1), 1, false, &blocks)?;
    let r = unsafe { ctx.upload(&etc2, &GlUploadOptions::default()) }?;
    assert!(!r.decoded);
    // ATC is not: decoded with `Always` and readable.
    let atc = texture(Format::AtcRgb, (8, 8, 1), 1, false, &blocks)?;
    let always = GlUploadOptions {
        decode_fallback: GlDecodeFallback::Always,
        ..Default::default()
    };
    let r = unsafe { ctx.upload(&atc, &always) }?;
    assert!(r.decoded);
    let mut want = vec![0u8; 256];
    imgio::decode(Format::AtcRgb, &blocks[0], 8, 8, &mut want)?;
    assert_eq!(read(r.name, GL_TEXTURE_2D, 0, 8, 8), want);
    Ok(())
}

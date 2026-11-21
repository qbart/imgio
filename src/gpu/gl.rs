// SPDX-FileCopyrightText: 2010-2026 Mark Callow, The Khronos Group Inc. and KTX-Software contributors
// SPDX-License-Identifier: Apache-2.0
// Translated to Rust from KTX-Software (via ktx-rs `src/glloader.rs`, `src/glupload.rs`) and
// modified; see NOTICE.md.
//! OpenGL {,ES} texture upload (feature `opengl`): port of KTX-Software `lib/src/glloader.c`
//! (`ktxTexture_GLUpload`, `discoverContextCapabilities`) and `gl_funcs.c` for
//! [`TextureSource`]s.
//!
//! This is the GL loader of AGENTS.md rule 9, one of the two modules that may use `unsafe`:
//! every GL command is called through a function pointer of the application's GL library, on
//! the context current on the calling thread. Loading and uploading are therefore `unsafe fn`.
//!
//! imgio never loads GL itself: [`GlContext::load`] takes the application's `GetProcAddress`
//! (e.g. `eglGetProcAddress`), [`GlContext::load_from_process`] searches the running process
//! (`dlopen(NULL)`; on Windows an already loaded `libGLESv2.dll`, `libGLES_CM.dll` or
//! `opengl32.dll` + `wglGetProcAddress`). The loaded commands and the context capabilities
//! (discovered at the first upload) live in the [`GlContext`], not in process globals as in C.
//!
//! Targets: 2D, 2D array, cubemap (one image per face target), cubemap array and 3D; 1D
//! textures are not supported (imgio sources are 2D or 3D). Deviations from C: a missing
//! optional command or an unsupported target returns `Unsupported` before anything is created;
//! cubemap arrays are uploaded with `layers * 6` layer-faces; the software fallback for formats
//! the context rejects uses imgio's decoders and can be extended to every decodable format
//! ([`GlDecodeFallback::Always`]); contexts without sized formats get the unsized base format
//! for every uncompressed format.
#![allow(unsafe_code)]

use std::ffi::{CStr, c_char, c_void};

use super::gl_format::{GlFormat, gl_format, is_etc};
use super::texture::{Shape, TextureSource};
use crate::{Buffer, Error, Format, Result};

pub type GLenum = u32;
pub type GLuint = u32;
pub type GLint = i32;
pub type GLsizei = i32;

pub const GL_NO_ERROR: GLenum = 0;
pub const GL_INVALID_ENUM: GLenum = 0x0500;
pub const GL_INVALID_VALUE: GLenum = 0x0501;
pub const GL_TEXTURE_2D: GLenum = 0x0DE1;
pub const GL_TEXTURE_3D: GLenum = 0x806F;
pub const GL_TEXTURE_2D_ARRAY: GLenum = 0x8C1A;
pub const GL_TEXTURE_CUBE_MAP: GLenum = 0x8513;
pub const GL_TEXTURE_CUBE_MAP_POSITIVE_X: GLenum = 0x8515;
pub const GL_TEXTURE_CUBE_MAP_ARRAY: GLenum = 0x9009;
const GL_TEXTURE_MAX_LEVEL: GLenum = 0x813D;
const GL_GENERATE_MIPMAP: GLenum = 0x8191;
const GL_UNPACK_ALIGNMENT: GLenum = 0x0CF5;
const GL_VERSION: GLenum = 0x1F02;
const GL_EXTENSIONS: GLenum = 0x1F03;
const GL_NUM_EXTENSIONS: GLenum = 0x821D;
const GL_MAJOR_VERSION: GLenum = 0x821B;
const GL_MINOR_VERSION: GLenum = 0x821C;
const GL_CONTEXT_PROFILE_MASK: GLenum = 0x9126;
const GL_CONTEXT_CORE_PROFILE_BIT: GLint = 0x1;
const GL_CONTEXT_COMPATIBILITY_PROFILE_BIT: GLint = 0x2;
const CONTEXT_ES_PROFILE_BIT: GLint = 0x4;
const GL_TRUE: GLint = 1;

/// Non-legacy sized formats are supported.
const NON_LEGACY_FORMATS: GLint = 0x1;
/// Legacy sized formats are supported.
const LEGACY_FORMATS: GLint = 0x2;
const ALL_SIZED_FORMATS: GLint = NON_LEGACY_FORMATS | LEGACY_FORMATS;
const NO_SIZED_FORMATS: GLint = 0;
/// `_KTX_R16_FORMATS_NORM` / `_SNORM`.
const R16_FORMATS_NORM: u32 = 0x1;
const R16_FORMATS_SNORM: u32 = 0x2;

type PfnBindTexture = unsafe extern "system" fn(GLenum, GLuint);
type PfnCompressedTexImage2D = unsafe extern "system" fn(
    GLenum,
    GLint,
    GLenum,
    GLsizei,
    GLsizei,
    GLint,
    GLsizei,
    *const c_void,
);
type PfnCompressedTexImage3D = unsafe extern "system" fn(
    GLenum,
    GLint,
    GLenum,
    GLsizei,
    GLsizei,
    GLsizei,
    GLint,
    GLsizei,
    *const c_void,
);
type PfnDeleteTextures = unsafe extern "system" fn(GLsizei, *const GLuint);
type PfnGenerateMipmap = unsafe extern "system" fn(GLenum);
type PfnGenTextures = unsafe extern "system" fn(GLsizei, *mut GLuint);
type PfnGetError = unsafe extern "system" fn() -> GLenum;
type PfnGetIntegerv = unsafe extern "system" fn(GLenum, *mut GLint);
type PfnGetString = unsafe extern "system" fn(GLenum) -> *const u8;
type PfnGetStringi = unsafe extern "system" fn(GLenum, GLuint) -> *const u8;
type PfnPixelStorei = unsafe extern "system" fn(GLenum, GLint);
type PfnTexImage2D = unsafe extern "system" fn(
    GLenum,
    GLint,
    GLint,
    GLsizei,
    GLsizei,
    GLint,
    GLenum,
    GLenum,
    *const c_void,
);
type PfnTexImage3D = unsafe extern "system" fn(
    GLenum,
    GLint,
    GLint,
    GLsizei,
    GLsizei,
    GLsizei,
    GLint,
    GLenum,
    GLenum,
    *const c_void,
);
type PfnTexParameteri = unsafe extern "system" fn(GLenum, GLenum, GLint);

/// The GL commands used by the upload (`gl_funclist.inl`, without the unused ones).
#[derive(Clone, Copy)]
struct Commands {
    bind_texture: PfnBindTexture,
    compressed_tex_image_2d: PfnCompressedTexImage2D,
    delete_textures: PfnDeleteTextures,
    gen_textures: PfnGenTextures,
    get_error: PfnGetError,
    get_integerv: PfnGetIntegerv,
    get_string: PfnGetString,
    pixel_storei: PfnPixelStorei,
    tex_image_2d: PfnTexImage2D,
    tex_parameteri: PfnTexParameteri,
    get_stringi: Option<PfnGetStringi>,
    generate_mipmap: Option<PfnGenerateMipmap>,
    tex_image_3d: Option<PfnTexImage3D>,
    compressed_tex_image_3d: Option<PfnCompressedTexImage3D>,
}

/// The `discoverContextCapabilities` statics of `glloader.c`.
#[derive(Debug, Clone, Copy)]
struct Caps {
    discovered: bool,
    context_profile: GLint,
    sized_formats: GLint,
    r16_formats: u32,
    supports_srgb: bool,
    supports_cube_map_arrays: bool,
    supports_max_level: bool,
}

impl Default for Caps {
    fn default() -> Self {
        Caps {
            discovered: false,
            context_profile: 0,
            sized_formats: ALL_SIZED_FORMATS,
            r16_formats: R16_FORMATS_NORM | R16_FORMATS_SNORM,
            supports_srgb: true,
            supports_cube_map_arrays: false,
            supports_max_level: false,
        }
    }
}

/// When to decode a compressed format the context rejects (`GL_INVALID_ENUM` /
/// `GL_INVALID_VALUE`) and upload the decoded pixels instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GlDecodeFallback {
    /// Never: the GL error is returned.
    Never,
    /// ETC1, ETC2 and EAC, as KTX-Software (`_ktxUnpackETC`).
    #[default]
    Etc,
    /// Every format imgio can decode ([`crate::can_decode`]).
    Always,
}

/// Options of [`GlContext::upload`].
#[derive(Debug, Clone, Copy, Default)]
pub struct GlUploadOptions {
    /// Texture name to upload to; 0 generates one (deleted again if the upload fails).
    pub texture: GLuint,
    /// Upload level 0 and generate the mip chain (`glGenerateMipmap`, or
    /// `GL_GENERATE_MIPMAP` on contexts without it).
    pub generate_mipmaps: bool,
    pub decode_fallback: GlDecodeFallback,
}

/// A GL texture created by [`GlContext::upload`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlTexture {
    /// Texture name.
    pub name: GLuint,
    /// `GL_TEXTURE_2D`, `GL_TEXTURE_2D_ARRAY`, `GL_TEXTURE_CUBE_MAP`,
    /// `GL_TEXTURE_CUBE_MAP_ARRAY` or `GL_TEXTURE_3D`.
    pub target: GLenum,
    /// True when the images were decoded in software ([`GlDecodeFallback`]).
    pub decoded: bool,
}

/// `PFNGLGETPROCADDRESS`: the address of a GL command, null if unavailable.
pub type GetProcAddress<'a> = &'a mut dyn FnMut(&CStr) -> *const c_void;

type WglGetProcAddress = unsafe extern "system" fn(*const c_char) -> *const c_void;

/// The process (or GL dll) searched by [`GlContext::load_from_process`].
#[cfg(unix)]
fn gl_module() -> Option<(libloading::Library, Option<WglGetProcAddress>)> {
    // dlopen(NULL): the application created the context, so GL is in the process.
    Some((libloading::os::unix::Library::this().into(), None))
}

/// `ktxFindOpenGL`: an OpenGL {,ES} dll the process has already loaded.
#[cfg(windows)]
fn gl_module() -> Option<(libloading::Library, Option<WglGetProcAddress>)> {
    use libloading::os::windows::Library;
    // Emulators probably also have opengl32.dll loaded so check that last.
    for name in ["libGLESv2.dll", "libGLES_CM.dll"] {
        if let Ok(m) = Library::open_already_loaded(name) {
            return Some((m.into(), None));
        }
    }
    let module: libloading::Library = Library::open_already_loaded("opengl32.dll").ok()?.into();
    // wglGetProcAddress for the commands after OpenGL 1.1.
    // SAFETY: wglGetProcAddress has this signature.
    let wgl = unsafe { module.get::<WglGetProcAddress>(b"wglGetProcAddress\0") }
        .ok()
        .map(|s| *s)?;
    Some((module, Some(wgl)))
}

#[cfg(not(any(unix, windows)))]
fn gl_module() -> Option<(libloading::Library, Option<WglGetProcAddress>)> {
    None
}

/// The GL commands of a context and its capabilities.
pub struct GlContext {
    gl: Commands,
    caps: Caps,
}

impl GlContext {
    /// Port of `ktxLoadOpenGL`: loads the GL commands through `get_proc_address`.
    ///
    /// Errors: `Unsupported` when a command present in every GL {,ES} version is missing.
    ///
    /// # Safety
    /// The returned addresses must be the named GL commands (or null) of the context the
    /// application makes current when uploading.
    pub unsafe fn load(get_proc_address: GetProcAddress<'_>) -> Result<GlContext> {
        macro_rules! cmd {
            ($t:ty, $name:literal) => {{
                let p = get_proc_address($name);
                if p.is_null() {
                    None
                } else {
                    // SAFETY: the address of the command named like `$t` (caller contract).
                    Some(unsafe { std::mem::transmute::<*const c_void, $t>(p) })
                }
            }};
        }
        macro_rules! required {
            ($t:ty, $name:literal) => {
                cmd!($t, $name).ok_or(Error::Unsupported)?
            };
        }
        Ok(GlContext {
            gl: Commands {
                bind_texture: required!(PfnBindTexture, c"glBindTexture"),
                compressed_tex_image_2d: required!(
                    PfnCompressedTexImage2D,
                    c"glCompressedTexImage2D"
                ),
                delete_textures: required!(PfnDeleteTextures, c"glDeleteTextures"),
                gen_textures: required!(PfnGenTextures, c"glGenTextures"),
                get_error: required!(PfnGetError, c"glGetError"),
                get_integerv: required!(PfnGetIntegerv, c"glGetIntegerv"),
                get_string: required!(PfnGetString, c"glGetString"),
                pixel_storei: required!(PfnPixelStorei, c"glPixelStorei"),
                tex_image_2d: required!(PfnTexImage2D, c"glTexImage2D"),
                tex_parameteri: required!(PfnTexParameteri, c"glTexParameteri"),
                get_stringi: cmd!(PfnGetStringi, c"glGetStringi"),
                generate_mipmap: cmd!(PfnGenerateMipmap, c"glGenerateMipmap"),
                tex_image_3d: cmd!(PfnTexImage3D, c"glTexImage3D"),
                compressed_tex_image_3d: cmd!(PfnCompressedTexImage3D, c"glCompressedTexImage3D"),
            },
            caps: Caps::default(),
        })
    }

    /// Port of `ktxLoadOpenGL(NULL)`: loads the GL commands from the GL library the process
    /// has already loaded (`Unsupported` when there is none).
    ///
    /// # Safety
    /// As [`GlContext::load`]: the process's GL commands must be those of the context the
    /// application makes current when uploading.
    pub unsafe fn load_from_process() -> Result<GlContext> {
        let (module, wgl) = gl_module().ok_or(Error::Unsupported)?;
        let mut get = |name: &CStr| -> *const c_void {
            if let Some(wgl) = wgl {
                // SAFETY: wglGetProcAddress on a NUL terminated name.
                let p = unsafe { wgl(name.as_ptr()) };
                if !p.is_null() {
                    return p;
                }
            }
            // SAFETY: only the address is read.
            unsafe { module.get::<*const c_void>(name.to_bytes_with_nul()) }
                .ok()
                .map_or(std::ptr::null(), |s| *s)
        };
        // SAFETY: forwarded contract. The library stays loaded: it is the process itself or a
        // dll the application loaded.
        unsafe { GlContext::load(&mut get) }
    }

    /// True when the context is GL ES (known after the first upload).
    pub fn is_es(&self) -> bool {
        self.caps.context_profile & CONTEXT_ES_PROFILE_BIT != 0
    }

    /// Port of `ktxTexture_GLUpload`: creates a GL texture holding the images of `src` and
    /// returns its name and target. The context's capabilities are discovered at the first
    /// upload. `GL_UNPACK_ALIGNMENT` is set to the source's row alignment and restored.
    ///
    /// Errors: `Gpu(gl_error)` (the texture name is deleted if it was generated),
    /// `InvalidArgument` (bad source shape), `Unsupported` (no GL format, target not supported
    /// by the context), and the source's errors.
    ///
    /// # Safety
    /// The context the commands were loaded for must be current on this thread.
    pub unsafe fn upload(
        &mut self,
        src: &(impl TextureSource + ?Sized),
        options: &GlUploadOptions,
    ) -> Result<GlTexture> {
        let shape = Shape::of(src)?;
        let glf = gl_format(shape.format).ok_or(Error::Unsupported)?;
        if !self.caps.discovered {
            self.discover();
        }
        let gl = &self.gl;
        let caps = &self.caps;
        let compressed = glf.is_compressed();
        let cube = shape.faces == 6;
        let (target, three_d) = if shape.depth > 1 {
            (GL_TEXTURE_3D, true)
        } else if shape.array {
            if cube {
                (GL_TEXTURE_CUBE_MAP_ARRAY, true)
            } else {
                (GL_TEXTURE_2D_ARRAY, true)
            }
        } else if cube {
            (GL_TEXTURE_CUBE_MAP, false)
        } else {
            (GL_TEXTURE_2D, false)
        };
        if three_d
            && ((compressed && gl.compressed_tex_image_3d.is_none())
                || (!compressed && gl.tex_image_3d.is_none()))
        {
            return Err(Error::Unsupported);
        }
        if target == GL_TEXTURE_CUBE_MAP_ARRAY && !caps.supports_cube_map_arrays {
            return Err(Error::Unsupported);
        }

        let mut internal_format = glf.internal_format;
        if !compressed && caps.sized_formats == NO_SIZED_FORMATS {
            internal_format = glf.base_internal_format;
        }
        let up = Upload {
            gl,
            caps,
            shape: &shape,
            glf,
            internal_format,
            fallback: options.decode_fallback,
        };

        let alignment = shape.row_alignment as GLint;
        // SAFETY (whole upload): the commands belong to the current context (caller
        // contract); every pointer passed points to a live local or to image data of at least
        // the size GL reads (checked by `Upload`).
        unsafe {
            let mut previous = 0;
            (gl.get_integerv)(GL_UNPACK_ALIGNMENT, &mut previous);
            if previous != alignment {
                (gl.pixel_storei)(GL_UNPACK_ALIGNMENT, alignment);
            }
            let name = if options.texture != 0 {
                options.texture
            } else {
                let mut t = 0;
                (gl.gen_textures)(1, &mut t);
                t
            };
            (gl.bind_texture)(target, name);
            let generate = options.generate_mipmaps;
            if generate && gl.generate_mipmap.is_none() {
                (gl.tex_parameteri)(target, GL_GENERATE_MIPMAP, GL_TRUE);
            }
            if !generate && caps.supports_max_level {
                (gl.tex_parameteri)(target, GL_TEXTURE_MAX_LEVEL, shape.levels as GLint - 1);
            }
            let levels = if generate { 1 } else { shape.levels };
            let mut decoded = false;
            let mut result = Ok(());
            for level in 0..levels {
                let r = if three_d {
                    up.level_3d(src, target, level)
                } else {
                    up.level_2d(src, target, level, cube)
                };
                match r {
                    Ok(d) => decoded |= d,
                    Err(e) => {
                        result = Err(e);
                        break;
                    }
                }
            }
            if result.is_ok()
                && generate
                && let Some(generate_mipmap) = gl.generate_mipmap
            {
                generate_mipmap(target);
            }
            if result.is_err() && options.texture == 0 {
                (gl.delete_textures)(1, &name);
            }
            if previous != alignment {
                (gl.pixel_storei)(GL_UNPACK_ALIGNMENT, previous);
            }
            result.map(|()| GlTexture {
                name,
                target,
                decoded,
            })
        }
    }

    fn get_string(&self, name: GLenum) -> String {
        // SAFETY: GL returns a static NUL terminated string or null.
        unsafe {
            let p = (self.gl.get_string)(name);
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p.cast::<c_char>())
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }

    fn get_integer(&self, pname: GLenum, value: &mut GLint) {
        // SAFETY: one integer is written.
        unsafe { (self.gl.get_integerv)(pname, value) };
    }

    fn get_error(&self) -> GLenum {
        // SAFETY: no arguments.
        unsafe { (self.gl.get_error)() }
    }

    /// `hasExtension`.
    fn has_extension(&self, extension: &str) -> bool {
        match self.gl.get_stringi {
            None => self
                .get_string(GL_EXTENSIONS)
                .split_ascii_whitespace()
                .any(|e| e == extension),
            Some(get_stringi) => {
                let mut n = 0;
                self.get_integer(GL_NUM_EXTENSIONS, &mut n);
                (0..n.max(0) as GLuint).any(|i| {
                    // SAFETY: index below GL_NUM_EXTENSIONS; static string or null.
                    let p = unsafe { get_stringi(GL_EXTENSIONS, i) };
                    // SAFETY: NUL terminated when not null.
                    !p.is_null()
                        && unsafe { CStr::from_ptr(p.cast::<c_char>()) }.to_bytes()
                            == extension.as_bytes()
                })
            }
        }
    }

    /// Port of `discoverContextCapabilities` (the context must be current).
    fn discover(&mut self) {
        let mut caps = Caps {
            discovered: true,
            ..Caps::default()
        };
        let mut major: GLint = 1;
        let mut minor: GLint = 0;
        let version = self.get_string(GL_VERSION);
        if version.contains("GL ES") {
            caps.context_profile = CONTEXT_ES_PROFILE_BIT;
        }
        // MAJOR & MINOR only introduced in GL {,ES} 3.0.
        self.get_integer(GL_MAJOR_VERSION, &mut major);
        self.get_integer(GL_MINOR_VERSION, &mut minor);
        if self.get_error() != GL_NO_ERROR {
            // < v3.0; resort to the old-fashioned way.
            let prefix = if caps.context_profile & CONTEXT_ES_PROFILE_BIT != 0 {
                "OpenGL ES "
            } else {
                "OpenGL "
            };
            scan_version(&version, prefix, &mut major, &mut minor);
        }
        if caps.context_profile & CONTEXT_ES_PROFILE_BIT != 0 {
            if major < 3 {
                caps.sized_formats = NO_SIZED_FORMATS;
                caps.r16_formats = 0;
                caps.supports_srgb = false;
                // The same driver may export ES 3 commands to an ES 2 context.
                self.gl.get_stringi = None;
                if !self.has_extension("GL_OES_texture_3D") {
                    self.gl.tex_image_3d = None;
                    self.gl.compressed_tex_image_3d = None;
                }
                if major < 2 {
                    self.gl.generate_mipmap = None;
                }
            } else {
                caps.sized_formats = NON_LEGACY_FORMATS;
                caps.supports_cube_map_arrays = self.has_extension("GL_EXT_texture_cube_map_array");
                caps.supports_max_level = true;
            }
            if self.has_extension("GL_OES_required_internalformat") {
                caps.sized_formats |= ALL_SIZED_FORMATS;
            }
            // There are no OES extensions for sRGB textures or R16 formats.
        } else {
            // PROFILE_MASK was introduced in OpenGL 3.2.
            let mut profile = caps.context_profile;
            self.get_integer(GL_CONTEXT_PROFILE_MASK, &mut profile);
            caps.context_profile = profile;
            if self.get_error() == GL_NO_ERROR {
                // >= 3.2
                if caps.context_profile & GL_CONTEXT_CORE_PROFILE_BIT != 0 {
                    caps.sized_formats &= !LEGACY_FORMATS;
                }
                caps.supports_cube_map_arrays = major >= 4;
                caps.supports_max_level = true;
            } else {
                // < 3.2
                caps.context_profile = GL_CONTEXT_COMPATIBILITY_PROFILE_BIT;
                // sRGB textures introduced in 2.0.
                if major < 2 && !self.has_extension("GL_EXT_texture_sRGB") {
                    caps.supports_srgb = false;
                }
                // R{,G}16 introduced in 3.0; R{,G}16_SNORM introduced in 3.1.
                if major == 3 {
                    if minor == 0 {
                        caps.r16_formats &= !R16_FORMATS_SNORM;
                    }
                    caps.supports_max_level =
                        minor >= 1 || self.has_extension("GL_ARB_texture_query_levels");
                } else if self.has_extension("GL_ARB_texture_rg") {
                    caps.r16_formats &= !R16_FORMATS_SNORM;
                } else {
                    caps.r16_formats = 0;
                }
            }
            if !caps.supports_cube_map_arrays && self.has_extension("GL_ARB_texture_cube_map_array")
            {
                caps.supports_cube_map_arrays = true;
            }
        }
        self.caps = caps;
    }
}

/// `sscanf(version, "<prefix>%d.%d", &major, &minor)`: values are only set when matched.
fn scan_version(version: &str, prefix: &str, major: &mut GLint, minor: &mut GLint) {
    let Some(rest) = version.strip_prefix(prefix) else {
        return;
    };
    let int = |s: &str| -> Option<(GLint, usize)> {
        let n = s.bytes().take_while(u8::is_ascii_digit).count();
        s.get(..n)?.parse().ok().map(|v| (v, n))
    };
    let Some((maj, n)) = int(rest) else {
        return;
    };
    *major = maj;
    if let Some(after) = rest.get(n..).and_then(|r| r.strip_prefix('.'))
        && let Some((min, _)) = int(after)
    {
        *minor = min;
    }
}

/// One upload: commands, capabilities and formats.
struct Upload<'a> {
    gl: &'a Commands,
    caps: &'a Caps,
    shape: &'a Shape,
    glf: GlFormat,
    internal_format: u32,
    fallback: GlDecodeFallback,
}

fn gl_size(n: usize) -> Result<GLsizei> {
    GLsizei::try_from(n).map_err(|_| Error::InvalidArgument)
}

fn gl_result(error: GLenum) -> Result<()> {
    if error == GL_NO_ERROR {
        Ok(())
    } else {
        Err(Error::Gpu(error as i32))
    }
}

impl Upload<'_> {
    /// `glTexImage2D` / `glCompressedTexImage2D` of every face of `level`; true when decoded.
    ///
    /// # Safety
    /// The context of the commands is current.
    unsafe fn level_2d(
        &self,
        src: &(impl TextureSource + ?Sized),
        target: GLenum,
        level: u32,
        cube: bool,
    ) -> Result<bool> {
        let size = self.shape.image_bytes(level)?;
        let mut decoded = false;
        for face in 0..self.shape.faces {
            let data = src.image_data(level, 0, face)?;
            let data = data.get(..size).ok_or(Error::InvalidData)?;
            let face_target = if cube {
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + face
            } else {
                target
            };
            // SAFETY: forwarded contract.
            decoded |= unsafe { self.image(face_target, level, 1, false, data) }?;
        }
        Ok(decoded)
    }

    /// `glTexImage3D` / `glCompressedTexImage3D` of `level`: every layer-face of an array, or
    /// every slice of a 3D texture; true when decoded.
    ///
    /// # Safety
    /// The context of the commands is current.
    unsafe fn level_3d(
        &self,
        src: &(impl TextureSource + ?Sized),
        target: GLenum,
        level: u32,
    ) -> Result<bool> {
        let s = self.shape;
        let size = s.image_bytes(level)?;
        let images = s.images();
        let joined;
        let data = if images == 1 {
            src.image_data(level, 0, 0)?
                .get(..size)
                .ok_or(Error::InvalidData)?
        } else {
            let mut all = Buffer::default();
            all.reserve(
                size.checked_mul(images as usize)
                    .ok_or(Error::InvalidArgument)?,
            )
            .map_err(|_| Error::OutOfMemory)?;
            for layer in 0..s.layers {
                for face in 0..s.faces {
                    let image = src.image_data(level, layer, face)?;
                    all.extend_from_slice(image.get(..size).ok_or(Error::InvalidData)?)
                        .map_err(|_| Error::OutOfMemory)?;
                }
            }
            joined = all;
            &joined[..]
        };
        let slices = if s.depth > 1 {
            s.extent(level).2
        } else {
            images
        };
        // SAFETY: forwarded contract.
        unsafe { self.image(target, level, slices, true, data) }
    }

    /// Uploads `slices` images of `level` (all in `data`) with the 2D or 3D command, decoding
    /// them in software if the context rejects the compressed format.
    ///
    /// # Safety
    /// The context of the commands is current.
    unsafe fn image(
        &self,
        target: GLenum,
        level: u32,
        slices: u32,
        three_d: bool,
        data: &[u8],
    ) -> Result<bool> {
        let gl = self.gl;
        let (w, h, _) = self.shape.extent(level);
        let (w, h, d) = (
            gl_size(w as usize)?,
            gl_size(h as usize)?,
            gl_size(slices as usize)?,
        );
        let lvl = level as GLint;
        // SAFETY (all calls): forwarded contract; `data` holds every byte GL reads: the
        // caller checked its size against `Shape::image_bytes` at the current unpack
        // alignment.
        let error = unsafe {
            if self.glf.is_compressed() {
                let size = gl_size(data.len())?;
                let ptr = data.as_ptr().cast::<c_void>();
                match (three_d, gl.compressed_tex_image_3d) {
                    (false, _) => (gl.compressed_tex_image_2d)(
                        target,
                        lvl,
                        self.internal_format,
                        w,
                        h,
                        0,
                        size,
                        ptr,
                    ),
                    (true, Some(f)) => f(target, lvl, self.internal_format, w, h, d, 0, size, ptr),
                    (true, None) => return Err(Error::Unsupported),
                }
            } else {
                let ptr = data.as_ptr().cast::<c_void>();
                let internal = self.internal_format as GLint;
                let (format, ty) = (self.glf.format, self.glf.ty);
                match (three_d, gl.tex_image_3d) {
                    (false, _) => {
                        (gl.tex_image_2d)(target, lvl, internal, w, h, 0, format, ty, ptr)
                    }
                    (true, Some(f)) => f(target, lvl, internal, w, h, d, 0, format, ty, ptr),
                    (true, None) => return Err(Error::Unsupported),
                }
            }
            (gl.get_error)()
        };
        let rejected = error == GL_INVALID_ENUM || error == GL_INVALID_VALUE;
        let decode = self.glf.is_compressed()
            && rejected
            && match self.fallback {
                GlDecodeFallback::Never => false,
                GlDecodeFallback::Etc => is_etc(self.glf.internal_format),
                GlDecodeFallback::Always => crate::can_decode(self.shape.format),
            };
        if !decode {
            return gl_result(error).map(|()| false);
        }
        // SAFETY: forwarded contract.
        unsafe { self.decoded(target, level, slices, three_d, data) }.map(|()| true)
    }

    /// The software fallback (`_ktxUnpackETC` generalized): decodes `slices` images and
    /// uploads them as uncompressed pixels, 8 bit when the context lacks R16 formats and
    /// linear when it lacks sRGB.
    ///
    /// # Safety
    /// The context of the commands is current.
    unsafe fn decoded(
        &self,
        target: GLenum,
        level: u32,
        slices: u32,
        three_d: bool,
        data: &[u8],
    ) -> Result<()> {
        let (w, h, _) = self.shape.extent(level);
        let format = self.shape.format;
        let mut out_format = format.decoded();
        let src_size = format.surface_bytes(w, h)?;
        let dst_size = out_format.surface_bytes(w, h)?;
        let mut pixels = crate::buffer::zeroed(
            dst_size
                .checked_mul(slices as usize)
                .ok_or(Error::InvalidArgument)?,
        )?;
        for (i, dst) in pixels.chunks_exact_mut(dst_size.max(1)).enumerate() {
            let at = i * src_size;
            let src = data.get(at..at + src_size).ok_or(Error::InvalidData)?;
            crate::decode(format, src, w, h, dst)?;
        }
        let narrow = match out_format {
            Format::R16Unorm | Format::Rg16Unorm => self.caps.r16_formats & R16_FORMATS_NORM == 0,
            Format::R16Snorm | Format::Rg16Snorm => self.caps.r16_formats & R16_FORMATS_SNORM == 0,
            _ => false,
        };
        if narrow {
            // Keep the high byte of each 16-bit little-endian value.
            let n = pixels.len() / 2;
            for i in 0..n {
                if let Some(&hi) = pixels.get(2 * i + 1)
                    && let Some(lo) = pixels.get_mut(i)
                {
                    *lo = hi;
                }
            }
            pixels.truncate(n);
            out_format = match out_format {
                Format::R16Unorm => Format::R8Unorm,
                Format::R16Snorm => Format::R8Snorm,
                Format::Rg16Unorm => Format::Rg8Unorm,
                _ => Format::Rg8Snorm,
            };
        }
        if out_format == Format::Rgba8Srgb && !self.caps.supports_srgb {
            out_format = Format::Rgba8Unorm;
        }
        let glf = gl_format(out_format).ok_or(Error::Unsupported)?;
        let internal = if self.caps.sized_formats & NON_LEGACY_FORMATS == 0 {
            glf.base_internal_format
        } else {
            glf.internal_format
        };
        let gl = self.gl;
        let lvl = level as GLint;
        let (w, h, d) = (
            gl_size(w as usize)?,
            gl_size(h as usize)?,
            gl_size(slices as usize)?,
        );
        let ptr = pixels.as_ptr().cast::<c_void>();
        // SAFETY: forwarded contract; `pixels` holds tightly packed rows, uploaded with
        // GL_UNPACK_ALIGNMENT 1 (restored afterwards).
        let error = unsafe {
            let mut previous = 0;
            (gl.get_integerv)(GL_UNPACK_ALIGNMENT, &mut previous);
            (gl.pixel_storei)(GL_UNPACK_ALIGNMENT, 1);
            match (three_d, gl.tex_image_3d) {
                (false, _) => (gl.tex_image_2d)(
                    target,
                    lvl,
                    internal as GLint,
                    w,
                    h,
                    0,
                    glf.format,
                    glf.ty,
                    ptr,
                ),
                (true, Some(f)) => f(
                    target,
                    lvl,
                    internal as GLint,
                    w,
                    h,
                    d,
                    0,
                    glf.format,
                    glf.ty,
                    ptr,
                ),
                (true, None) => {}
            }
            let error = (gl.get_error)();
            (gl.pixel_storei)(GL_UNPACK_ALIGNMENT, previous);
            error
        };
        if three_d && gl.tex_image_3d.is_none() {
            return Err(Error::Unsupported);
        }
        gl_result(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_scan() {
        let (mut major, mut minor) = (1, 0);
        scan_version(
            "OpenGL ES 2.0 Mesa 24.0",
            "OpenGL ES ",
            &mut major,
            &mut minor,
        );
        assert_eq!((major, minor), (2, 0));
        scan_version("OpenGL 1.5", "OpenGL ES ", &mut major, &mut minor);
        assert_eq!((major, minor), (2, 0));
        scan_version("OpenGL 1.5", "OpenGL ", &mut major, &mut minor);
        assert_eq!((major, minor), (1, 5));
    }
}

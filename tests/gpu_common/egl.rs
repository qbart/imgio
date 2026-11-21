// SPDX-FileCopyrightText: 2010-2026 Mark Callow, The Khronos Group Inc. and KTX-Software contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from ktx-rs `tests/gl_common/mod.rs`; see NOTICE.md.
//! Headless EGL (Mesa surfaceless platform) contexts for the GL upload tests.
#![allow(unsafe_code)]

use khronos_egl as egl;
use std::ffi::c_void;
use std::sync::{Mutex, MutexGuard};

const PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31DD;

/// EGL displays are shared per process: one context at a time.
static LOCK: Mutex<()> = Mutex::new(());

pub struct Gl {
    pub egl: egl::DynamicInstance<egl::EGL1_5>,
    display: egl::Display,
    context: egl::Context,
    _lock: MutexGuard<'static, ()>,
}

impl Gl {
    /// A current context, desktop core 3.3+ (`es == false`) or GL ES 3.0, or `None` (test
    /// skipped) without EGL or a surfaceless platform.
    pub fn new(es: bool) -> Option<Gl> {
        let gl = Self::try_new(es);
        if gl.is_none() {
            eprintln!("no EGL context: test skipped");
        }
        gl
    }

    fn try_new(es: bool) -> Option<Gl> {
        let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: loads the system libEGL; EGL calls on handles created here.
        unsafe {
            let egl = egl::DynamicInstance::<egl::EGL1_5>::load_required().ok()?;
            let display = egl
                .get_platform_display(
                    PLATFORM_SURFACELESS_MESA,
                    egl::DEFAULT_DISPLAY,
                    &[egl::ATTRIB_NONE],
                )
                .ok()?;
            egl.initialize(display).ok()?;
            egl.bind_api(if es {
                egl::OPENGL_ES_API
            } else {
                egl::OPENGL_API
            })
            .ok()?;
            let renderable = if es {
                egl::OPENGL_ES3_BIT
            } else {
                egl::OPENGL_BIT
            };
            let config = egl
                .choose_first_config(
                    display,
                    &[
                        egl::RENDERABLE_TYPE,
                        renderable,
                        egl::SURFACE_TYPE,
                        egl::PBUFFER_BIT,
                        egl::NONE,
                    ],
                )
                .ok()??;
            let attrs = if es {
                vec![egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE]
            } else {
                vec![
                    egl::CONTEXT_MAJOR_VERSION,
                    3,
                    egl::CONTEXT_MINOR_VERSION,
                    3,
                    egl::CONTEXT_OPENGL_PROFILE_MASK,
                    egl::CONTEXT_OPENGL_CORE_PROFILE_BIT,
                    egl::NONE,
                ]
            };
            let context = egl.create_context(display, config, None, &attrs).ok()?;
            egl.make_current(display, None, None, Some(context)).ok()?;
            Some(Gl {
                egl,
                display,
                context,
                _lock: lock,
            })
        }
    }

    pub fn proc_address(&self, name: &str) -> *const c_void {
        self.egl
            .get_proc_address(name)
            .map_or(std::ptr::null(), |f| f as *const c_void)
    }

    /// A GL command by name, as type `T` (a function pointer type).
    pub fn func<T: Copy>(&self, name: &str) -> Result<T, String> {
        let p = self.proc_address(name);
        if p.is_null() {
            return Err(format!("{name} not found"));
        }
        // SAFETY: `T` is the function pointer type of the command.
        Ok(unsafe { std::mem::transmute_copy::<*const c_void, T>(&p) })
    }
}

impl Drop for Gl {
    fn drop(&mut self) {
        // The display is not terminated: it is shared by the process.
        let _ = self.egl.make_current(self.display, None, None, None);
        let _ = self.egl.destroy_context(self.display, self.context);
    }
}

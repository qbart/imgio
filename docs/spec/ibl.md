# Image based lighting (`imgio::ibl`)

Spec: environment conversions and prefiltering on RGBA f32 copies, any uncompressed output.

- Conventions: faces +X -X +Y -Y +Z -Z, OpenGL major axis table (`face_dir` / `dir_face`,
  `t` down the rows); equirect `(u, v)` -> `(sin v pi sin phi, cos v pi, -sin v pi cos phi)`,
  `phi = (u - 0.5) 2 pi` (center = -Z, top = +Y).
- `equirect_to_cube` / `cube_to_equirect`: bilinear (equirect wraps in u), n x n supersampling
  with n = ceil(source texels per output texel), 1..8.
- `sh_project`: bands 0..2, texel solid angles (atan area element), f64 sums;
  `Sh9::irradiance` = E / pi with band factors 1, 2/3, 1/4 (Ramamoorthi and Hanrahan).
- `irradiance`: SH or brute force cosine convolution of the box mip nearest above
  `source_size`; outputs E / pi.
- `prefilter_specular`: levels down to 4x4 by default, roughness `i / (n - 1)`, alpha =
  roughness^2; Hammersley + GGX importance sampling, N = V = R, N.L weights, source mip
  `0.5 log2(omega_s / omega_p)` (trilinear, per-face bilinear without cross-face filtering);
  level 0 is the source copy (or resampled to `size`).
- `brdf_lut` / `brdf_lut_value`: Karis split sum, Smith GGX with k = alpha / 2, Schlick
  Fresnel; x = N.V, y = roughness at texel centers, row 0 smooth; R scale, G bias.

## Audit

- tests/ibl.rs: equirect -> cube and cube -> equirect of a direction-encoded image match the
  OpenGL face table (< 0.02 / 0.03 off the poles), face round trips, orientation facts; SH and
  convolved irradiance of constant and linear environments equal the analytic E / pi (< 1e-3,
  2e-3), cosine lobe pole = 2/3; constant environments stay constant at every specular level,
  level 0 is the source byte for byte, blur grows with roughness, deterministic; split sum
  values within 0.01 of numpy brute force hemisphere integration; error cases.

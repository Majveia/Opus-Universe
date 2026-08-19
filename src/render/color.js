// Physically-derived colour.
//
// Stars, galaxies and nebulae in this universe are coloured from temperature
// rather than from a palette, using the Planckian locus in CIE chromaticity
// space and then converting to linear sRGB. A 3000 K red dwarf and a 30000 K
// O-type star therefore differ by the same amount they do in a telescope.
//
// Locus fit: Kim et al. (2002), "Design of Advanced Color Temperature Control
// System for HDTV Applications", valid from 1667 K to 25000 K.

export function planckianChromaticity(T) {
  const t = Math.min(Math.max(T, 1667), 25000);
  const t2 = t * t, t3 = t2 * t;
  let x;
  if (t < 4000) {
    x = -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910;
  } else {
    x = -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390;
  }
  const x2 = x * x, x3 = x2 * x;
  let y;
  if (t < 2222) y = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683;
  else if (t < 4000) y = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867;
  else y = 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;
  return [x, y];
}

// Linear sRGB (unclamped, normalised to unit luminance) for a blackbody at T.
export function blackbodyRGB(T) {
  const [x, y] = planckianChromaticity(T);
  const Y = 1, X = (Y / y) * x, Z = (Y / y) * (1 - x - y);
  let r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  let g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  let b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
  r = Math.max(0, r); g = Math.max(0, g); b = Math.max(0, b);
  const m = Math.max(r, g, b) || 1;
  return [r / m, g / m, b / m];
}

// Main-sequence relations, for turning a stellar mass into something to draw.
// Mass-luminosity is the standard piecewise power law; the radius and effective
// temperature follow from L = 4 pi R^2 sigma T^4.
export function mainSequence(massSolar) {
  const M = Math.max(0.07, Math.min(120, massSolar));
  let L;
  if (M < 0.43) L = 0.23 * Math.pow(M, 2.3);
  else if (M < 2) L = Math.pow(M, 4);
  else if (M < 55) L = 1.4 * Math.pow(M, 3.5);
  else L = 32000 * M;
  const R = M < 1 ? Math.pow(M, 0.8) : Math.pow(M, 0.57);
  const T = 5772 * Math.pow(L / (R * R), 0.25);
  return { mass: M, luminosity: L, radius: R, temperature: T, color: blackbodyRGB(T) };
}

// Spectral class from temperature, for labelling.
export function spectralClass(T) {
  if (T >= 30000) return 'O';
  if (T >= 10000) return 'B';
  if (T >= 7500) return 'A';
  if (T >= 6000) return 'F';
  if (T >= 5200) return 'G';
  if (T >= 3700) return 'K';
  if (T >= 2400) return 'M';
  return 'L';
}

// The GLSL counterpart. Same fit, so CPU-side labels and GPU-side pixels agree.
export const BLACKBODY_GLSL = `
vec3 blackbodyRGB(float T) {
  T = clamp(T, 1667.0, 25000.0);
  float t2 = T * T, t3 = t2 * T;
  float x;
  if (T < 4000.0) {
    x = -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / T + 0.179910;
  } else {
    x = -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / T + 0.240390;
  }
  float x2 = x * x, x3 = x2 * x;
  float y;
  if (T < 2222.0)      y = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683;
  else if (T < 4000.0) y = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867;
  else                 y =  3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;

  float Y = 1.0;
  float X = (Y / y) * x;
  float Z = (Y / y) * (1.0 - x - y);
  vec3 c = vec3(
     3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z,
    -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z,
     0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z);
  c = max(c, vec3(0.0));
  return c / max(max(c.r, c.g), max(c.b, 1e-5));
}`;

// Integer hashes shared by every procedural shader, matching src/core/rng.js so
// that CPU-generated catalogues and GPU-generated detail agree.
export const HASH_GLSL = `
uint hashU32(uint x) {
  x = (x ^ 61u) ^ (x >> 16);
  x = x + (x << 3);
  x = x ^ (x >> 4);
  x = x * 0x27d4eb2du;
  x = x ^ (x >> 15);
  return x;
}
float hash1(uint s) { return float(hashU32(s)) * (1.0 / 4294967296.0); }
float hash1f(float s) { return hash1(uint(int(s))); }
vec2 hash2(uint s) { return vec2(hash1(s), hash1(s ^ 0x9e3779b9u)); }
vec3 hash3(uint s) { return vec3(hash1(s), hash1(s ^ 0x9e3779b9u), hash1(s ^ 0x85ebca6bu)); }

// Gradient-free value noise, adequate for the smooth fields used here.
float vnoise(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float n = 0.0;
  for (int k = 0; k < 2; k++) for (int j = 0; j < 2; j++) for (int m = 0; m < 2; m++) {
    vec3 o = vec3(float(m), float(j), float(k));
    vec3 g = i + o;
    float h = hash1(uint(int(g.x) * 374761393 + int(g.y) * 668265263 + int(g.z) * 2147483647));
    float w = mix(1.0 - f.x, f.x, o.x) * mix(1.0 - f.y, f.y, o.y) * mix(1.0 - f.z, f.z, o.z);
    n += h * w;
  }
  return n * 2.0 - 1.0;
}

float fbm(vec3 p, int octaves) {
  float a = 0.5, s = 0.0, norm = 0.0;
  for (int i = 0; i < 8; i++) {
    if (i >= octaves) break;
    s += a * vnoise(p);
    norm += a;
    a *= 0.5;
    p *= 2.02;
  }
  return s / norm;
}`;

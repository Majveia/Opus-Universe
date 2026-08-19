// Deterministic pseudo-randomness. Every object in the universe is derived from
// integer coordinates through these hashes, so the same seed always rebuilds the
// same cosmos without storing anything.

// PCG-style 32-bit integer hash. Good avalanche, cheap, mirrors the GLSL version.
export function hashU32(x) {
  x = (x ^ 61) ^ (x >>> 16);
  x = (x + (x << 3)) | 0;
  x = x ^ (x >>> 4);
  x = Math.imul(x, 0x27d4eb2d);
  x = x ^ (x >>> 15);
  return x >>> 0;
}

export function hash2(x, y) { return hashU32(Math.imul(x, 0x9e3779b1) ^ hashU32(y)); }
export function hash3(x, y, z) { return hashU32(Math.imul(x, 0x9e3779b1) ^ Math.imul(y, 0x85ebca6b) ^ hashU32(z)); }

// xorshift128+ : fast, long period, fully deterministic across platforms.
export class RNG {
  constructor(seed = 1) {
    this.s0 = hashU32(seed | 0) || 1;
    this.s1 = hashU32((seed | 0) ^ 0x9e3779b9) || 2;
    this.s2 = hashU32((seed | 0) + 0x6d2b79f5) || 3;
    this.s3 = hashU32((seed | 0) ^ 0x85ebca6b) || 4;
    for (let i = 0; i < 12; i++) this.u32();
    this._spare = null;
  }
  u32() {
    // xoshiro128**
    const s0 = this.s0, s1 = this.s1, s2 = this.s2, s3 = this.s3;
    let r = Math.imul(s1, 5);
    r = ((r << 7) | (r >>> 25)) >>> 0;
    r = Math.imul(r, 9) >>> 0;
    const t = (s1 << 9) >>> 0;
    this.s2 = (s2 ^ s0) >>> 0;
    this.s3 = (s3 ^ s1) >>> 0;
    this.s1 = (s1 ^ this.s2) >>> 0;
    this.s0 = (s0 ^ this.s3) >>> 0;
    this.s2 = (this.s2 ^ t) >>> 0;
    this.s3 = ((this.s3 << 11) | (this.s3 >>> 21)) >>> 0;
    return r;
  }
  // Uniform in [0,1).
  f() { return this.u32() * 2.3283064365386963e-10; }
  range(a, b) { return a + (b - a) * this.f(); }
  int(n) { return Math.floor(this.f() * n); }
  pick(arr) { return arr[this.int(arr.length)]; }
  // Standard normal via Box-Muller with a cached spare.
  normal() {
    if (this._spare !== null) { const v = this._spare; this._spare = null; return v; }
    let u = 0, v = 0, s = 0;
    do {
      u = this.f() * 2 - 1; v = this.f() * 2 - 1; s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt(-2 * Math.log(s) / s);
    this._spare = v * m;
    return u * m;
  }
  // Uniform point on the unit sphere.
  onSphere(out) {
    const z = this.f() * 2 - 1, t = this.f() * Math.PI * 2, r = Math.sqrt(1 - z * z);
    out[0] = r * Math.cos(t); out[1] = r * Math.sin(t); out[2] = z;
    return out;
  }
  // Draw from a power law p(x) ~ x^alpha on [lo,hi] by inverse transform.
  powerLaw(lo, hi, alpha) {
    const u = this.f();
    if (Math.abs(alpha + 1) < 1e-9) return lo * Math.pow(hi / lo, u);
    const a1 = alpha + 1;
    return Math.pow(u * (Math.pow(hi, a1) - Math.pow(lo, a1)) + Math.pow(lo, a1), 1 / a1);
  }
}

// Value-noise gradient field used for procedural surfaces on the CPU side.
export function valueNoise3(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf), w = zf * zf * (3 - 2 * zf);
  const g = (i, j, k) => hash3(xi + i, yi + j, zi + k) * 2.3283064365386963e-10;
  const lerp = (a, b, t) => a + (b - a) * t;
  const c00 = lerp(g(0, 0, 0), g(1, 0, 0), u), c10 = lerp(g(0, 1, 0), g(1, 1, 0), u);
  const c01 = lerp(g(0, 0, 1), g(1, 0, 1), u), c11 = lerp(g(0, 1, 1), g(1, 1, 1), u);
  return lerp(lerp(c00, c10, v), lerp(c01, c11, v), w) * 2 - 1;
}

export function fbm3(x, y, z, octaves = 4, lac = 2.0, gain = 0.5) {
  let a = 0.5, f = 1, s = 0, norm = 0;
  for (let i = 0; i < octaves; i++) {
    s += a * valueNoise3(x * f, y * f, z * f);
    norm += a; a *= gain; f *= lac;
  }
  return s / norm;
}

// Iterative radix-2 Cooley-Tukey FFT, plus the 3D transform used to build the
// initial density field. Runs on the CPU because it happens once, at universe
// creation; the per-step transforms during evolution run on the GPU.
//
// Convention:
//   forward:  X[k] = sum_n x[n] exp(-2 pi i n k / N)
//   inverse:  x[n] = (1/N) sum_k X[k] exp(+2 pi i n k / N)

export class FFT1D {
  constructor(n) {
    if ((n & (n - 1)) !== 0) throw new Error(`FFT size must be a power of two, got ${n}`);
    this.n = n;
    this.levels = Math.log2(n) | 0;

    // Twiddle factors for the largest butterfly stage; smaller stages stride in.
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos(2 * Math.PI * i / n);
      this.sin[i] = Math.sin(2 * Math.PI * i / n);
    }

    // Bit-reversal permutation table.
    this.rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let x = i, r = 0;
      for (let j = 0; j < this.levels; j++) { r = (r << 1) | (x & 1); x >>= 1; }
      this.rev[i] = r;
    }
  }

  // In-place transform of contiguous re/im arrays of length n.
  transform(re, im, inverse = false) {
    const n = this.n, rev = this.rev, cosT = this.cos, sinT = this.sin;
    const sgn = inverse ? 1 : -1;

    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }

    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const wr = cosT[k], wi = sgn * sinT[k];
          const l = j + half;
          const tr = re[l] * wr - im[l] * wi;
          const ti = re[l] * wi + im[l] * wr;
          re[l] = re[j] - tr; im[l] = im[j] - ti;
          re[j] += tr;        im[j] += ti;
        }
      }
    }

    if (inverse) {
      const inv = 1 / n;
      for (let i = 0; i < n; i++) { re[i] *= inv; im[i] *= inv; }
    }
  }
}

// Naive DFT, used only by the test suite to prove the fast path is correct.
export function naiveDFT(re, im, inverse = false) {
  const n = re.length;
  const or_ = new Float64Array(n), oi = new Float64Array(n);
  const sgn = inverse ? 1 : -1;
  for (let k = 0; k < n; k++) {
    let sr = 0, si = 0;
    for (let t = 0; t < n; t++) {
      const ang = sgn * 2 * Math.PI * t * k / n;
      const c = Math.cos(ang), s = Math.sin(ang);
      sr += re[t] * c - im[t] * s;
      si += re[t] * s + im[t] * c;
    }
    or_[k] = inverse ? sr / n : sr;
    oi[k] = inverse ? si / n : si;
  }
  return [or_, oi];
}

// Three-dimensional complex transform over an N^3 grid stored as flat arrays
// with index = x + N*(y + N*z).
export class FFT3D {
  constructor(n) {
    this.n = n;
    this.n3 = n * n * n;
    this.fft = new FFT1D(n);
    this.lineRe = new Float64Array(n);
    this.lineIm = new Float64Array(n);
  }

  // Transforms `re`/`im` in place. Progress callback lets the loading screen
  // stay responsive on large grids.
  transform(re, im, inverse = false, onProgress = null) {
    const n = this.n, n2 = n * n;
    const lr = this.lineRe, li = this.lineIm;

    // --- along x (stride 1, already contiguous) ---
    for (let z = 0; z < n; z++) {
      for (let y = 0; y < n; y++) {
        const base = n * (y + n * z);
        for (let i = 0; i < n; i++) { lr[i] = re[base + i]; li[i] = im[base + i]; }
        this.fft.transform(lr, li, inverse);
        for (let i = 0; i < n; i++) { re[base + i] = lr[i]; im[base + i] = li[i]; }
      }
      if (onProgress && (z & 15) === 0) onProgress(z / (3 * n));
    }

    // --- along y (stride n) ---
    for (let z = 0; z < n; z++) {
      for (let x = 0; x < n; x++) {
        const base = x + n2 * z;
        for (let i = 0; i < n; i++) { const j = base + i * n; lr[i] = re[j]; li[i] = im[j]; }
        this.fft.transform(lr, li, inverse);
        for (let i = 0; i < n; i++) { const j = base + i * n; re[j] = lr[i]; im[j] = li[i]; }
      }
      if (onProgress && (z & 15) === 0) onProgress((n + z) / (3 * n));
    }

    // --- along z (stride n^2) ---
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const base = x + n * y;
        for (let i = 0; i < n; i++) { const j = base + i * n2; lr[i] = re[j]; li[i] = im[j]; }
        this.fft.transform(lr, li, inverse);
        for (let i = 0; i < n; i++) { const j = base + i * n2; re[j] = lr[i]; im[j] = li[i]; }
      }
      if (onProgress && (y & 15) === 0) onProgress((2 * n + y) / (3 * n));
    }
  }
}

// Signed wavenumber index for FFT bin j on a grid of size n: 0,1,...,n/2-1,-n/2,...,-1
export const waveIndex = (j, n) => (j <= n / 2 ? j : j - n);

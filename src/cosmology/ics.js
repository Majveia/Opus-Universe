// Cosmological initial conditions.
//
// Builds a Gaussian random density field with the given power spectrum, then
// displaces a uniform grid of particles off their lattice sites using
// second-order Lagrangian perturbation theory (2LPT). Starting from 2LPT rather
// than the Zel'dovich approximation suppresses the transients that would
// otherwise contaminate the first e-folds of growth.
//
// References:
//   Zel'dovich (1970), A&A 5, 84
//   Scoccimarro (1998), MNRAS 299, 1097
//   Crocce, Pueblas & Scoccimarro (2006), MNRAS 373, 369

import { FFT3D, waveIndex } from './fft.js';
import { RNG } from '../core/rng.js';

const TWO_PI = Math.PI * 2;

export class InitialConditions {
  /**
   * @param {object} o
   * @param {Cosmology} o.cosmology
   * @param {PowerSpectrum} o.powerSpectrum
   * @param {number} o.gridSize        N, a power of two
   * @param {number} o.boxSize         L, comoving Mpc/h
   * @param {number} o.seed
   * @param {number} o.aInit           starting scale factor
   * @param {boolean} o.use2LPT
   */
  constructor(o) {
    this.cosmo = o.cosmology;
    this.ps = o.powerSpectrum;
    this.n = o.gridSize;
    this.L = o.boxSize;
    this.seed = o.seed >>> 0;
    this.aInit = o.aInit ?? 1 / 50;
    this.use2LPT = o.use2LPT !== false;
    this.n3 = this.n ** 3;
  }

  // Runs the whole pipeline. `onProgress(fraction, label)` drives the loader.
  generate(onProgress = () => {}) {
    const n = this.n, n3 = this.n3;
    const fft = new FFT3D(n);
    const report = (f, label) => onProgress(Math.min(1, Math.max(0, f)), label);

    /* -- 1. white noise, then to Fourier space ------------------------- */
    report(0.02, 'seeding quantum fluctuations');
    const re = new Float64Array(n3);
    const im = new Float64Array(n3);
    const rng = new RNG(this.seed);
    for (let i = 0; i < n3; i++) re[i] = rng.normal();

    report(0.08, 'transforming to Fourier space');
    fft.transform(re, im, false);

    /* -- 2. impose the power spectrum ---------------------------------- */
    // A white-noise field has unit power in every mode; scaling each mode by
    // sqrt(P(k)/dV) gives exactly the target spectrum, and because the field
    // started real the spectrum is automatically Hermitian, so the inverse
    // transforms below come back real without any symmetry bookkeeping.
    report(0.22, 'applying the matter power spectrum');
    const dV = Math.pow(this.L / n, 3);         // comoving cell volume, (Mpc/h)^3
    const kFund = TWO_PI / this.L;              // fundamental wavenumber, h/Mpc
    const kx = new Int32Array(n);
    for (let i = 0; i < n; i++) kx[i] = waveIndex(i, n);

    let sumP = 0;
    for (let z = 0; z < n; z++) {
      const nz = kx[z];
      for (let y = 0; y < n; y++) {
        const ny = kx[y];
        const rowBase = n * (y + n * z);
        for (let x = 0; x < n; x++) {
          const nxi = kx[x];
          const i = rowBase + x;
          const n2 = nxi * nxi + ny * ny + nz * nz;
          if (n2 === 0) { re[i] = 0; im[i] = 0; continue; }
          const k = Math.sqrt(n2) * kFund;
          const amp = Math.sqrt(this.ps.P(k) / dV);
          re[i] *= amp; im[i] *= amp;
          sumP += this.ps.P(k);
        }
      }
    }
    // Theoretical variance of the grid-sampled field: sum over modes of P(k)/V.
    this.theoreticalSigmaGrid = Math.sqrt(sumP / (this.L ** 3));

    // Keep the linear density field (in Fourier space) for the caller.
    const deltaRe = Float64Array.from(re);
    const deltaIm = Float64Array.from(im);

    /* -- 3. first-order (Zel'dovich) displacement ---------------------- */
    report(0.3, 'solving first-order displacement');
    const psi1 = [new Float64Array(n3), new Float64Array(n3), new Float64Array(n3)];
    this._displacementFromSource(fft, deltaRe, deltaIm, psi1, (f) => report(0.3 + f * 0.25, 'solving first-order displacement'));

    /* -- 4. second-order source and displacement ----------------------- */
    let psi2 = null;
    if (this.use2LPT) {
      report(0.56, 'evaluating second-order tensor');
      psi2 = [new Float64Array(n3), new Float64Array(n3), new Float64Array(n3)];
      const source = this._secondOrderSource(fft, deltaRe, deltaIm, (f) => report(0.56 + f * 0.24, 'evaluating second-order tensor'));
      report(0.8, 'solving second-order displacement');
      // Transform the real-space source into Fourier space, then reuse the
      // same inverse-Laplacian-and-gradient machinery.
      const sRe = source, sIm = new Float64Array(n3);
      fft.transform(sRe, sIm, false);
      sRe[0] = 0; sIm[0] = 0;
      this._displacementFromSource(fft, sRe, sIm, psi2, (f) => report(0.8 + f * 0.14, 'solving second-order displacement'));
    }

    /* -- 5. displace the particles ------------------------------------- */
    report(0.95, 'placing particles on the light cone');
    const a = this.aInit;
    const D1 = this.cosmo.growth(a);
    const dD1 = this.cosmo.growthDeriv(a);
    const E = this.cosmo.E(a);
    // Second-order growth: D2 = -(3/7) D1^2 Omega_m(a)^(-1/143) (Bouchet 1995).
    // The sign is folded in here so that x = q + D1 psi1 + E2 psi2.
    const omA = this.cosmo.omegaMz(a);
    const k2 = (3 / 7) * Math.pow(omA, -1 / 143);
    const E2 = k2 * D1 * D1;
    const dE2 = k2 * 2 * D1 * dD1;

    // Canonical momentum p = a^2 dx/dt, in box units per Hubble time.
    const pv1 = a * a * a * E * dD1;
    const pv2 = a * a * a * E * dE2;

    const pos = new Float32Array(n3 * 3);
    const vel = new Float32Array(n3 * 3);
    const invN = 1 / n;
    let maxDisp = 0, sumDisp = 0;

    for (let z = 0; z < n; z++) {
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          const i = x + n * (y + n * z);
          const q = [x * invN, y * invN, z * invN];
          let d2 = 0;
          for (let c = 0; c < 3; c++) {
            const s1 = psi1[c][i];
            const s2 = psi2 ? psi2[c][i] : 0;
            const disp = D1 * s1 + E2 * s2;
            d2 += disp * disp;
            let p = q[c] + disp;
            p -= Math.floor(p);                 // periodic wrap into [0,1)
            pos[i * 3 + c] = p;
            vel[i * 3 + c] = pv1 * s1 + pv2 * s2;
          }
          const d = Math.sqrt(d2);
          if (d > maxDisp) maxDisp = d;
          sumDisp += d;
        }
      }
    }

    /* -- 6. the real-space linear density, for diagnostics and colouring - */
    report(0.98, 'measuring the linear field');
    const dRe = Float64Array.from(deltaRe), dIm = Float64Array.from(deltaIm);
    fft.transform(dRe, dIm, true);
    const delta = new Float32Array(n3);
    let s = 0, s2 = 0;
    for (let i = 0; i < n3; i++) { const v = dRe[i]; delta[i] = v; s += v; s2 += v * v; }
    const mean = s / n3;
    this.measuredSigmaGrid = Math.sqrt(s2 / n3 - mean * mean);

    report(1, 'ready');

    return {
      positions: pos,
      velocities: vel,
      linearDelta: delta,
      stats: {
        gridSize: n,
        boxSize: this.L,
        aInit: a,
        zInit: 1 / a - 1,
        D1, dD1, E2,
        meanDisplacementCells: (sumDisp / n3) * n,
        maxDisplacementCells: maxDisp * n,
        sigmaGridTheory: this.theoreticalSigmaGrid,
        sigmaGridMeasured: this.measuredSigmaGrid,
        sigmaLinearAtInit: this.measuredSigmaGrid * D1,
        nyquistK: Math.PI * n / this.L,
        fundamentalK: kFund,
        particleMass: this.cosmo.rhoMeanComoving * Math.pow(this.L, 3) / n3,
      },
    };
  }

  // psi_j = FFT^-1[ i n_j S(k) / (2 pi |n|^2) ], the gradient of the inverse
  // Laplacian, expressed in box units where the fundamental mode is n = 1.
  _displacementFromSource(fft, srcRe, srcIm, out, onProgress = () => {}) {
    const n = this.n, n3 = this.n3;
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = waveIndex(i, n);
    const tRe = new Float64Array(n3), tIm = new Float64Array(n3);

    for (let c = 0; c < 3; c++) {
      for (let z = 0; z < n; z++) {
        const nz = idx[z];
        for (let y = 0; y < n; y++) {
          const ny = idx[y];
          const base = n * (y + n * z);
          for (let x = 0; x < n; x++) {
            const nx = idx[x];
            const i = base + x;
            const nn = nx * nx + ny * ny + nz * nz;
            if (nn === 0) { tRe[i] = 0; tIm[i] = 0; continue; }
            const nj = c === 0 ? nx : c === 1 ? ny : nz;
            const f = nj / (TWO_PI * nn);
            // multiply by i * f: (re + i im) * i f = -f*im + i f*re
            tRe[i] = -f * srcIm[i];
            tIm[i] = f * srcRe[i];
          }
        }
      }
      fft.transform(tRe, tIm, true);
      out[c].set(tRe);
      onProgress((c + 1) / 3);
    }
  }

  // delta^(2) = sum_{i<j} [ phi_,ii phi_,jj - phi_,ij^2 ], where phi is the
  // first-order Lagrangian potential with laplacian(phi) = delta.
  // In Fourier space phi_,ij = FFT^-1[ -n_i n_j / |n|^2 * delta(k) ]... the
  // (2 pi)^2 factors from the box-unit gradient cancel against those in the
  // inverse Laplacian, leaving the clean ratio below.
  _secondOrderSource(fft, deltaRe, deltaIm, onProgress = () => {}) {
    const n = this.n, n3 = this.n3;
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = waveIndex(i, n);

    const pairs = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]];
    const comp = pairs.map(() => new Float32Array(n3));
    const tRe = new Float64Array(n3), tIm = new Float64Array(n3);

    for (let pi = 0; pi < pairs.length; pi++) {
      const [ci, cj] = pairs[pi];
      for (let z = 0; z < n; z++) {
        const nv = [0, 0, idx[z]];
        for (let y = 0; y < n; y++) {
          nv[1] = idx[y];
          const base = n * (y + n * z);
          for (let x = 0; x < n; x++) {
            nv[0] = idx[x];
            const i = base + x;
            const nn = nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2];
            if (nn === 0) { tRe[i] = 0; tIm[i] = 0; continue; }
            const f = -(nv[ci] * nv[cj]) / nn;
            tRe[i] = f * deltaRe[i];
            tIm[i] = f * deltaIm[i];
          }
        }
      }
      fft.transform(tRe, tIm, true);
      const dst = comp[pi];
      for (let i = 0; i < n3; i++) dst[i] = tRe[i];
      onProgress((pi + 1) / pairs.length);
    }

    const [xx, yy, zz, xy, xz, yz] = comp;
    const src = new Float64Array(n3);
    for (let i = 0; i < n3; i++) {
      src[i] = xx[i] * yy[i] - xy[i] * xy[i]
             + xx[i] * zz[i] - xz[i] * xz[i]
             + yy[i] * zz[i] - yz[i] * yz[i];
    }
    return src;
  }
}

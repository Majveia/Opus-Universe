// End-to-end validation of the initial-condition generator. The strongest test
// is a closed loop: draw a random field from P(k), then measure P(k) back out
// of it and check the two agree bin by bin.

import { Cosmology } from '../src/cosmology/cosmology.js';
import { PowerSpectrum } from '../src/cosmology/powerspectrum.js';
import { InitialConditions } from '../src/cosmology/ics.js';
import { FFT3D, waveIndex } from '../src/cosmology/fft.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`${cond ? '  PASS' : '  FAIL'}  ${name}${detail ? '\n          ' + detail : ''}`);
  if (!cond) failures++;
};

const N = 64, L = 150.0;
const cosmo = new Cosmology();
const ps = new PowerSpectrum(cosmo);
const ic = new InitialConditions({
  cosmology: cosmo, powerSpectrum: ps, gridSize: N, boxSize: L, seed: 20240819, aInit: 1 / 50,
});

console.log(`Initial conditions: N=${N}, L=${L} Mpc/h, z_init=49\n`);
const t0 = Date.now();
const out = await ic.generate();
const dt = Date.now() - t0;
const s = out.stats;
console.log(`  generated in ${dt} ms`);
console.log(`  D(a_init) = ${s.D1.toExponential(4)}   particle mass = ${s.particleMass.toExponential(3)} Msun/h`);
console.log(`  k_fundamental = ${s.fundamentalK.toFixed(4)} h/Mpc   k_Nyquist = ${s.nyquistK.toFixed(3)} h/Mpc`);
console.log(`  mean displacement = ${s.meanDisplacementCells.toFixed(4)} cells   max = ${s.maxDisplacementCells.toFixed(3)} cells\n`);

/* ---- 1. field variance matches the sum over modes --------------------- */
{
  const rel = Math.abs(s.sigmaGridMeasured - s.sigmaGridTheory) / s.sigmaGridTheory;
  check('linear field variance matches sum_k P(k)/V', rel < 0.05,
    `measured sigma = ${s.sigmaGridMeasured.toFixed(4)}, theory = ${s.sigmaGridTheory.toFixed(4)}, rel diff ${(rel * 100).toFixed(2)}%`);
}

/* ---- 2. measured P(k) vs input P(k) ----------------------------------- */
{
  const n3 = N ** 3;
  const re = Float64Array.from(out.linearDelta);
  const im = new Float64Array(n3);
  new FFT3D(N).transform(re, im, false);

  const V = L ** 3, kF = 2 * Math.PI / L;
  const nBins = 14;
  const kMin = kF, kMax = Math.PI * N / L;
  const lo = Math.log(kMin), hi = Math.log(kMax);
  // Accumulate the theory over the very same modes. Comparing a bin average
  // against P(<k>) would bias high wherever P(k) is convex, which at high k it
  // strongly is - that is a property of the binning, not of the field.
  const sumP = new Float64Array(nBins), sumK = new Float64Array(nBins);
  const sumT = new Float64Array(nBins), cnt = new Int32Array(nBins);

  const wi = new Int32Array(N);
  for (let i = 0; i < N; i++) wi[i] = waveIndex(i, N);
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const nx = wi[x], ny = wi[y], nz = wi[z];
    const nn = nx * nx + ny * ny + nz * nz;
    if (nn === 0) continue;
    const k = Math.sqrt(nn) * kF;
    if (k < kMin || k > kMax) continue;
    const b = Math.floor((Math.log(k) - lo) / (hi - lo) * nBins);
    if (b < 0 || b >= nBins) continue;
    const i = x + N * (y + N * z);
    const p = (re[i] * re[i] + im[i] * im[i]) * V / (n3 * n3);
    sumP[b] += p; sumK[b] += k; sumT[b] += ps.P(k); cnt[b]++;
  }

  console.log('  measured vs input power spectrum');
  console.log('     k [h/Mpc]   modes   P_measured      P_input      ratio    dev');
  // The field was built from real white noise, so mode k and mode -k are
  // conjugates rather than independent draws. Each bin therefore carries
  // N_modes/2 independent complex amplitudes, and |delta_k|^2 averaged over
  // them is chi-squared distributed with relative scatter 2/sqrt(N_modes).
  let worstDev = 0, worstK = 0, nUsed = 0, sumW = 0, sumWR = 0;
  for (let b = 0; b < nBins; b++) {
    if (cnt[b] < 30) continue;
    const k = sumK[b] / cnt[b];
    const pm = sumP[b] / cnt[b];
    const pi = sumT[b] / cnt[b];
    const r = pm / pi;
    const sigma = 2 / Math.sqrt(cnt[b]);
    const dev = (r - 1) / sigma;
    nUsed++;
    sumW += cnt[b]; sumWR += cnt[b] * r;
    if (Math.abs(dev) > Math.abs(worstDev)) { worstDev = dev; worstK = k; }
    console.log(`     ${k.toFixed(4).padStart(8)} ${String(cnt[b]).padStart(7)}   ${pm.toExponential(4)}   ${pi.toExponential(4)}   ${r.toFixed(4)}  ${dev >= 0 ? '+' : ''}${dev.toFixed(2)}s`);
  }
  const meanR = sumWR / sumW;
  check('binned P(k) recovers the input spectrum', Math.abs(meanR - 1) < 0.05 && Math.abs(worstDev) < 3.5,
    `mode-weighted mean ratio ${meanR.toFixed(4)} over ${nUsed} bins; largest deviation ${worstDev.toFixed(2)} sigma at k=${worstK.toFixed(4)} (cosmic variance allows up to ~3)`);
}

/* ---- 3. displacement field has zero mean (no bulk drift) -------------- */
{
  const n3 = N ** 3;
  let mx = 0, my = 0, mz = 0;
  for (let i = 0; i < n3; i++) { mx += out.velocities[i * 3]; my += out.velocities[i * 3 + 1]; mz += out.velocities[i * 3 + 2]; }
  mx /= n3; my /= n3; mz /= n3;
  let rms = 0;
  for (let i = 0; i < n3; i++) {
    rms += out.velocities[i * 3] ** 2 + out.velocities[i * 3 + 1] ** 2 + out.velocities[i * 3 + 2] ** 2;
  }
  rms = Math.sqrt(rms / n3);
  const drift = Math.hypot(mx, my, mz) / rms;
  check('no net momentum in the box', drift < 1e-4,
    `|<p>| / p_rms = ${drift.toExponential(2)}`);
}

/* ---- 4. CIC deposit of the displaced particles, with the assignment window - */
{
  const grid = new Float64Array(N ** 3);
  const pos = out.positions, np = N ** 3;
  for (let p = 0; p < np; p++) {
    const gx = pos[p * 3] * N, gy = pos[p * 3 + 1] * N, gz = pos[p * 3 + 2] * N;
    const i0 = Math.floor(gx), j0 = Math.floor(gy), k0 = Math.floor(gz);
    const fx = gx - i0, fy = gy - j0, fz = gz - k0;
    for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
      const ii = ((i0 + dx) % N + N) % N, jj = ((j0 + dy) % N + N) % N, kk = ((k0 + dz) % N + N) % N;
      grid[ii + N * (jj + N * kk)] += w;
    }
  }
  let sum = 0, sum2 = 0;
  for (let i = 0; i < grid.length; i++) { const d = grid[i] - 1; sum += d; sum2 += d * d; }
  const measured = Math.sqrt(sum2 / grid.length);

  // Cloud-in-cell assignment convolves the field with a triangular kernel, so
  // the sampled variance is suppressed by the CIC window W(k) = prod_i
  // sinc^2(pi n_i / N). Aliasing of power from beyond the Nyquist frequency
  // pushes it back up, so the honest expectation is a value between the
  // windowed and unwindowed sums - much nearer the windowed one.
  const kF = 2 * Math.PI / L, V = L ** 3;
  const wi = (j) => (j <= N / 2 ? j : j - N);
  const sinc = (x) => (Math.abs(x) < 1e-12 ? 1 : Math.sin(x) / x);
  let sPlain = 0, sCIC = 0;
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const nx = wi(x), ny = wi(y), nz = wi(z), nn = nx * nx + ny * ny + nz * nz;
    if (!nn) continue;
    const P = ps.P(Math.sqrt(nn) * kF);
    const W = Math.pow(sinc(Math.PI * nx / N) * sinc(Math.PI * ny / N) * sinc(Math.PI * nz / N), 2);
    sPlain += P; sCIC += P * W * W;
  }
  const expectedCIC = Math.sqrt(sCIC / V) * s.D1;
  const expectedPlain = Math.sqrt(sPlain / V) * s.D1;
  const rel = Math.abs(measured - expectedCIC) / expectedCIC;
  check('CIC density of displaced particles matches the windowed prediction',
    rel < 0.15 && measured > expectedCIC * 0.9 && measured < expectedPlain,
    `measured ${measured.toExponential(4)}; CIC-windowed theory ${expectedCIC.toExponential(4)} (${(rel * 100).toFixed(1)}% high, aliasing); unwindowed ${expectedPlain.toExponential(4)}; mean delta ${(sum / grid.length).toExponential(2)}`);
}

/* ---- 5. 2LPT correction is a genuine, subdominant correction ---------- */
{
  const ic1 = new InitialConditions({
    cosmology: cosmo, powerSpectrum: ps, gridSize: N, boxSize: L, seed: 20240819, aInit: 1 / 50, use2LPT: false,
  });
  const zel = await ic1.generate();
  let diff = 0, mag = 0;
  const n3 = N ** 3;
  for (let i = 0; i < n3; i++) {
    for (let c = 0; c < 3; c++) {
      let d = out.positions[i * 3 + c] - zel.positions[i * 3 + c];
      d -= Math.round(d);                 // periodic difference
      diff += d * d;
      let q = zel.positions[i * 3 + c] - [(i % N), ((i / N) | 0) % N, (i / (N * N)) | 0][c] / N;
      q -= Math.round(q);
      mag += q * q;
    }
  }
  const ratio = Math.sqrt(diff / mag);
  check('2LPT correction is present and subdominant at z=49', ratio > 1e-4 && ratio < 0.2,
    `|x_2LPT - x_ZA| / |x_ZA - q| = ${ratio.toExponential(3)}`);
}

/* ---- 6. determinism ---------------------------------------------------- */
{
  const again = await new InitialConditions({
    cosmology: cosmo, powerSpectrum: ps, gridSize: 32, boxSize: L, seed: 777, aInit: 1 / 50,
  }).generate();
  const twice = await new InitialConditions({
    cosmology: cosmo, powerSpectrum: ps, gridSize: 32, boxSize: L, seed: 777, aInit: 1 / 50,
  }).generate();
  let same = true;
  for (let i = 0; i < again.positions.length; i++) if (again.positions[i] !== twice.positions[i]) { same = false; break; }
  check('same seed reproduces the identical universe', same);
}

console.log(failures === 0 ? '\nAll initial-condition tests passed.' : `\n${failures} test(s) FAILED.`);
process.exit(failures ? 1 : 0);

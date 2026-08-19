// Runs the Particle-Mesh solver in the browser and measures whether the
// simulated universe grows structure at the rate linear theory predicts.
import { GLContext } from '../src/core/gl.js';
import { Cosmology } from '../src/cosmology/cosmology.js';
import { PowerSpectrum } from '../src/cosmology/powerspectrum.js';
import { InitialConditions } from '../src/cosmology/ics.js';
import { ParticleMesh } from '../src/sim/pm.js';
import { atlasIndex } from '../src/sim/atlas.js';
import { FFT3D, waveIndex } from '../src/cosmology/fft.js';

export default async function run({ N = 32, L = 200, steps = 250, aInit = 1 / 50, spectral = true, kBins = null }) {
  const canvas = document.getElementById('c');
  const ctx = new GLContext(canvas);

  const cosmo = new Cosmology();
  const ps = new PowerSpectrum(cosmo);
  const ic = new InitialConditions({
    cosmology: cosmo, powerSpectrum: ps, gridSize: N, boxSize: L, seed: 424242, aInit,
  });
  const gen = await ic.generate();

  const pm = new ParticleMesh(ctx, {
    cosmology: cosmo, gridSize: N, boxSize: L,
    positions: gen.positions, velocities: gen.velocities,
    aInit, aMax: 1.0, steps,
  });
  pm.spectralGradient = spectral;

  // Density field in grid order, from the atlas readback.
  const layout = pm.layout;
  const grabDelta = () => {
    pm.depositDensity();
    const flat = pm.readDensity();
    const out = new Float64Array(N ** 3);
    for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      out[x + N * (y + N * z)] = flat[atlasIndex(layout, x, y, z)] / pm.meanDensity - 1;
    }
    return out;
  };

  const d0 = grabDelta();

  const t0 = performance.now();
  let n = 0;
  while (pm.step()) n++;
  ctx.gl.finish();
  const elapsed = performance.now() - t0;

  const d1 = grabDelta();

  // Transform both fields.
  const fft = new FFT3D(N);
  const r0 = Float64Array.from(d0), i0 = new Float64Array(N ** 3);
  const r1 = Float64Array.from(d1), i1 = new Float64Array(N ** 3);
  fft.transform(r0, i0, false);
  fft.transform(r1, i1, false);

  // Mode-by-mode growth via a matched filter: sum Re(d1 d0*) / sum |d0|^2.
  // Using the same realisation at both epochs cancels cosmic variance, so even
  // the handful of modes in the largest-scale bin give a sharp measurement.
  const kF = 2 * Math.PI / L;
  const edges = kBins || [0.03, 0.06, 0.10, 0.16, 0.25, 0.40, 0.70];
  const bins = [];
  for (let i = 0; i < edges.length - 1; i++) {
    bins.push({ lo: edges[i], hi: edges[i + 1], num: 0, den: 0, auto: 0, cnt: 0, sumK: 0 });
  }

  const wi = new Int32Array(N);
  for (let i = 0; i < N; i++) wi[i] = waveIndex(i, N);
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const nx = wi[x], ny = wi[y], nz = wi[z], nn = nx * nx + ny * ny + nz * nz;
    if (!nn) continue;
    const k = Math.sqrt(nn) * kF;
    const b = bins.find((bb) => k >= bb.lo && k < bb.hi);
    if (!b) continue;
    const i = x + N * (y + N * z);
    b.num += r1[i] * r0[i] + i1[i] * i0[i];      // cross power
    b.den += r0[i] * r0[i] + i0[i] * i0[i];      // initial auto power
    b.auto += r1[i] * r1[i] + i1[i] * i1[i];     // final auto power
    b.cnt++; b.sumK += k;
  }

  const expected = cosmo.growth(1) / cosmo.growth(aInit);
  const growth = bins.filter((b) => b.cnt > 0).map((b) => ({
    k: b.sumK / b.cnt, modes: b.cnt,
    cross: b.num / b.den,                        // decorrelation-sensitive
    autoRatio: Math.sqrt(b.auto / b.den),        // amplitude growth
    // How much of the final field still traces the initial one. Unity means
    // purely linear evolution; below unity means mode coupling has scrambled it.
    correlation: b.num / Math.sqrt(b.den * b.auto),
    expected,
  }));

  // Momentum conservation: the box must not acquire a net drift.
  const velFlat = new Float32Array(layout.width * layout.height * 4);
  pm.velFBO[pm.curVel].readPixels(velFlat);
  let mx = 0, my = 0, mz = 0, rms = 0, np = 0;
  for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const a = atlasIndex(layout, x, y, z);
    mx += velFlat[a * 4]; my += velFlat[a * 4 + 1]; mz += velFlat[a * 4 + 2];
    rms += velFlat[a * 4] ** 2 + velFlat[a * 4 + 1] ** 2 + velFlat[a * 4 + 2] ** 2;
    np++;
  }
  const drift = Math.hypot(mx / np, my / np, mz / np) / Math.sqrt(rms / np);

  // Density statistics: a mature cosmic web is strongly non-Gaussian.
  const stat = (d) => {
    let s = 0, s2 = 0, s3 = 0, max = -1e30;
    for (let i = 0; i < d.length; i++) { const v = d[i]; s += v; s2 += v * v; s3 += v * v * v; max = Math.max(max, v); }
    const m = s / d.length, v = s2 / d.length - m * m;
    return { mean: m, sigma: Math.sqrt(v), skew: (s3 / d.length) / Math.pow(v, 1.5), max };
  };

  const out = {
    renderer: ctx.rendererName,
    steps: n, elapsedMs: elapsed, msPerStep: elapsed / n,
    N, L, aInit, expectedGrowth: expected, spectral,
    growth,
    drift,
    initial: stat(d0),
    final: stat(d1),
    particleCount: pm.particleCount,
  };
  pm.dispose();
  return out;
}

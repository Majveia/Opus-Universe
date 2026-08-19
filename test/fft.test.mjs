import { FFT1D, FFT3D, naiveDFT, waveIndex } from '../src/cosmology/fft.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`${cond ? '  PASS' : '  FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) failures++;
};

console.log('FFT correctness');

// 1. Against a naive DFT for several sizes.
for (const n of [2, 4, 8, 16, 64]) {
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) { re[i] = Math.sin(i * 1.7) + i * 0.01; im[i] = Math.cos(i * 0.3); }
  const [nr, ni] = naiveDFT(re, im, false);
  const fr = Float64Array.from(re), fi = Float64Array.from(im);
  new FFT1D(n).transform(fr, fi, false);
  let err = 0;
  for (let i = 0; i < n; i++) err = Math.max(err, Math.abs(fr[i] - nr[i]), Math.abs(fi[i] - ni[i]));
  check(`forward N=${n} matches naive DFT`, err < 1e-9, `max err ${err.toExponential(2)}`);
}

// 2. Round trip.
{
  const n = 256;
  const f = new FFT1D(n);
  const re = new Float64Array(n), im = new Float64Array(n);
  const re0 = new Float64Array(n), im0 = new Float64Array(n);
  for (let i = 0; i < n; i++) { re[i] = re0[i] = Math.random() * 2 - 1; im[i] = im0[i] = Math.random() * 2 - 1; }
  f.transform(re, im, false);
  f.transform(re, im, true);
  let err = 0;
  for (let i = 0; i < n; i++) err = Math.max(err, Math.abs(re[i] - re0[i]), Math.abs(im[i] - im0[i]));
  check('1D round trip is identity', err < 1e-12, `max err ${err.toExponential(2)}`);
}

// 3. A pure tone must land in exactly one bin.
{
  const n = 64, k0 = 7;
  const f = new FFT1D(n);
  const re = new Float64Array(n), im = new Float64Array(n);
  // exp(+2 pi i k0 t / n); under the forward kernel exp(-2 pi i n k / N) this
  // must land in bin +k0 (the conjugate signal would land in bin n - k0).
  for (let i = 0; i < n; i++) { re[i] = Math.cos(2 * Math.PI * k0 * i / n); im[i] = Math.sin(2 * Math.PI * k0 * i / n); }
  f.transform(re, im, false);
  let peak = -1, peakVal = 0, leak = 0;
  for (let i = 0; i < n; i++) {
    const m = Math.hypot(re[i], im[i]);
    if (m > peakVal) { peakVal = m; peak = i; }
  }
  for (let i = 0; i < n; i++) if (i !== peak) leak = Math.max(leak, Math.hypot(re[i], im[i]));
  check(`pure tone k=${k0} lands in bin ${k0}`, peak === k0 && peakVal > n - 1e-6 && leak < 1e-9,
    `bin ${peak}, amp ${peakVal.toFixed(4)}, leak ${leak.toExponential(2)}`);
}

// 4. Parseval's theorem.
{
  const n = 512, f = new FFT1D(n);
  const re = new Float64Array(n), im = new Float64Array(n);
  let e1 = 0;
  for (let i = 0; i < n; i++) { re[i] = Math.random() - 0.5; im[i] = Math.random() - 0.5; e1 += re[i] ** 2 + im[i] ** 2; }
  f.transform(re, im, false);
  let e2 = 0;
  for (let i = 0; i < n; i++) e2 += re[i] ** 2 + im[i] ** 2;
  check("Parseval's theorem", Math.abs(e2 / n - e1) / e1 < 1e-12, `${(e2 / n).toFixed(6)} vs ${e1.toFixed(6)}`);
}

// 5. 3D round trip and a known 3D plane wave.
{
  const n = 16, f3 = new FFT3D(n), n3 = n * n * n;
  const re = new Float64Array(n3), im = new Float64Array(n3);
  const re0 = new Float64Array(n3);
  for (let i = 0; i < n3; i++) { re[i] = re0[i] = Math.random() - 0.5; }
  f3.transform(re, im, false);
  f3.transform(re, im, true);
  let err = 0, imErr = 0;
  for (let i = 0; i < n3; i++) { err = Math.max(err, Math.abs(re[i] - re0[i])); imErr = Math.max(imErr, Math.abs(im[i])); }
  check('3D round trip is identity', err < 1e-12 && imErr < 1e-12, `re err ${err.toExponential(2)}, im ${imErr.toExponential(2)}`);

  // A real 3D plane wave cos(2 pi (2x + 3y - z)/n) must produce exactly two
  // conjugate bins at (2,3,-1) and (-2,-3,1).
  const kx = 2, ky = 3, kz = -1;
  const re2 = new Float64Array(n3), im2 = new Float64Array(n3);
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    re2[x + n * (y + n * z)] = Math.cos(2 * Math.PI * (kx * x + ky * y + kz * z) / n);
  }
  f3.transform(re2, im2, false);
  let nBig = 0, bigList = [];
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = x + n * (y + n * z);
    if (Math.hypot(re2[i], im2[i]) > 1e-6) { nBig++; bigList.push([waveIndex(x, n), waveIndex(y, n), waveIndex(z, n)]); }
  }
  const has = (a, b, c) => bigList.some(v => v[0] === a && v[1] === b && v[2] === c);
  check('3D plane wave -> two conjugate bins', nBig === 2 && has(kx, ky, kz) && has(-kx, -ky, -kz),
    `${nBig} bins: ${JSON.stringify(bigList)}`);
}

// 6. Hermitian symmetry of the transform of a real field - the property the
//    initial-condition generator relies on to produce a real displacement.
{
  const n = 16, f3 = new FFT3D(n), n3 = n * n * n;
  const re = new Float64Array(n3), im = new Float64Array(n3);
  for (let i = 0; i < n3; i++) re[i] = Math.random() - 0.5;
  f3.transform(re, im, false);
  const idx = (x, y, z) => ((x % n) + n) % n + n * (((y % n) + n) % n + n * (((z % n) + n) % n));
  let err = 0;
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const a = idx(x, y, z), b = idx(-x, -y, -z);
    err = Math.max(err, Math.abs(re[a] - re[b]), Math.abs(im[a] + im[b]));
  }
  check('real field -> Hermitian spectrum', err < 1e-11, `max err ${err.toExponential(2)}`);
}

console.log(failures === 0 ? '\nAll FFT tests passed.' : `\n${failures} FFT test(s) FAILED.`);
process.exit(failures ? 1 : 0);

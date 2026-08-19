// Runs inside the browser: compares the GPU FFT against the CPU reference.
import { GLContext } from '../src/core/gl.js';
import { atlasLayout, atlasIndex } from '../src/sim/atlas.js';
import { GPUFFT3D } from '../src/sim/gpufft.js';
import { FFT3D } from '../src/cosmology/fft.js';

export default async function run({ sizes }) {
  const canvas = document.getElementById('c');
  const ctx = new GLContext(canvas);
  const gl = ctx.gl;
  const results = [];

  for (const N of sizes) {
    const layout = atlasLayout(N);
    const fft = new GPUFFT3D(ctx, layout);
    const n3 = N ** 3;

    // A deterministic complex field.
    const re = new Float64Array(n3), im = new Float64Array(n3);
    let seed = 12345;
    const rnd = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296 - 0.5;
    };
    for (let i = 0; i < n3; i++) { re[i] = rnd(); im[i] = rnd(); }

    // Upload in atlas order.
    const data = new Float32Array(layout.width * layout.height * 2);
    for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const gi = x + N * (y + N * z);
      const ai = atlasIndex(layout, x, y, z);
      data[ai * 2] = re[gi]; data[ai * 2 + 1] = im[gi];
    }
    fft.a.tex.upload(data);

    // GPU forward.
    fft.transform(false);
    const out = new Float32Array(layout.width * layout.height * 2);
    fft.currentFBO.readPixels(out);

    // CPU reference.
    const cre = Float64Array.from(re), cim = Float64Array.from(im);
    new FFT3D(N).transform(cre, cim, false);

    let maxErr = 0, maxMag = 0;
    for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const gi = x + N * (y + N * z);
      const ai = atlasIndex(layout, x, y, z);
      maxErr = Math.max(maxErr, Math.abs(out[ai * 2] - cre[gi]), Math.abs(out[ai * 2 + 1] - cim[gi]));
      maxMag = Math.max(maxMag, Math.abs(cre[gi]), Math.abs(cim[gi]));
    }

    // Round trip: inverse should restore the original field.
    fft.a.tex.upload(data);
    fft.transform(false);
    fft.transform(true);
    const rt = new Float32Array(layout.width * layout.height * 2);
    fft.currentFBO.readPixels(rt);
    let rtErr = 0;
    for (let z = 0; z < N; z++) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const gi = x + N * (y + N * z);
      const ai = atlasIndex(layout, x, y, z);
      rtErr = Math.max(rtErr, Math.abs(rt[ai * 2] - re[gi]), Math.abs(rt[ai * 2 + 1] - im[gi]));
    }

    results.push({ N, maxErr, maxMag, relErr: maxErr / maxMag, rtErr,
      atlas: `${layout.width}x${layout.height}` });
    fft.dispose();
  }
  return { renderer: ctx.rendererName, results };
}

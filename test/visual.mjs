// Loads the real application in a headless browser, runs the universe forward,
// and captures screenshots. This is the test that answers "does it actually
// look like anything", which no unit test can.

import { serve, launch } from './harness.mjs';
import fs from 'node:fs';
import path from 'node:path';

const OUT = process.argv[2] || '/tmp/opus-shots';
fs.mkdirSync(OUT, { recursive: true });

const { server, port } = await serve();
const origin = `http://127.0.0.1:${port}`;
const { browser, page } = await launch({ width: 1280, height: 720 });

const shot = async (name) => {
  const p = path.join(OUT, `${name}.png`);
  await page.screenshot({ path: p, timeout: 180000 });
  const kb = (fs.statSync(p).size / 1024).toFixed(0);
  console.log(`  captured ${name}.png  (${kb} kB)`);
  return p;
};

// Mean luminance and the fraction of non-black pixels, so a blank frame is
// caught by the test rather than by eye.
const measure = () => page.evaluate(() => window.__opus.measureFrame());

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? '   ' + detail : ''}`);
  if (!cond) failures++;
};

try {
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const query = process.env.OPUS_QUERY || 'preserve=1&maxpixels=921600&seed=20240819';
  const pagePath = process.env.OPUS_PAGE || '/index.html';
  await page.goto(`${origin}${pagePath}?${query}`, { waitUntil: 'load' });
  console.log(`  page: ${pagePath}`);

  console.log('waiting for the universe to be built...');
  await page.waitForFunction(() => document.getElementById('loader').classList.contains('done'), { timeout: 180000 });
  const info = await page.evaluate(() => ({
    renderer: window.__opus.ctx.rendererName,
    grid: window.__opus.state.gridSize,
    particles: window.__opus.universe.pm.particleCount,
    box: window.__opus.state.boxSize,
    quality: window.__opus.state.quality.label,
    stats: window.__opus.universe.stats,
  }));
  console.log(`  ${info.renderer}`);
  console.log(`  quality "${info.quality}": ${info.grid}^3 mesh, ${info.particles} particles, ${info.box} Mpc/h box`);
  check('no page errors during startup', errors.length === 0, errors.join(' | '));

  // Advance to a given redshift by driving the solver directly.
  const runTo = async (targetZ) => {
    await page.evaluate((z) => new Promise((resolve) => {
      const pm = window.__opus.universe.pm;
      const a = 1 / (1 + z);
      const tick = () => {
        let n = 0;
        while (pm.a < a && n < 40 && pm.step()) n++;
        if (pm.a >= a || pm.finished) resolve(); else requestAnimationFrame(tick);
      };
      tick();
    }), targetZ);
    await page.waitForTimeout(220);
  };

  const setView = async (key) => {
    await page.evaluate((k) => window.__opus.placeCamera(k), key);
    await page.waitForTimeout(420);
  };

  const results = [];
  const epochs = process.env.OPUS_EPOCHS
    ? JSON.parse(process.env.OPUS_EPOCHS)
    : [[49, 'z49-initial'], [2, 'z02-cosmic-noon'], [0, 'z00-today']];
  for (const [z, label] of epochs) {
    await runTo(z);
    await setView(2);
    const m = await measure();
    await shot(label);
    results.push({ label, ...m });
    console.log(`     z=${String(z).padEnd(3)} mean luma ${m.meanLuma.toFixed(2).padStart(6)}   lit ${(m.litFraction * 100).toFixed(1)}%   peak ${m.maxLuma}`);
  }

  const initial = results[0], final = results[results.length - 1];
  check('something is actually rendered', final.meanLuma > 1 && final.litFraction > 0.02,
    `mean luma ${final.meanLuma.toFixed(2)}, ${(final.litFraction * 100).toFixed(1)}% of pixels lit`);
  check('the frame is not blown out', final.litFraction < 0.9);
  check('structure becomes more contrasty as the universe evolves',
    final.maxLuma >= initial.maxLuma,
    `peak ${initial.maxLuma} -> ${final.maxLuma}`);

  // Other vantage points at z = 0.
  for (const [k, name] of (process.env.OPUS_VANTAGES === '0' ? [] : [[1, 'vantage-whole-volume'], [3, 'vantage-inside-filament']])) {
    await setView(k);
    await shot(name);
  }

  check('no page errors during the whole run', errors.length === 0, errors.join(' | '));
  console.log(`\n  screenshots in ${OUT}`);
} catch (e) {
  console.error('visual test error:', e.message, e.stack);
  failures++;
} finally {
  await browser.close();
  server.close();
}
process.exit(failures ? 1 : 0);

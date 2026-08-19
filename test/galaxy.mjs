// Focused check on the galaxy renderer: build a small universe, run it to the
// present day, find haloes, populate them, and photograph the result from
// several distances so both the point pass and the volumetric pass are hit.

import { serve, launch } from './harness.mjs';
import fs from 'node:fs';
import path from 'node:path';

const OUT = process.argv[2] || '/tmp/opus-gal';
fs.mkdirSync(OUT, { recursive: true });
const { server, port } = await serve();
const { browser, page } = await launch({ width: 1100, height: 640 });
let failures = 0;
const check = (n, c, d = '') => { console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${d ? '   ' + d : ''}`); if (!c) failures++; };

try {
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`http://127.0.0.1:${port}/index.html?preserve=1&maxpixels=300000&seed=20240819&grid=32&steps=90`, { waitUntil: 'load' });
  await page.waitForFunction(() => document.getElementById('loader').classList.contains('done'), { timeout: 180000 });

  await page.evaluate(() => new Promise((res) => {
    const pm = window.__opus.universe.pm;
    const tick = () => { let n = 0; while (!pm.finished && n < 30 && pm.step()) n++; if (pm.finished) res(); else requestAnimationFrame(tick); };
    tick();
  }));
  await page.waitForTimeout(300);

  const info = await page.evaluate(() => {
    window.__opus.refreshGalaxies();
    const hfDiag = window.__opus.haloDiag ? window.__opus.haloDiag() : null;
    const s = window.__opus.state.galaxyStats;
    const halos = window.__opus.halos;
    return {
      halos: halos.length, galaxies: s.count, ms: s.ms,
      contour: window.__opus.__hf ? 0 : 0,
      spirals: s.spirals, ellipticals: s.ellipticals, irregulars: s.irregulars,
      topHalo: halos.length ? { mass: halos[0].mass, rVir: halos[0].rVir, spin: halos[0].spin, np: halos[0].nParticles } : null,
      brightest: s.brightest ? {
        mStar: s.brightest.mStar, type: s.brightest.type, radius: s.brightest.radius,
        pos: s.brightest.pos, arms: s.brightest.armCount,
      } : null,
      uploaded: window.__opus.galaxyRenderer.count,
      diag: hfDiag,
    };
  });
  if (info.diag) console.log(`  particle mass ${info.diag.particleMass.toExponential(2)} Msun/h; smallest resolved halo ${info.diag.minResolvedMass.toExponential(2)}; linking length ${info.diag.linkingLengthMpc.toFixed(3)} Mpc/h`);
  console.log(`  ${info.halos} haloes -> ${info.galaxies} galaxies in ${info.ms.toFixed(0)} ms`);
  console.log(`  ${info.spirals} spiral, ${info.ellipticals} elliptical, ${info.irregulars} irregular`);
  if (info.topHalo) console.log(`  most massive halo: ${info.topHalo.mass.toExponential(2)} Msun/h, R_vir ${info.topHalo.rVir.toFixed(2)} Mpc/h, spin ${info.topHalo.spin.toFixed(3)}, ${info.topHalo.np} particles`);
  if (info.brightest) console.log(`  brightest galaxy: logM* ${Math.log10(info.brightest.mStar).toFixed(2)}, type ${info.brightest.type}, R ${(info.brightest.radius * 1000).toFixed(1)} kpc/h, ${info.brightest.arms} arms`);

  check('haloes found', info.halos > 5, `${info.halos}`);
  check('galaxies populated and uploaded', info.galaxies > 10 && info.uploaded === info.galaxies);
  check('no page errors', errors.length === 0, errors.join(' | '));

  const shots = [];
  for (const factor of [400, 60, 9, 2.4]) {
    await page.evaluate((f) => {
      const o = window.__opus;
      const g = o.state.galaxyStats.brightest;
      const d = g.radius * f;
      const cam = o.camera;
      cam.setPose(new Float32Array([g.pos[0] + d * 0.6, g.pos[1] + d * 0.45, g.pos[2] + d * 0.66]), cam.orientation);
      cam.lookAt(new Float32Array(g.pos), new Float32Array([0, 1, 0]));
      cam.near = Math.max(1e-5, d * 0.001);
      for (let i = 0; i < 90; i++) cam.update(1 / 60, 1);
    }, factor);
    await page.waitForTimeout(500);
    const name = `galaxy-${String(factor).padStart(4, '0')}R`;
    await page.screenshot({ path: path.join(OUT, `${name}.png`), timeout: 120000 });
    const lum = await page.evaluate(() => window.__opus.measureFrame());
    shots.push({ factor, ...lum });
    console.log(`  ${name}: mean luma ${lum.meanLuma.toFixed(2)}, lit ${(lum.litFraction * 100).toFixed(1)}%, peak ${lum.maxLuma.toFixed(0)}`);
  }
  check('galaxy is visible from close range', shots[shots.length - 1].maxLuma > 40,
    `peak luma ${shots[shots.length - 1].maxLuma.toFixed(0)}`);
  check('approaching a galaxy makes it brighter', shots[3].meanLuma > shots[0].meanLuma,
    `${shots[0].meanLuma.toFixed(2)} -> ${shots[3].meanLuma.toFixed(2)}`);
  check('no page errors after rendering', errors.length === 0, errors.join(' | '));
  console.log(`\n  screenshots in ${OUT}`);
} catch (e) {
  console.error('galaxy test error:', e.message, e.stack); failures++;
} finally {
  await browser.close(); server.close();
}
process.exit(failures ? 1 : 0);

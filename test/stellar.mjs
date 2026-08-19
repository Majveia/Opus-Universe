// Exercises the descent from the cosmic web into a star system, and photographs
// the system, a planet, and a gas giant with rings.

import { serve, launch } from './harness.mjs';
import fs from 'node:fs';
import path from 'node:path';

const OUT = process.argv[2] || '/tmp/opus-stellar';
fs.mkdirSync(OUT, { recursive: true });
const { server, port } = await serve();
const { browser, page } = await launch({ width: 1100, height: 640 });
let failures = 0;
const check = (n, c, d = '') => { console.log(`  ${c ? 'PASS' : 'FAIL'}  ${n}${d ? '   ' + d : ''}`); if (!c) failures++; };
const shot = async (name) => {
  await page.screenshot({ path: path.join(OUT, `${name}.png`), timeout: 120000 });
  console.log(`  captured ${name}.png`);
};

try {
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`http://127.0.0.1:${port}/index.html?preserve=1&maxpixels=300000&seed=20240819&grid=32&steps=80`, { waitUntil: 'load' });
  await page.waitForFunction(() => document.getElementById('loader').classList.contains('done'), { timeout: 180000 });

  await page.evaluate(() => new Promise((res) => {
    const pm = window.__opus.universe.pm;
    const tick = () => { let n = 0; while (!pm.finished && n < 30 && pm.step()) n++; if (pm.finished) res(); else requestAnimationFrame(tick); };
    tick();
  }));
  await page.evaluate(() => window.__opus.refreshGalaxies());
  await page.waitForTimeout(200);

  // Search the population for a system worth photographing: one with a ringed
  // giant and, if possible, a habitable world.
  const found = await page.evaluate(() => {
    const o = window.__opus;
    const pop = o.galaxyRenderer.lastPopulation;
    if (!pop || !pop.length) return null;
    let best = null;
    for (let i = 0; i < Math.min(pop.length, 400); i++) {
      const g = pop[i];
      o.descendToSystem(g);
      const s = o.stellar.system;
      const ringed = s.planets.filter((p) => p.hasRings).length;
      const hab = s.habitableCount;
      const score = ringed * 2 + hab * 3 + s.planets.length * 0.2 + (s.companion ? 1 : 0);
      if (!best || score > best.score) {
        best = {
          score, index: i, name: s.name, planets: s.planets.length, ringed, hab,
          starClass: s.star.class, starMass: s.star.mass, binary: !!s.companion,
          types: s.planets.map((p) => p.typeName),
          skyStars: o.starfield.count,
        };
      }
      if (score > 8) break;
    }
    if (best) o.descendToSystem(pop[best.index]);
    return best;
  });

  if (!found) { check('a system could be entered', false, 'no galaxies in the population'); }
  else {
    console.log(`  entered ${found.name}: ${found.starClass}-type ${found.starMass.toFixed(2)} Msun${found.binary ? ' binary' : ''}`);
    console.log(`  ${found.planets} planets (${found.ringed} ringed, ${found.hab} habitable): ${found.types.join(', ')}`);
    console.log(`  night sky: ${found.skyStars} stars generated from the host galaxy`);
    check('descended into a star system', found.planets > 0);
    check('night sky generated', found.skyStars > 500, `${found.skyStars} stars`);
  }

  await page.evaluate(() => { window.__opus.state.paused = true; });
  await page.waitForTimeout(600);
  const wide = await page.evaluate(() => window.__opus.measureFrame());
  await shot('system-wide');
  console.log(`  system view: mean luma ${wide.meanLuma.toFixed(2)}, peak ${wide.maxLuma.toFixed(0)}`);
  check('the system renders', wide.maxLuma > 30 && wide.meanLuma > 0.2,
    `mean ${wide.meanLuma.toFixed(2)}, peak ${wide.maxLuma.toFixed(0)}`);

  // Frame each interesting body in turn.
  const bodies = await page.evaluate(() => window.__opus.stellar.placed.map((b, i) => ({
    i, kind: b.kind, label: b.label,
    type: b.ref.type, rings: !!b.ref.hasRings, habitable: !!b.ref.habitable,
  })));
  const picks = [];
  const giant = bodies.find((b) => b.rings) || bodies.find((b) => b.type >= 6);
  const hab = bodies.find((b) => b.habitable);
  const anyPlanet = bodies.find((b) => b.kind === 'planet');
  if (giant) picks.push(['ringed-giant', giant.i]);
  if (hab) picks.push(['habitable-world', hab.i]);
  if (!hab && anyPlanet) picks.push(['planet', anyPlanet.i]);

  for (const [name, idx] of picks) {
    const m = await page.evaluate((i) => {
      const o = window.__opus;
      o.state.targetBodyIndex = i;
      o.frameBody(o.stellar.placed[i]);
      for (let k = 0; k < 120; k++) o.camera.update(1 / 60, 1.7);
      return null;
    }, idx);
    await page.waitForTimeout(700);
    await shot(name);
    const lum = await page.evaluate(() => window.__opus.measureFrame());
    console.log(`  ${name}: mean luma ${lum.meanLuma.toFixed(2)}, lit ${(lum.litFraction * 100).toFixed(1)}%, peak ${lum.maxLuma.toFixed(0)}`);
    check(`${name} is visible`, lum.maxLuma > 25, `peak ${lum.maxLuma.toFixed(0)}`);
  }

  check('no page errors throughout', errors.length === 0, errors.slice(0, 3).join(' | '));
  console.log(`\n  screenshots in ${OUT}`);
} catch (e) {
  console.error('stellar test error:', e.message, e.stack); failures++;
} finally {
  await browser.close(); server.close();
}
process.exit(failures ? 1 : 0);

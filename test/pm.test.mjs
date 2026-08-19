// Validates the Particle-Mesh solver by asking whether the simulated universe
// grows structure at the rate general relativity says it should, and whether
// that agreement improves as the force mesh is refined.

import { serve, launch, runModule } from './harness.mjs';

const { server, port } = await serve();
const origin = `http://127.0.0.1:${port}`;
const { browser, page } = await launch();

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? '\n          ' + detail : ''}`);
  if (!cond) failures++;
};

const report = (r, title) => {
  console.log(`\n  ${title}`);
  console.log(`  ${r.particleCount} particles, ${r.N}^3 mesh (${(r.L / r.N).toFixed(2)} Mpc/h cells), ${r.steps} steps, ${(r.elapsedMs / 1000).toFixed(1)} s`);
  console.log('     k [h/Mpc]  modes    amplitude growth   / linear    correlation');
  for (const g of r.growth) {
    console.log(`     ${g.k.toFixed(4).padStart(8)} ${String(g.modes).padStart(6)}   ${g.autoRatio.toFixed(2).padStart(12)}   ${(g.autoRatio / g.expected).toFixed(3).padStart(8)}    ${g.correlation.toFixed(3)}`);
  }
};

try {
  const args = { L: 200, steps: 120, kBins: [0.03, 0.06, 0.10, 0.16, 0.25, 0.40] };
  const lo = await runModule(page, origin, 'test/pm.page.js', { ...args, N: 32 });
  console.log(`Particle-Mesh solver   (${lo.renderer})`);
  console.log(`  linear theory predicts amplitude growth D(1)/D(z=49) = ${lo.expectedGrowth.toFixed(2)}`);
  report(lo, '32^3 force mesh');

  const hi = await runModule(page, origin, 'test/pm.page.js', { ...args, N: 64 });
  report(hi, '64^3 force mesh');

  // The largest scales in the box are linear at z=0 and must reproduce D(a).
  const linLo = lo.growth[0], linHi = hi.growth[0];
  check('largest scales grow at the linear rate (32^3)',
    Math.abs(linLo.autoRatio / linLo.expected - 1) < 0.12,
    `k=${linLo.k.toFixed(3)}: ${(linLo.autoRatio / linLo.expected).toFixed(3)} of linear, correlation ${linLo.correlation.toFixed(3)}`);
  check('largest scales grow at the linear rate (64^3)',
    Math.abs(linHi.autoRatio / linHi.expected - 1) < 0.08,
    `k=${linHi.k.toFixed(3)}: ${(linHi.autoRatio / linHi.expected).toFixed(3)} of linear, correlation ${linHi.correlation.toFixed(3)}`);

  // Refining the mesh must move every scale toward or past linear growth: a
  // coarse mesh can only under-resolve the force, never over-resolve it.
  let improved = 0, compared = 0;
  console.log('\n  convergence with mesh refinement');
  console.log('     k [h/Mpc]     32^3      64^3');
  for (let i = 0; i < Math.min(lo.growth.length, hi.growth.length); i++) {
    const a = lo.growth[i].autoRatio / lo.growth[i].expected;
    const b = hi.growth[i].autoRatio / hi.growth[i].expected;
    console.log(`     ${lo.growth[i].k.toFixed(4).padStart(8)}   ${a.toFixed(3).padStart(7)}   ${b.toFixed(3).padStart(7)}   ${b > a ? '+' : ''}${((b / a - 1) * 100).toFixed(1)}%`);
    compared++;
    if (b >= a - 0.02) improved++;
  }
  check('refining the mesh recovers more power at every scale', improved === compared,
    `${improved}/${compared} bins improved or held`);

  // Small scales must go nonlinear: that is the entire point.
  const small = hi.growth[hi.growth.length - 1];
  check('small scales exceed linear growth (nonlinear collapse)',
    small.autoRatio / small.expected > 1.2,
    `k=${small.k.toFixed(3)}: ${(small.autoRatio / small.expected).toFixed(2)}x linear, correlation ${small.correlation.toFixed(3)} (mode coupling)`);

  check('no net momentum acquired over the whole run', hi.drift < 1e-3,
    `|<p>| / p_rms = ${hi.drift.toExponential(2)}`);

  console.log(`\n  64^3 density field: sigma ${hi.initial.sigma.toExponential(2)} -> ${hi.final.sigma.toFixed(3)}`);
  console.log(`                      skew  ${hi.initial.skew.toFixed(2)} -> ${hi.final.skew.toFixed(2)}`);
  console.log(`                      peak  ${hi.initial.max.toFixed(2)} -> ${hi.final.max.toFixed(0)}\n`);
  check('field becomes strongly non-Gaussian (voids empty, knots collapse)',
    hi.final.skew > 3 && hi.final.max > 50,
    `skewness ${hi.final.skew.toFixed(2)}, peak overdensity ${hi.final.max.toFixed(0)}`);
} catch (e) {
  console.error('harness error:', e.message, e.stack);
  failures++;
} finally {
  await browser.close();
  server.close();
}
console.log(failures === 0 ? '\nParticle-Mesh solver validated.' : `\n${failures} check(s) FAILED.`);
process.exit(failures ? 1 : 0);

import { serve, launch, runModule } from './harness.mjs';

const { server, port } = await serve();
const origin = `http://127.0.0.1:${port}`;
const { browser, page } = await launch();

let failures = 0;
try {
  const { renderer, results } = await runModule(page, origin, 'test/gpufft.page.js', { sizes: [8, 16, 32] });
  console.log(`GPU FFT vs CPU reference   (${renderer})\n`);
  console.log('     N    atlas        max abs err    rel err      round-trip err');
  for (const r of results) {
    const ok = r.relErr < 2e-5 && r.rtErr < 2e-5;
    if (!ok) failures++;
    console.log(`  ${ok ? 'PASS' : 'FAIL'} ${String(r.N).padStart(3)}  ${r.atlas.padEnd(11)} ${r.maxErr.toExponential(3)}     ${r.relErr.toExponential(3)}    ${r.rtErr.toExponential(3)}`);
  }
} catch (e) {
  console.error('harness error:', e.message, e.stack);
  failures++;
} finally {
  await browser.close();
  server.close();
}
console.log(failures === 0 ? '\nGPU FFT matches the CPU reference.' : `\n${failures} GPU FFT check(s) FAILED.`);
process.exit(failures ? 1 : 0);

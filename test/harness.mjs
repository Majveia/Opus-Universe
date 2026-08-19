// Serves the repository over HTTP and runs a page in headless Chromium with
// SwiftShader, so WebGL2 code can be tested from the command line.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '/opt/node22/lib/node_modules/playwright/index.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.png': 'image/png',
};

export function serve(root = ROOT) {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0]);
    let file = path.join(root, url === '/' ? '/index.html' : url);
    if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
    fs.readFile(file, (err, data) => {
      if (err) { res.writeHead(404).end('not found: ' + url); return; }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

export async function launch({ width = 1280, height = 720, verbose = true } = {}) {
  const browser = await chromium.launch({
    args: [
      '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader',
      '--ignore-gpu-blocklist', '--disable-dev-shm-usage',
    ],
  });
  const page = await browser.newPage({ viewport: { width, height } });
  if (verbose) {
    page.on('console', (m) => {
      const t = m.text();
      if (!t.startsWith('__')) console.log(`    [page:${m.type()}] ${t}`);
    });
    page.on('pageerror', (e) => console.log(`    [page:error] ${e.message}\n${e.stack || ''}`));
  }
  return { browser, page };
}

// Runs a module in the page and returns whatever its default export resolves to.
export async function runModule(page, origin, modulePath, args = {}) {
  await page.goto(`${origin}/test/blank.html`);
  return page.evaluate(async ({ mod, args }) => {
    const m = await import(mod);
    return await m.default(args);
  }, { mod: `${origin}/${modulePath}`, args });
}

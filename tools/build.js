#!/usr/bin/env node
// A small dependency-free bundler.
//
// The project is written as plain ES modules so it can be served and debugged
// with no build step at all. This produces the other artefact: one
// self-contained HTML file with every module inlined, which can be opened from
// a file:// URL, emailed, or published as a single page.
//
// Each module becomes an immediately-invoked function returning its exports,
// and each import becomes a destructuring assignment from that value. Modules
// are emitted in dependency order.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENTRY_HTML = path.join(ROOT, 'index.html');
const OUT_DIR = path.join(ROOT, 'dist');

const modules = new Map();     // absolute path -> { id, source, deps, exports }
let nextId = 0;

function resolveImport(fromFile, spec) {
  if (!spec.startsWith('.')) throw new Error(`bare import "${spec}" in ${fromFile} - this project has no dependencies`);
  return path.resolve(path.dirname(fromFile), spec);
}

const IMPORT_RE = /^\s*import\s+(?:([\w$]+)\s*,\s*)?(?:\{([^}]*)\}|\*\s*as\s+([\w$]+)|([\w$]+))?\s*from\s*['"]([^'"]+)['"]\s*;?\s*$/gm;
const SIDE_EFFECT_IMPORT_RE = /^\s*import\s*['"]([^'"]+)['"]\s*;?\s*$/gm;

function load(file) {
  if (modules.has(file)) return modules.get(file);
  const source = fs.readFileSync(file, 'utf8');
  const mod = { file, id: `__m${nextId++}`, source, deps: [], exports: new Set(), body: '' };
  modules.set(file, mod);

  let body = source;

  // --- rewrite imports -------------------------------------------------
  body = body.replace(SIDE_EFFECT_IMPORT_RE, (m, spec) => {
    const dep = load(resolveImport(file, spec));
    mod.deps.push(dep.file);
    return `/* import ${spec} */`;
  });

  body = body.replace(IMPORT_RE, (m, defaultA, named, star, defaultB, spec) => {
    const dep = load(resolveImport(file, spec));
    mod.deps.push(dep.file);
    const parts = [];
    const def = defaultA || defaultB;
    if (def) parts.push(`const ${def} = ${dep.id}.default;`);
    if (star) parts.push(`const ${star} = ${dep.id};`);
    if (named) {
      const spec2 = named.split(',').map((s) => s.trim()).filter(Boolean).map((s) => {
        const as = s.split(/\s+as\s+/);
        return as.length === 2 ? `${as[0].trim()}: ${as[1].trim()}` : s;
      }).join(', ');
      if (spec2) parts.push(`const { ${spec2} } = ${dep.id};`);
    }
    return parts.join(' ');
  });

  // --- collect and strip exports ---------------------------------------
  body = body.replace(/^\s*export\s+(async\s+)?function\s+([\w$]+)/gm, (m, asy, name) => {
    mod.exports.add(name);
    return `${asy || ''}function ${name}`;
  });
  body = body.replace(/^\s*export\s+class\s+([\w$]+)/gm, (m, name) => {
    mod.exports.add(name);
    return `class ${name}`;
  });
  body = body.replace(/^\s*export\s+(const|let|var)\s+([\w$]+)/gm, (m, kind, name) => {
    mod.exports.add(name);
    return `${kind} ${name}`;
  });
  body = body.replace(/^\s*export\s*\{([^}]*)\}\s*;?\s*$/gm, (m, list) => {
    for (const item of list.split(',')) {
      const t = item.trim();
      if (!t) continue;
      const as = t.split(/\s+as\s+/);
      mod.exports.add((as[1] || as[0]).trim());
    }
    return '';
  });
  body = body.replace(/^\s*export\s+default\s+/gm, () => {
    mod.exports.add('default');
    return 'const __default__ = ';
  });

  mod.body = body;
  return mod;
}

function emit(entryFile) {
  load(entryFile);

  // Depth-first ordering, so a module is defined before anything that uses it.
  const ordered = [];
  const state = new Map();
  const visit = (file) => {
    const s = state.get(file);
    if (s === 'done') return;
    if (s === 'visiting') throw new Error(`circular import involving ${path.relative(ROOT, file)}`);
    state.set(file, 'visiting');
    for (const d of modules.get(file).deps) visit(d);
    state.set(file, 'done');
    ordered.push(file);
  };
  visit(entryFile);

  const chunks = [];
  for (const file of ordered) {
    const m = modules.get(file);
    const exportList = [...m.exports].map((e) => (e === 'default' ? 'default: __default__' : e)).join(', ');
    chunks.push(
      `/* ${path.relative(ROOT, file)} */\n` +
      `const ${m.id} = (() => {\n${m.body}\nreturn { ${exportList} };\n})();\n`,
    );
  }
  return { code: chunks.join('\n'), entryId: modules.get(entryFile).id, count: ordered.length };
}

/* ------------------------------------------------------------------ main -- */

const entry = path.join(ROOT, 'src', 'app.js');
const { code, entryId, count } = emit(entry);

let html = fs.readFileSync(ENTRY_HTML, 'utf8');
const scriptBlock = /<script type="module">[\s\S]*?<\/script>/;
if (!scriptBlock.test(html)) throw new Error('could not find the module script block in index.html');

const runtime = `<script>
(function () {
${code}
try {
  ${entryId}.main().catch(reportFailure);
} catch (e) { reportFailure(e); }
function reportFailure(e) {
  document.getElementById('loader').classList.add('done');
  document.getElementById('error').style.display = 'flex';
  document.getElementById('errorText').textContent = (e && e.stack) ? e.stack : String(e);
  console.error(e);
}
})();
</script>`;

html = html.replace(scriptBlock, runtime);
// The bundle has no module scope, so the entry point cannot be a module either.
html = html.replace('<script type="module">', '<script>');

fs.mkdirSync(OUT_DIR, { recursive: true });
const out = path.join(OUT_DIR, 'opus-universe.html');
fs.writeFileSync(out, html);

const kb = (fs.statSync(out).size / 1024).toFixed(0);
console.log(`bundled ${count} modules -> ${path.relative(ROOT, out)} (${kb} kB, no external requests)`);

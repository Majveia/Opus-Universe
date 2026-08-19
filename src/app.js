// Opus Universe - application shell.
//
// Builds a universe from cosmological initial conditions, evolves it with the
// GPU Particle-Mesh solver, and renders it. The camera flies through a
// periodic volume tiled without end, so there is no wall to hit.

import { GLContext } from './core/gl.js';
import { Camera } from './core/camera.js';
import { Input } from './core/input.js';
import { PostChain } from './render/passes.js';
import { CosmicWebRenderer } from './render/cosmicweb.js';
import { Cosmology, PRESETS } from './cosmology/cosmology.js';
import { PowerSpectrum } from './cosmology/powerspectrum.js';
import { InitialConditions } from './cosmology/ics.js';
import { ParticleMesh } from './sim/pm.js';
import { v3, v3set, quat, clamp, mix, damp, DEG } from './core/math.js';
import { Hud } from './ui/hud.js';

const $ = (id) => document.getElementById(id);
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

/* -------------------------------------------------------------- quality -- */

// Chooses a grid resolution the device can actually sustain. The cost of a step
// scales as N^3 log N, so guessing wrong is expensive in both directions.
function pickQuality(ctx) {
  const r = (ctx.rendererName || '').toLowerCase();
  const software = r.includes('swiftshader') || r.includes('llvmpipe') || r.includes('software');
  const mem = navigator.deviceMemory || 4;
  const mobile = /android|iphone|ipad|mobile/i.test(navigator.userAgent);
  if (software) return { grid: 32, steps: 200, label: 'software' };
  if (mobile || mem <= 4) return { grid: 64, steps: 320, label: 'mobile' };
  if (ctx.limits.maxTextureSize >= 8192 && mem >= 8) return { grid: 128, steps: 420, label: 'high' };
  return { grid: 64, steps: 380, label: 'standard' };
}

/* ------------------------------------------------------------ the world -- */

class Universe {
  constructor(ctx, opts) {
    this.ctx = ctx;
    this.opts = opts;
    this.pm = null;
    this.cosmo = null;
    this.ps = null;
    this.stats = null;
  }

  async build({ params, gridSize, boxSize, seed, steps, aInit = 1 / 50, onProgress }) {
    this.cosmo = new Cosmology(params);
    this.ps = new PowerSpectrum(this.cosmo);

    const ic = new InitialConditions({
      cosmology: this.cosmo, powerSpectrum: this.ps,
      gridSize, boxSize, seed, aInit, use2LPT: true,
    });
    const data = await ic.generate(async (f, label) => {
      onProgress(f * 0.92, label);
      await nextFrame();
    });

    onProgress(0.95, 'uploading to the graphics device');
    await nextFrame();

    if (this.pm) this.pm.dispose();
    this.pm = new ParticleMesh(this.ctx, {
      cosmology: this.cosmo, gridSize, boxSize,
      positions: data.positions, velocities: data.velocities,
      aInit, aMax: 1.0, steps,
    });
    this.stats = data.stats;
    this.boxSize = boxSize;
    this.seed = seed;

    onProgress(1, 'ready');
    return this.stats;
  }

  get a() { return this.pm ? this.pm.a : 1; }
  get redshift() { return 1 / this.a - 1; }
  get ageGyr() { return this.cosmo.ageGyr(this.a); }
}

/* ------------------------------------------------------------- vantages -- */

// Named camera placements, in units of the box size, so they work at any scale.
const VANTAGES = {
  1: { name: 'the whole volume', dist: 1.75, inside: false },
  2: { name: 'above the web', dist: 0.62, inside: false },
  3: { name: 'inside a filament', dist: 0.0, inside: true },
  4: { name: 'deep field', dist: 3.1, inside: false },
};

/* ----------------------------------------------------------------- main -- */

// URL overrides, so a specific universe can be linked to and so the automated
// tests can pin resolution and read the framebuffer back.
function readOverrides() {
  const q = new URLSearchParams(location.search);
  const num = (k, d) => (q.has(k) ? parseFloat(q.get(k)) : d);
  return {
    seed: q.has('seed') ? (parseInt(q.get('seed'), 10) >>> 0) : null,
    grid: q.has('grid') ? Math.max(16, parseInt(q.get('grid'), 10)) : null,
    box: q.has('box') ? num('box') : null,
    steps: q.has('steps') ? num('steps') : null,
    maxPixels: q.has('maxpixels') ? num('maxpixels') : null,
    preserve: q.get('preserve') === '1',
    autoplay: q.get('autoplay') !== '0',
  };
}

export async function main() {
  const canvas = $('view');
  const overrides = readOverrides();
  const ctx = new GLContext(canvas, { preserveDrawingBuffer: overrides.preserve });
  const gl = ctx.gl;

  const loader = $('loader'), loadBar = $('loadBar'), loadPhase = $('loadPhase'), loadPct = $('loadPct');
  const setLoad = (f, label) => {
    loadBar.style.width = `${(f * 100).toFixed(1)}%`;
    loadPct.textContent = `${Math.round(f * 100)}%`;
    if (label) loadPhase.textContent = label;
  };

  const quality = pickQuality(ctx);
  if (overrides.grid) quality.grid = overrides.grid;
  if (overrides.steps) quality.steps = overrides.steps;
  const state = {
    boxSize: overrides.box || 150,
    seed: overrides.seed ?? ((Math.random() * 1e9) | 0),
    gridSize: quality.grid,
    steps: quality.steps,
    params: { ...PRESETS.planck18 },
    presetName: 'planck18',
    stepsPerSecond: 60,
    baseExposure: 1.4,
    paused: false,
    hudVisible: true,
    building: false,
    quality,
  };

  const universe = new Universe(ctx);
  const camera = new Camera({ fov: 62, near: 0.02, speed: 6 });
  const input = new Input(canvas);
  const web = new CosmicWebRenderer(ctx);
  let post = null;
  const hud = new Hud();

  /* ------------------------------------------------------------- sizing -- */
  let dpr = 1, vw = 1, vh = 1;
  function resize() {
    const maxPixels = overrides.maxPixels || (quality.label === 'software' ? 900 * 506 : 2_600_000);
    const cssW = canvas.clientWidth || window.innerWidth;
    const cssH = canvas.clientHeight || window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    let w = Math.max(1, Math.round(cssW * dpr));
    let h = Math.max(1, Math.round(cssH * dpr));
    const scale = Math.sqrt(maxPixels / (w * h));
    if (scale < 1) { w = Math.round(w * scale); h = Math.round(h * scale); }
    if (w === vw && h === vh) return;
    vw = w; vh = h;
    canvas.width = w; canvas.height = h;
    if (post) post.dispose();
    post = new PostChain(ctx, w, h);
    syncRenderSettings();
  }
  window.addEventListener('resize', resize);

  /* --------------------------------------------------------- build flow -- */
  async function rebuild(message = 'assembling the universe') {
    if (state.building) return;
    state.building = true;
    loader.classList.remove('done');
    setLoad(0, message);
    await nextFrame();
    try {
      await universe.build({
        params: state.params,
        gridSize: state.gridSize,
        boxSize: state.boxSize,
        seed: state.seed,
        steps: state.steps,
        onProgress: setLoad,
      });
      placeCamera(2);
      await nextFrame();
      loader.classList.add('done');
      $('hud').classList.add('on');
    } finally {
      state.building = false;
    }
  }

  function placeCamera(key) {
    const v = VANTAGES[key] || VANTAGES[2];
    const L = state.boxSize;
    const c = L * 0.5;
    if (v.inside) {
      camera.setPose(v3(c * 0.72, c * 0.9, c * 1.1), quat());
      camera.lookAt(v3(c, c, c), v3(0, 1, 0));
      camera.speed = L * 0.03;
    } else {
      const d = L * v.dist;
      camera.setPose(v3(c + d * 0.62, c + d * 0.44, c + d * 0.72), quat());
      camera.lookAt(v3(c, c, c), v3(0, 1, 0));
      camera.speed = L * 0.04 * Math.max(0.3, v.dist);
    }
    camera.update(1 / 60, vw / vh);
    hud.flash(v.name);
  }

  /* ------------------------------------------------------------ controls */
  function syncRenderSettings() {
    if (!post) return;
    state.baseExposure = parseFloat($('sExp').value);
    post.settings.bloomStrength = parseFloat($('sBloom').value);
    web.settings.kernelScale = parseFloat($('sSize').value);
    web.settings.brightness = parseFloat($('sBright').value);
    $('vExp').textContent = state.baseExposure.toFixed(2);
    $('vBloom').textContent = post.settings.bloomStrength.toFixed(2);
    $('vSize').textContent = web.settings.kernelScale.toFixed(2);
    $('vBright').textContent = web.settings.brightness.toFixed(3);
  }
  ['sExp', 'sBloom', 'sSize', 'sBright'].forEach((id) => $(id).addEventListener('input', syncRenderSettings));

  $('sOm').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    state.params = { ...state.params, omegaM: v, omegaLambda: 1 - v };
    $('vOm').textContent = v.toFixed(3);
  });
  $('sS8').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    state.params = { ...state.params, sigma8: v };
    $('vS8').textContent = v.toFixed(3);
  });
  $('sBox').addEventListener('input', (e) => {
    state.boxSize = parseFloat(e.target.value);
    $('vBox').textContent = state.boxSize.toFixed(0);
  });
  $('sRate').addEventListener('input', (e) => {
    state.stepsPerSecond = parseFloat(e.target.value);
    $('vRate').textContent = state.stepsPerSecond.toFixed(0);
  });
  $('preset').addEventListener('change', (e) => {
    const p = PRESETS[e.target.value];
    if (!p) return;
    state.presetName = e.target.value;
    state.params = { ...p };
    $('sOm').value = p.omegaM; $('vOm').textContent = p.omegaM.toFixed(3);
    $('sS8').value = p.sigma8; $('vS8').textContent = p.sigma8.toFixed(3);
  });
  $('bRegen').addEventListener('click', () => { state.seed = (Math.random() * 1e9) | 0; rebuild('rebuilding from new initial conditions'); });
  $('bPause').addEventListener('click', () => { state.paused = !state.paused; $('bPause').textContent = state.paused ? 'resume' : 'pause'; });
  $('bRestart').addEventListener('click', () => { if (universe.pm) { rebuild('replaying cosmic history'); } });

  input.onWorldClick(() => { if (!panelOpen) input.requestPointerLock(); });

  let panelOpen = false;
  const setPanel = (open) => {
    panelOpen = open;
    $('panel').classList.toggle('open', open);
    if (open) input.exitPointerLock();
  };

  addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    switch (e.code) {
      case 'Tab': e.preventDefault(); setPanel(!panelOpen); break;
      case 'KeyH': state.hudVisible = !state.hudVisible; $('hud').classList.toggle('hidden', !state.hudVisible); break;
      case 'KeyP': case 'Backslash':
        state.paused = !state.paused; $('bPause').textContent = state.paused ? 'resume' : 'pause';
        hud.flash(state.paused ? 'time paused' : 'time running'); break;
      case 'KeyR': state.seed = (Math.random() * 1e9) | 0; rebuild('rebuilding from new initial conditions'); break;
      case 'BracketLeft': state.stepsPerSecond = clamp(state.stepsPerSecond - 10, 0, 240); $('sRate').value = state.stepsPerSecond; $('vRate').textContent = state.stepsPerSecond; break;
      case 'BracketRight': state.stepsPerSecond = clamp(state.stepsPerSecond + 10, 0, 240); $('sRate').value = state.stepsPerSecond; $('vRate').textContent = state.stepsPerSecond; break;
      case 'Digit1': case 'Digit2': case 'Digit3': case 'Digit4':
        placeCamera(parseInt(e.code.slice(5), 10)); break;
      case 'Escape': setPanel(false); break;
    }
  });

  const hint = $('hint');
  const dismissHint = () => hint.classList.add('gone');
  canvas.addEventListener('mousedown', dismissHint, { once: true });
  canvas.addEventListener('touchstart', dismissHint, { once: true });
  setTimeout(dismissHint, 9000);
  $('keys').classList.add('on');

  /* --------------------------------------------------------- the loop --- */
  resize();
  await rebuild('assembling the universe');

  let last = performance.now();
  let stepAccumulator = 0;
  let fpsAvg = 60;

  function frame(now) {
    const dtRaw = (now - last) / 1000;
    last = now;
    const dt = Math.min(dtRaw, 0.1);
    fpsAvg = mix(fpsAvg, 1 / Math.max(dtRaw, 1e-4), 0.08);

    resize();
    camera.update(dt, vw / vh);
    if (!panelOpen) camera.fly(input, dt, { baseFov: 62 });
    $('reticle').classList.toggle('off', !input.pointerLocked);

    // Advance cosmic time. The simulation is decoupled from the frame rate:
    // a slow device runs the same history, just fewer steps per second.
    const pm = universe.pm;
    if (pm && !state.paused && state.stepsPerSecond > 0) {
      stepAccumulator += dt * state.stepsPerSecond;
      const budget = Math.min(stepAccumulator | 0, 8);
      for (let i = 0; i < budget; i++) { if (!pm.step()) break; }
      stepAccumulator -= (stepAccumulator | 0);
    }

    // --- draw ---
    post.sceneFramebuffer.bind();
    gl.clearColor(0, 0, 0, 1);
    gl.clearDepth(0);                       // reversed-Z: far plane is 0
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.depthFunc(gl.GREATER);

    if (pm) {
      // Analytic auto-exposure. The mean column density through the box is
      // fixed by mass conservation, but the peaks brighten as structure grows,
      // so the exposure is trimmed by the linear amplitude of the epoch. This
      // needs no framebuffer readback and never hunts.
      const sigma = universe.cosmo.sigma8 * universe.cosmo.growth(universe.a);
      post.settings.exposure = state.baseExposure / (1 + 0.42 * sigma);
      web.render(pm, camera, state.boxSize);
    }

    post.render(now / 1000, canvas.width, canvas.height);

    if (state.hudVisible) {
      hud.update({
        universe, camera, web, state, fps: fpsAvg,
        pointerLocked: input.pointerLocked,
      });
    }

    input.endFrame();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // Expose for debugging and for the automated visual tests.
  window.__opus = {
    ctx, camera, universe, web, state, placeCamera, input,
    get post() { return post; },
    // Renders one frame and reads it back before the compositor discards it.
    // Only meaningful when the context was created with preserveDrawingBuffer.
    measureFrame() {
      const w = canvas.width, h = canvas.height;
      const px = new Uint8Array(w * h * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
      let sum = 0, lit = 0, max = 0, hist = new Array(16).fill(0);
      for (let i = 0; i < px.length; i += 4) {
        const l = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
        sum += l; if (l > 8) lit++; if (l > max) max = l;
        hist[Math.min(15, l >> 4)]++;
      }
      const n = px.length / 4;
      return { meanLuma: sum / n, litFraction: lit / n, maxLuma: max, w, h, hist: hist.map((c) => c / n) };
    },
  };
}

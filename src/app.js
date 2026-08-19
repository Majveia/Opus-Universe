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
import { GalaxyRenderer } from './render/galaxies.js';
import { HaloFinder } from './universe/halos.js';
import { populate, packGalaxies, populationStats } from './universe/galaxies.js';
import { Cosmology, PRESETS } from './cosmology/cosmology.js';
import { PowerSpectrum } from './cosmology/powerspectrum.js';
import { InitialConditions } from './cosmology/ics.js';
import { ParticleMesh } from './sim/pm.js';
import { v3, v3set, quat, clamp, mix, damp, DEG } from './core/math.js';
import { Hud } from './ui/hud.js';
import { BodyRenderer } from './render/bodies.js';
import { Starfield } from './render/starfield.js';
import { StellarScene } from './render/stellarscene.js';
import { generateSystem } from './universe/system.js';
import { NebulaRenderer, generateNebulae } from './render/nebula.js';
import { GALAXY_TYPE } from './universe/galaxies.js';
import { hash3 } from './core/rng.js';

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
    showDarkMatter: true,
    showGalaxies: true,
    scale: 'cosmic',
    transition: 0,          // 0 = settled; counts down through a scale change
    targetGalaxy: null,
    targetBodyIndex: 0,
    orbitLock: false,
    yearsPerSecond: 0.01,
    lastGalaxyA: 0,
    galaxyStats: null,
    paused: false,
    hudVisible: true,
    building: false,
    quality,
  };

  const universe = new Universe(ctx);
  const camera = new Camera({ fov: 62, near: 0.02, speed: 6 });
  const input = new Input(canvas);
  const web = new CosmicWebRenderer(ctx);
  const galaxyRenderer = new GalaxyRenderer(ctx);
  const bodyRenderer = new BodyRenderer(ctx);
  const starfield = new Starfield(ctx);
  const nebulaRenderer = new NebulaRenderer(ctx);
  const stellar = new StellarScene(bodyRenderer, starfield, nebulaRenderer);
  let haloFinder = null;
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
      haloFinder = new HaloFinder(universe.pm, universe.cosmo);
      galaxyRenderer.setGalaxies(new Float32Array(0), 0);
      state.lastGalaxyA = 0;
      state.galaxyStats = null;
      placeCamera(2);
      await nextFrame();
      loader.classList.add('done');
      $('hud').classList.add('on');
    } finally {
      state.building = false;
    }
  }

  // Finds haloes in the current density field and populates them with galaxies.
  function refreshGalaxies() {
    if (!haloFinder || !universe.pm) return;
    const t0 = performance.now();
    const a = universe.a;
    // The density field the solver leaves behind is one step stale; refresh it
    // so haloes are found in the configuration actually being displayed.
    universe.pm.depositDensity();
    const halos = haloFinder.find({ a, linkingLength: 0.2, minParticles: 20, maxHalos: 2500 });
    const galaxies = populate(halos, universe.cosmo, {
      a, boxSize: state.boxSize, seed: state.seed, maxGalaxies: 6000,
    });
    galaxyRenderer.setGalaxies(packGalaxies(galaxies, state.boxSize), galaxies.length, galaxies);
    state.lastGalaxyA = a;
    state.galaxyStats = { ...populationStats(galaxies, state.boxSize), halos: halos.length, ms: performance.now() - t0 };
  }

  // The galaxy closest to the line of sight, weighted so that a bright galaxy
  // a little off-axis wins over a faint one dead centre.
  function pickGalaxy() {
    const gs = state.galaxyStats;
    if (!gs || !gs.count) return null;
    const list = galaxyRenderer.lastPopulation;
    if (!list || !list.length) return null;
    const o = camera.position, f = camera.forward;
    let best = null, bestScore = -Infinity;
    for (const g of list) {
      const dx = g.pos[0] - o[0], dy = g.pos[1] - o[1], dz = g.pos[2] - o[2];
      const dist = Math.hypot(dx, dy, dz);
      if (dist < 1e-6) continue;
      const cosang = (dx * f[0] + dy * f[1] + dz * f[2]) / dist;
      if (cosang < 0.55) continue;                       // behind or far off-axis
      const score = Math.log10(Math.max(1, g.luminosity)) + 12 * cosang - Math.log10(dist + 1);
      if (score > bestScore) { bestScore = score; best = g; }
    }
    return best;
  }

  // Enters a star system inside the given galaxy. The system's seed is derived
  // from the galaxy's own position, so the same galaxy always contains the same
  // stars however many times it is visited.
  function descendToSystem(galaxy) {
    const q = galaxy.pos.map((v) => Math.round(v * 1e4));
    const seed = (hash3(q[0], q[1], q[2]) ^ state.seed ^ (state.systemNonce | 0)) >>> 0;
    const system = generateSystem(seed);
    stellar.setSystem(system);

    // The night sky comes from the host galaxy: its disk scale length, how
    // bulge-dominated it is, and where in the disk this system happens to sit.
    const diskKpc = galaxy.radius * 1000;
    const rng = ((seed >>> 8) % 1000) / 1000;
    starfield.build({
      seed: seed ^ 0x9e3779b9,
      count: quality.label === 'software' ? 12000 : 55000,
      diskScaleKpc: Math.max(0.6, diskKpc),
      scaleHeightKpc: Math.max(0.08, diskKpc * 0.12),
      observerRadiusKpc: Math.max(0.4, diskKpc * (0.8 + 2.2 * rng)),
      bulgeFraction: galaxy.type === GALAXY_TYPE.ELLIPTICAL ? 0.85 : galaxy.bulgeFraction * 0.6,
      bulgeScaleKpc: Math.max(0.2, diskKpc * 0.3),
      galaxyAgeGyr: 10,
      dustOpacityPerKpc: galaxy.type === GALAXY_TYPE.ELLIPTICAL ? 0.02 : 0.22 * (0.5 + galaxy.young),
    });

    // Nebulae belong to the host galaxy: a star-forming spiral is full of HII
    // regions, an elliptical has almost no cold gas and so almost none.
    nebulaRenderer.set(generateNebulae(seed ^ 0x51ed270b, {
      young: galaxy.young,
      elliptical: galaxy.type === GALAXY_TYPE.ELLIPTICAL,
      count: quality.label === 'software' ? 4 : null,
    }));

    state.scale = 'stellar';
    state.targetBodyIndex = 0;
    stellar.layout();
    // Arrive looking back at the star from a little beyond the outermost world.
    const far = system.planets.length ? system.planets[system.planets.length - 1].semiMajorAU : 4;
    const d = Math.max(2.5, far * 1.5);
    camera.setPose(v3(d * 0.55, d * 0.42, d * 0.72), quat());
    camera.lookAt(v3(0, 0, 0), v3(0, 1, 0));
    camera.speed = d * 0.06;
    camera.near = 1e-6;
    camera.update(1 / 60, vw / vh);
    hud.flash(system.name);
  }

  function ascendToCosmic() {
    state.scale = 'cosmic';
    state.orbitLock = false;
    camera.mode = 'free';
    placeCamera(2);
  }

  // Moves the camera to a comfortable viewing distance from a body.
  function frameBody(entry) {
    if (!entry) return;
    // Lock on. A close-in planet completes an orbit in days, so a camera left
    // parked in inertial space watches its subject fly out of frame within a
    // second of simulated time.
    state.orbitLock = true;
    camera.mode = 'orbit';
    const d = Math.max(entry.radius * 4.2, 1e-6);
    camera.setPose(
      v3(entry.pos[0] + d * 0.7, entry.pos[1] + d * 0.42, entry.pos[2] + d * 0.6),
      camera.orientation);
    camera.lookAt(v3(entry.pos[0], entry.pos[1], entry.pos[2]), v3(0, 1, 0));
    camera.speed = d * 0.25;
    camera.orbit.center.set(entry.pos);
    camera.orbit.distance = d;
    camera.orbit.minDistance = entry.radius * 1.02;
    camera.orbit.maxDistance = Math.max(entry.radius * 6000, 400);
    // Match the orbit angles to where the camera already is, so engaging the
    // lock does not snap the view somewhere else.
    camera.orbit.yaw = Math.atan2(camera.position[0] - entry.pos[0], camera.position[2] - entry.pos[2]);
    camera.orbit.pitch = Math.asin(Math.max(-1, Math.min(1, (camera.position[1] - entry.pos[1]) / d)));
    hud.flash(entry.label);
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
      case 'KeyX': {
        if (state.scale === 'cosmic') {
          const g = pickGalaxy();
          if (g) { state.targetGalaxy = g; beginTransition(() => descendToSystem(g)); }
          else hud.flash('no galaxy in view');
        } else {
          const b = stellar.placed[state.targetBodyIndex];
          if (b) frameBody(b);
        }
        break;
      }
      case 'KeyZ':
        if (state.scale === 'stellar') beginTransition(ascendToCosmic);
        break;
      case 'KeyT':
        if (state.scale === 'stellar' && stellar.placed.length) {
          state.targetBodyIndex = (state.targetBodyIndex + 1) % stellar.placed.length;
          hud.flash(stellar.placed[state.targetBodyIndex].label);
        }
        break;
      case 'KeyO':
        if (state.scale === 'stellar') {
          state.orbitLock = !state.orbitLock;
          hud.flash(state.orbitLock ? 'orbit lock engaged' : 'orbit lock released');
        }
        break;
      case 'KeyM': state.showDarkMatter = !state.showDarkMatter;
        hud.flash(state.showDarkMatter ? 'dark matter visible' : 'dark matter hidden'); break;
      case 'KeyG': state.showGalaxies = !state.showGalaxies;
        if (state.showGalaxies) state.lastGalaxyA = 0;
        hud.flash(state.showGalaxies ? 'galaxies visible' : 'galaxies hidden'); break;
      case 'Escape': setPanel(false); break;
    }
  });

  // A short fade to black between scales. Cutting straight from a hundred
  // megaparsecs to a few astronomical units is disorienting; a beat of darkness
  // lets the eye let go of one scale before picking up the next.
  let pendingTransition = null;
  function beginTransition(fn) {
    if (state.transition > 0) return;
    state.transition = 1.0;
    pendingTransition = fn;
  }

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
  let frameCount = 0;

  function frame(now) {
    frameCount++;
    const dtRaw = (now - last) / 1000;
    last = now;
    const dt = Math.min(dtRaw, 0.1);
    fpsAvg = mix(fpsAvg, 1 / Math.max(dtRaw, 1e-4), 0.08);

    resize();

    // Scale change: fade out, swap, fade back in.
    if (state.transition > 0) {
      state.transition = Math.max(0, state.transition - dt * 1.9);
      // transition counts 1 -> 0; fade = |2t - 1| dips to black at the midpoint
      // and returns, and the swap happens exactly at the bottom.
      const t = state.transition;
      post.settings.fade = Math.abs(2 * t - 1);
      if (pendingTransition && t <= 0.5) { pendingTransition(); pendingTransition = null; }
    } else {
      post.settings.fade = 1;
    }

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

    // Keep a live lock on whatever galaxy the camera is pointed at, so the
    // interface can say what pressing X would take you to.
    if (state.scale === 'cosmic') {
      state.hoverGalaxy = (frameCount % 6 === 0) ? pickGalaxy() : state.hoverGalaxy;
    }

    if (state.scale === 'stellar') {
      stellar.advance(state.paused ? 0 : dt);
      stellar.timeRate = state.yearsPerSecond;
      stellar.layout();
      const target = stellar.placed[state.targetBodyIndex];
      if (state.orbitLock && target) {
        camera.orbit.center.set(target.pos);
        camera.orbit.minDistance = target.radius * 1.05;
        camera.orbit.maxDistance = Math.max(target.radius * 4000, 200);
        if (camera.mode !== 'orbit') {
          camera.mode = 'orbit';
          camera.orbit.distance = Math.max(target.radius * 4.2,
            Math.hypot(camera.position[0] - target.pos[0], camera.position[1] - target.pos[1], camera.position[2] - target.pos[2]));
        }
        if (!panelOpen) camera.orbitUpdate(input, dt);
      } else {
        camera.mode = 'free';
      }
      // The near plane tracks the closest surface, which is the only way a
      // single depth buffer can span a planet's horizon and the outer system.
      const near = stellar.nearestSurface(camera.position);
      camera.near = Math.max(1e-8, Math.min(0.02, Math.abs(near) * 0.02));
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

      // Rebuilding the galaxy catalogue means reading particles back, which
      // stalls the pipeline, so it happens only when the universe has actually
      // changed enough to matter - a fixed fractional growth in scale factor.
      if (state.showGalaxies && pm.a > 1 / 13 && pm.a > state.lastGalaxyA * 1.09) {
        refreshGalaxies();
      }

      if (state.scale === 'cosmic') {
        if (state.showDarkMatter) web.render(pm, camera, state.boxSize);
        if (state.showGalaxies) galaxyRenderer.render(camera, state.boxSize);
      }
    }
    if (state.scale === 'stellar') {
      // A sunlit surface is a completely different signal from the integrated
      // column density of a filament, and wants its own exposure.
      post.settings.exposure = state.baseExposure * 3.2;
      stellar.render(camera, { time: now / 1000 });
    }

    post.render(now / 1000, canvas.width, canvas.height);

    if (state.hudVisible) {
      hud.update({
        universe, camera, web, state, fps: fpsAvg,
        galaxies: galaxyRenderer, stellar,
        pointerLocked: input.pointerLocked,
      });
    }

    input.endFrame();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // Expose for debugging and for the automated visual tests.
  window.__opus = {
    ctx, camera, universe, web, state, placeCamera, input, galaxyRenderer, refreshGalaxies,
    stellar, starfield, nebulaRenderer, descendToSystem, ascendToCosmic, pickGalaxy, frameBody,
    get halos() { return haloFinder ? haloFinder.halos : []; },
    haloDiag() {
      if (!haloFinder) return null;
      return {
        deltaVir: haloFinder.deltaVir,
        particleMass: haloFinder.particleMass,
        minResolvedMass: haloFinder.minResolvedMass,
        linkingLengthMpc: haloFinder.linkingLengthMpc,
      };
    },
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

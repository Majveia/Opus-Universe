// A cosmological Particle-Mesh N-body solver running entirely on the GPU.
//
// Each step:
//   1. deposit particle mass onto the grid with cloud-in-cell assignment
//   2. FFT the density contrast, divide by -k^2 to solve Poisson's equation,
//      and transform back to get the peculiar gravitational potential
//   3. differentiate the potential to get the force field
//   4. kick momenta and drift positions, using the scale factor as the time
//      coordinate
//
// The equations of motion are the standard comoving ones with canonical
// momentum p = a^2 dx/dt, in box units where the side of the periodic volume is
// 1 and time is measured in Hubble times:
//
//     dx/da = p / (a^3 E(a))
//     dp/da = -(3/2) Omega_m grad(psi) / (a^2 E(a)),   laplacian(psi) = delta
//
// Integrated with a leapfrog whose momenta are staggered half a step behind the
// positions, which is symplectic and so conserves energy over long runs far
// better than its cost suggests.

import { FULLSCREEN_VS } from '../core/gl.js';
import { atlasLayout, ATLAS_GLSL } from './atlas.js';
import { GPUFFT3D } from './gpufft.js';

/* ----------------------------------------------------------------- shaders */

const DEPOSIT_VS = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uPos;
uniform vec2 uParticleTexSize;
out float vWeight;
${ATLAS_GLSL}

void main() {
  int pid = gl_InstanceID;
  int w = int(uParticleTexSize.x);
  vec3 pos = texelFetch(uPos, ivec2(pid % w, pid / w), 0).xyz;

  float N = uAtlas.x;
  vec3 g = fract(pos) * N;
  ivec3 i0 = ivec3(floor(g));
  vec3 f = g - vec3(i0);

  // Eight vertices per particle, one per corner of the cloud-in-cell stencil.
  int c = gl_VertexID;
  ivec3 d = ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
  vec3 w3 = mix(1.0 - f, f, vec3(d));
  vWeight = w3.x * w3.y * w3.z;

  ivec2 texel = cellToTexel(i0 + d);
  vec2 ndc = (vec2(texel) + 0.5) / uAtlasSize * 2.0 - 1.0;
  gl_Position = vec4(ndc, 0.0, 1.0);
  gl_PointSize = 1.0;
}`;

const DEPOSIT_FS = `#version 300 es
precision highp float;
in float vWeight;
out float fragColor;
void main() { fragColor = vWeight; }`;

const POISSON_FS = `#version 300 es
precision highp float;
precision highp sampler2D;
out vec2 fragColor;
uniform sampler2D uSrc;
uniform float uDeconvolve;   // 1 = undo the CIC assignment window
uniform float uSmooth;       // Gaussian smoothing in cells
${ATLAS_GLSL}

float sinc(float x) { return abs(x) < 1e-6 ? 1.0 : sin(x) / x; }

void main() {
  int N = int(uAtlas.x);
  ivec3 c = texelToCell(ivec2(gl_FragCoord.xy));
  // Signed integer wavevector.
  ivec3 nv = ivec3(c.x <= N / 2 ? c.x : c.x - N,
                   c.y <= N / 2 ? c.y : c.y - N,
                   c.z <= N / 2 ? c.z : c.z - N);
  float n2 = float(nv.x * nv.x + nv.y * nv.y + nv.z * nv.z);
  if (n2 < 0.5) { fragColor = vec2(0.0); return; }   // the mean is not a source

  vec2 d = texelFetch(uSrc, ivec2(gl_FragCoord.xy), 0).xy;

  // In box units the Laplacian eigenvalue is -(2 pi n)^2, so psi = -delta / (4 pi^2 n^2).
  float green = -1.0 / (4.0 * 3.14159265358979 * 3.14159265358979 * n2);

  if (uDeconvolve > 0.5) {
    // Cloud-in-cell is applied twice - once scattering mass onto the grid and
    // once gathering the force back - so the transfer function enters squared.
    float fN = float(N);
    float wx = sinc(3.14159265358979 * float(nv.x) / fN);
    float wy = sinc(3.14159265358979 * float(nv.y) / fN);
    float wz = sinc(3.14159265358979 * float(nv.z) / fN);
    float W = pow(wx * wy * wz, 2.0);
    // Capped: near the Nyquist frequency the correction diverges and would
    // amplify aliasing noise into visible grid artefacts.
    green *= min(1.0 / (W * W), 4.0);
  }

  if (uSmooth > 0.0) {
    float kr = 2.0 * 3.14159265358979 * uSmooth / float(N);
    green *= exp(-0.5 * n2 * kr * kr);
  }

  fragColor = d * green;
}`;

// Extracts the real part of the potential into a scalar target.
const REALPART_FS = `#version 300 es
precision highp float;
out float fragColor;
uniform sampler2D uSrc;
void main() { fragColor = texelFetch(uSrc, ivec2(gl_FragCoord.xy), 0).x; }`;

// Copies the complex potential aside so it can be modulated twice, once for
// each of the two inverse transforms that produce the three force components.
const COPY_RG_FS = `#version 300 es
precision highp float;
out vec2 fragColor;
uniform sampler2D uSrc;
void main() { fragColor = texelFetch(uSrc, ivec2(gl_FragCoord.xy), 0).xy; }`;

// Spectral differentiation. The force is F = -grad(psi), which in Fourier space
// is F_j = -i 2 pi n_j psi - exact at every wavelength, unlike a finite
// difference, whose response falls away near the mesh scale precisely where the
// interesting structure lives.
//
// Two real components are recovered from a single inverse transform by packing
// them as the real and imaginary parts of one complex field: if F_x and F_y are
// both real then IFFT(F_x^ + i F_y^) = F_x + i F_y.
const SPECTRAL_GRAD_FS = `#version 300 es
precision highp float;
precision highp sampler2D;
out vec2 fragColor;
uniform sampler2D uPsi;
uniform int uPair;      // 0 -> pack (x, y); 1 -> z alone
${ATLAS_GLSL}

void main() {
  int N = int(uAtlas.x);
  ivec3 c = texelToCell(ivec2(gl_FragCoord.xy));
  ivec3 nv = ivec3(c.x <= N / 2 ? c.x : c.x - N,
                   c.y <= N / 2 ? c.y : c.y - N,
                   c.z <= N / 2 ? c.z : c.z - N);

  vec2 psi = texelFetch(uPsi, ivec2(gl_FragCoord.xy), 0).xy;
  const float TAU = 6.28318530717958;

  // The Nyquist plane has no signed partner, so its derivative is ill-defined;
  // zeroing it is the standard choice and keeps the force field real.
  float nx = (abs(nv.x) == N / 2) ? 0.0 : float(nv.x);
  float ny = (abs(nv.y) == N / 2) ? 0.0 : float(nv.y);
  float nz = (abs(nv.z) == N / 2) ? 0.0 : float(nv.z);

  if (uPair == 0) {
    // -i TAU (nx + i ny) * psi
    fragColor = vec2(TAU * (nx * psi.y + ny * psi.x),
                    -TAU * (nx * psi.x - ny * psi.y));
  } else {
    fragColor = vec2(TAU * nz * psi.y, -TAU * nz * psi.x);
  }
}`;

// Assembles the three real force components into one RGBA target.
const ASSEMBLE_FORCE_FS = `#version 300 es
precision highp float;
out vec4 fragColor;
uniform sampler2D uXY;   // .x = F_x, .y = F_y
uniform sampler2D uZ;    // .x = F_z
void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  fragColor = vec4(texelFetch(uXY, t, 0).xy, texelFetch(uZ, t, 0).x, 0.0);
}`;

const FORCE_FS = `#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 fragColor;
uniform sampler2D uPhi;
${ATLAS_GLSL}

float phi(ivec3 c) { return texelFetch(uPhi, cellToTexel(c), 0).x; }

void main() {
  ivec3 c = texelToCell(ivec2(gl_FragCoord.xy));
  float N = uAtlas.x;
  // Fourth-order central differences. The two-point stencil visibly softens
  // filaments at this resolution; four points sharpen them at negligible cost.
  vec3 grad;
  grad.x = (8.0 * (phi(c + ivec3(1,0,0)) - phi(c - ivec3(1,0,0)))
                 - (phi(c + ivec3(2,0,0)) - phi(c - ivec3(2,0,0)))) / 12.0;
  grad.y = (8.0 * (phi(c + ivec3(0,1,0)) - phi(c - ivec3(0,1,0)))
                 - (phi(c + ivec3(0,2,0)) - phi(c - ivec3(0,2,0)))) / 12.0;
  grad.z = (8.0 * (phi(c + ivec3(0,0,1)) - phi(c - ivec3(0,0,1)))
                 - (phi(c + ivec3(0,0,2)) - phi(c - ivec3(0,0,2)))) / 12.0;
  grad *= N;                       // 1 / cell size, in box units
  fragColor = vec4(-grad, 0.0);    // gravitational field g = -grad(psi)
}`;

const CIC_GATHER_GLSL = `
// Trilinear (cloud-in-cell) gather, the exact transpose of the deposit.
vec4 cicGather(sampler2D tex, vec3 pos) {
  float N = uAtlas.x;
  vec3 g = fract(pos) * N;
  ivec3 i0 = ivec3(floor(g));
  vec3 f = g - vec3(i0);
  vec4 sum = vec4(0.0);
  for (int c = 0; c < 8; c++) {
    ivec3 d = ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
    vec3 w3 = mix(1.0 - f, f, vec3(d));
    sum += texelFetch(tex, cellToTexel(i0 + d), 0) * (w3.x * w3.y * w3.z);
  }
  return sum;
}`;

const KICK_FS = `#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 fragColor;
uniform sampler2D uPos;
uniform sampler2D uVel;
uniform sampler2D uForce;
uniform float uKick;        // (3/2) Omega_m * Int da / (a^2 E)
${ATLAS_GLSL}
${CIC_GATHER_GLSL}

void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  vec3 pos = texelFetch(uPos, t, 0).xyz;
  vec3 p = texelFetch(uVel, t, 0).xyz;
  vec3 g = cicGather(uForce, pos).xyz;
  p += uKick * g;
  fragColor = vec4(p, length(p));
}`;

const DRIFT_FS = `#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 fragColor;
uniform sampler2D uPos;
uniform sampler2D uVel;
uniform sampler2D uDensity;
uniform float uDrift;       // Int da / (a^3 E)
uniform float uMeanDensity;
${ATLAS_GLSL}
${CIC_GATHER_GLSL}

void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  vec3 pos = texelFetch(uPos, t, 0).xyz;
  vec3 p = texelFetch(uVel, t, 0).xyz;
  pos = fract(pos + p * uDrift + 1.0);          // periodic box
  // Carry the local overdensity along with the particle; the renderer colours
  // by it, and reading it here costs nothing extra.
  float rho = cicGather(uDensity, pos).x / uMeanDensity;
  fragColor = vec4(pos, rho);
}`;

/* ------------------------------------------------------------------ solver */

export class ParticleMesh {
  /**
   * @param {GLContext} ctx
   * @param {object} o
   * @param {Cosmology} o.cosmology
   * @param {number} o.gridSize        force mesh resolution (power of two)
   * @param {number} o.boxSize         Mpc/h, for reporting physical units
   * @param {Float32Array} o.positions box units, 3 per particle, grid order
   * @param {Float32Array} o.velocities
   * @param {number} o.aInit
   * @param {number} o.aMax
   * @param {number} o.steps
   */
  constructor(ctx, o) {
    const gl = ctx.gl;
    this.ctx = ctx;
    this.gl = gl;
    this.cosmo = o.cosmology;
    this.boxSize = o.boxSize;
    this.layout = atlasLayout(o.gridSize);
    this.n = o.gridSize;

    if (!ctx.ext.floatBlend) {
      throw new Error('EXT_float_blend is required: mass assignment accumulates into a float target.');
    }

    this.aInit = o.aInit;
    this.aMax = o.aMax ?? 1.0;
    this.steps = o.steps ?? 400;
    this.stepIndex = 0;
    this.a = this.aInit;

    // Particles live in the same atlas layout as the grid, so particle i and
    // grid cell i are the same texel at t = 0.
    this.particleLayout = this.layout;
    this.particleCount = this.layout.count;
    const pw = this.layout.width, ph = this.layout.height;

    const rgba32 = { width: pw, height: ph, format: 'RGBA32F', filter: gl.NEAREST, wrap: gl.CLAMP_TO_EDGE };
    this.pos = [ctx.texture(rgba32), ctx.texture(rgba32)];
    this.vel = [ctx.texture(rgba32), ctx.texture(rgba32)];
    this.posFBO = this.pos.map((t) => ctx.framebuffer(t));
    this.velFBO = this.vel.map((t) => ctx.framebuffer(t));
    // Positions and momenta ping-pong independently: a kick touches only the
    // momenta and a drift only the positions, so neither needs copying.
    this.curPos = 0;
    this.curVel = 0;

    const r32 = { width: this.layout.width, height: this.layout.height, format: 'R32F', filter: gl.NEAREST, wrap: gl.CLAMP_TO_EDGE };
    this.density = ctx.texture(r32);
    this.densityFBO = ctx.framebuffer(this.density);
    this.phi = ctx.texture(r32);
    this.phiFBO = ctx.framebuffer(this.phi);
    this.force = ctx.texture({ ...r32, format: 'RGBA32F' });
    this.forceFBO = ctx.framebuffer(this.force);

    const rg32 = { ...r32, format: 'RG32F' };
    this.psiHat = ctx.texture(rg32);       // the potential in Fourier space
    this.psiHatFBO = ctx.framebuffer(this.psiHat);
    this.forceXY = ctx.texture(rg32);      // F_x and F_y, packed
    this.forceXYFBO = ctx.framebuffer(this.forceXY);
    this.forceZ = ctx.texture(rg32);
    this.forceZFBO = ctx.framebuffer(this.forceZ);

    this.fft = new GPUFFT3D(ctx, this.layout);

    this.progDeposit = ctx.program(DEPOSIT_VS, DEPOSIT_FS, 'pm-deposit');
    this.progPoisson = ctx.program(FULLSCREEN_VS, POISSON_FS, 'pm-poisson');
    this.progRealPart = ctx.program(FULLSCREEN_VS, REALPART_FS, 'pm-realpart');
    this.progForce = ctx.program(FULLSCREEN_VS, FORCE_FS, 'pm-force-fd');
    this.progCopyRG = ctx.program(FULLSCREEN_VS, COPY_RG_FS, 'pm-copy-rg');
    this.progSpectralGrad = ctx.program(FULLSCREEN_VS, SPECTRAL_GRAD_FS, 'pm-spectral-grad');
    this.progAssemble = ctx.program(FULLSCREEN_VS, ASSEMBLE_FORCE_FS, 'pm-assemble-force');
    this.progKick = ctx.program(FULLSCREEN_VS, KICK_FS, 'pm-kick');
    this.progDrift = ctx.program(FULLSCREEN_VS, DRIFT_FS, 'pm-drift');

    this.atlasU = new Float32Array([this.n, 1 / this.n, this.layout.tilesX, this.layout.tilesY]);
    this.atlasSizeU = new Float32Array([this.layout.width, this.layout.height]);
    this.particleTexSize = new Float32Array([pw, ph]);
    this.meanDensity = this.particleCount / (this.n ** 3);

    this.deconvolve = true;
    this.smoothCells = 0.0;
    // Spectral differentiation costs one extra inverse transform per step and
    // removes the mesh-scale force error entirely. Worth it.
    this.spectralGradient = true;

    if (o.positions) this.upload(o.positions, o.velocities);
    this._vao = gl.createVertexArray();
  }

  _atlas(prog) { prog.set('uAtlas', this.atlasU).set('uAtlasSize', this.atlasSizeU); }

  // Packs grid-ordered xyz triples into the RGBA atlas textures.
  upload(positions, velocities) {
    const { width, height, n } = this.layout;
    const posData = new Float32Array(width * height * 4);
    const velData = new Float32Array(width * height * 4);
    const idx = (x, y, z) => {
      const tx = z % this.layout.tilesX, ty = (z / this.layout.tilesX) | 0;
      return (ty * n + y) * width + (tx * n + x);
    };
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const g = x + n * (y + n * z);
      const a = idx(x, y, z);
      posData[a * 4] = positions[g * 3];
      posData[a * 4 + 1] = positions[g * 3 + 1];
      posData[a * 4 + 2] = positions[g * 3 + 2];
      posData[a * 4 + 3] = 1;
      velData[a * 4] = velocities[g * 3];
      velData[a * 4 + 1] = velocities[g * 3 + 1];
      velData[a * 4 + 2] = velocities[g * 3 + 2];
      velData[a * 4 + 3] = Math.hypot(velocities[g * 3], velocities[g * 3 + 1], velocities[g * 3 + 2]);
    }
    this.pos[0].upload(posData);
    this.vel[0].upload(velData);
    this.curPos = 0;
    this.curVel = 0;
    this.a = this.aInit;
    this.stepIndex = 0;
  }

  get positionTexture() { return this.pos[this.curPos]; }
  get velocityTexture() { return this.vel[this.curVel]; }
  get redshift() { return 1 / this.a - 1; }
  get finished() { return this.stepIndex >= this.steps; }

  // Scale factor at step i, logarithmically spaced so early times - when
  // structure is growing fastest in relative terms - get the resolution.
  aOfStep(i) {
    const t = Math.min(Math.max(i / this.steps, 0), 1);
    return this.aInit * Math.pow(this.aMax / this.aInit, t);
  }

  // Int_{a0}^{a1} da / (a^q E(a)), by Simpson's rule.
  _integrate(a0, a1, q) {
    if (a1 <= a0) return 0;
    const n = 16;
    const h = (a1 - a0) / n;
    let sum = 0;
    for (let i = 0; i <= n; i++) {
      const a = a0 + i * h;
      const f = 1 / (Math.pow(a, q) * this.cosmo.E(a));
      sum += (i === 0 || i === n ? 1 : (i % 2 ? 4 : 2)) * f;
    }
    return sum * h / 3;
  }

  driftFactor(a0, a1) { return this._integrate(a0, a1, 3); }
  kickFactor(a0, a1) { return this._integrate(a0, a1, 2); }

  /* ----------------------------------------------------------- one step -- */

  depositDensity() {
    const gl = this.gl;
    gl.disable(gl.DEPTH_TEST);
    this.densityFBO.bind();
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    this.progDeposit.use().set('uParticleTexSize', this.particleTexSize);
    this._atlas(this.progDeposit);
    this.progDeposit.tex('uPos', this.pos[this.curPos]);
    gl.bindVertexArray(this._vao);
    gl.drawArraysInstanced(gl.POINTS, 0, 8, this.particleCount);
    gl.disable(gl.BLEND);
  }

  // Transforms the density contrast, applies the Green's function, and leaves
  // the peculiar potential in Fourier space in `psiHat`.
  solvePotential() {
    this.fft.load(this.density, this.meanDensity, 1 / this.meanDensity);
    this.fft.transform(false);

    this.psiHatFBO.bind();
    this.progPoisson.use()
      .set('uDeconvolve', this.deconvolve ? 1 : 0)
      .set('uSmooth', this.smoothCells);
    this._atlas(this.progPoisson);
    this.progPoisson.tex('uSrc', this.fft.a.tex);
    this.ctx.drawFullscreen();
  }

  computeForce() {
    if (this.spectralGradient) {
      // F_x and F_y from one inverse transform, F_z from a second.
      for (let pair = 0; pair < 2; pair++) {
        this.fft.a.fbo.bind();
        this.progSpectralGrad.use().set('uPair', pair);
        this._atlas(this.progSpectralGrad);
        this.progSpectralGrad.tex('uPsi', this.psiHat);
        this.ctx.drawFullscreen();
        this.fft.transform(true);
        (pair === 0 ? this.forceXYFBO : this.forceZFBO).bind();
        this.progCopyRG.use().tex('uSrc', this.fft.current);
        this.ctx.drawFullscreen();
      }
      this.forceFBO.bind();
      this.progAssemble.use().tex('uXY', this.forceXY).tex('uZ', this.forceZ);
      this.ctx.drawFullscreen();
    } else {
      // Fallback: one inverse transform, then fourth-order finite differences.
      this.fft.a.fbo.bind();
      this.progCopyRG.use().tex('uSrc', this.psiHat);
      this.ctx.drawFullscreen();
      this.fft.transform(true);
      this.phiFBO.bind();
      this.progRealPart.use().tex('uSrc', this.fft.current);
      this.ctx.drawFullscreen();
      this.forceFBO.bind();
      this.progForce.use();
      this._atlas(this.progForce);
      this.progForce.tex('uPhi', this.phi);
      this.ctx.drawFullscreen();
    }
  }

  kick(factor) {
    const dst = 1 - this.curVel;
    this.velFBO[dst].bind();
    this.progKick.use().set('uKick', factor);
    this._atlas(this.progKick);
    this.progKick.tex('uPos', this.pos[this.curPos]).tex('uVel', this.vel[this.curVel]).tex('uForce', this.force);
    this.ctx.drawFullscreen();
    this.curVel = dst;
  }

  drift(factor) {
    const dst = 1 - this.curPos;
    this.posFBO[dst].bind();
    this.progDrift.use().set('uDrift', factor).set('uMeanDensity', this.meanDensity);
    this._atlas(this.progDrift);
    this.progDrift.tex('uPos', this.pos[this.curPos]).tex('uVel', this.vel[this.curVel]).tex('uDensity', this.density);
    this.ctx.drawFullscreen();
    this.curPos = dst;
  }

  // Advances the universe by one leapfrog step. Returns false once a_max is reached.
  step() {
    if (this.finished) return false;
    const i = this.stepIndex;
    const a0 = this.aOfStep(i);
    const a1 = this.aOfStep(i + 1);
    // Momenta are staggered: they live at the geometric midpoint of each step,
    // except at the very start where the initial conditions define them at a_init.
    const kickFrom = i === 0 ? a0 : Math.sqrt(this.aOfStep(i - 1) * a0);
    const kickTo = Math.sqrt(a0 * a1);

    this.depositDensity();
    this.solvePotential();
    this.computeForce();

    this.kick(1.5 * this.cosmo.omegaM * this.kickFactor(kickFrom, kickTo));
    this.drift(this.driftFactor(a0, a1));

    this.stepIndex = i + 1;
    this.a = a1;
    return true;
  }

  // Diagnostics: reads the density field back and reports its statistics.
  readDensity() {
    const out = new Float32Array(this.layout.width * this.layout.height);
    this.densityFBO.readPixels(out);
    return out;
  }

  readPositions() {
    const out = new Float32Array(this.layout.width * this.layout.height * 4);
    this.posFBO[this.curPos].readPixels(out);
    return out;
  }

  dispose() {
    this.fft.dispose();
    this.pos.forEach((t) => t.dispose());
    this.vel.forEach((t) => t.dispose());
    this.posFBO.forEach((f) => f.dispose());
    this.velFBO.forEach((f) => f.dispose());
    this.density.dispose(); this.densityFBO.dispose();
    this.phi.dispose(); this.phiFBO.dispose();
    this.force.dispose(); this.forceFBO.dispose();
    this.psiHat.dispose(); this.psiHatFBO.dispose();
    this.forceXY.dispose(); this.forceXYFBO.dispose();
    this.forceZ.dispose(); this.forceZFBO.dispose();
  }
}

// Renders the dark matter distribution as additive HDR splats.
//
// Two ideas do most of the work here. First, every particle carries equal mass,
// so simply adding their splats together produces a surface-density map - the
// bright knots and thin filaments emerge from the accumulation rather than from
// any authored shading. Second, the periodic box is tiled around the camera, so
// the universe has no edge: fly in any direction and structure keeps coming.
//
// Positions are transformed camera-relative before projection. At cosmological
// distances a float32 world position would visibly quantise; subtracting the
// eye first keeps every coordinate small.

import { v3, v3set, m4, m4mul, sphereInFrustum } from '../core/math.js';

const VS = `#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D uPos;        // .xyz box units [0,1), .w local overdensity rho/rho_bar
uniform sampler2D uVel;        // .xyz canonical momentum, .w its magnitude
uniform mat4 uViewProjRel;     // projection * rotation-only view
uniform vec3 uCameraPos;       // world units (Mpc/h)
uniform vec3 uTileOffset;
uniform vec2 uTexSize;
uniform float uBoxSize;
uniform int uStride;
uniform float uPixelsPerRadian; // (viewport height / 2) / tan(fov / 2)
uniform float uSpacing;         // mean interparticle separation, world units
uniform float uKernelScale;     // smoothing length in units of that spacing
uniform float uMaxSize;
uniform float uFadeNear;
uniform float uFadeFar;
uniform float uBrightness;
uniform float uVelocityScale;
uniform float uDensityGamma;
uniform float uStrideCompensation;

out vec3 vColor;
out float vIntensity;

// A colour ramp for cosmic structure: cold indigo voids, violet sheets, ember
// filaments, gold haloes, white-hot cluster cores. Chosen so the darkest values
// sit just above true black, which is where OLED panels look their best.
vec3 webColor(float logd) {
  float t = clamp((logd + 1.0) / 3.6, 0.0, 1.0);
  const vec3 cVoid   = vec3(0.055, 0.075, 0.230);
  const vec3 cSheet  = vec3(0.180, 0.130, 0.470);
  const vec3 cFil    = vec3(0.560, 0.235, 0.520);
  const vec3 cEmber  = vec3(0.990, 0.440, 0.215);
  const vec3 cGold   = vec3(1.000, 0.830, 0.470);
  const vec3 cCore   = vec3(1.000, 0.985, 0.960);

  vec3 c = mix(cVoid,  cSheet, smoothstep(0.00, 0.30, t));
  c = mix(c, cFil,   smoothstep(0.26, 0.50, t));
  c = mix(c, cEmber, smoothstep(0.46, 0.68, t));
  c = mix(c, cGold,  smoothstep(0.64, 0.84, t));
  c = mix(c, cCore,  smoothstep(0.82, 1.00, t));
  return c;
}

void main() {
  int id = gl_VertexID * uStride;
  int w = int(uTexSize.x);
  ivec2 t = ivec2(id % w, id / w);
  vec4 P = texelFetch(uPos, t, 0);
  vec4 V = texelFetch(uVel, t, 0);

  vec3 world = P.xyz * uBoxSize + uTileOffset;
  vec3 rel = world - uCameraPos;
  float dist = length(rel);

  gl_Position = uViewProjRel * vec4(rel, 1.0);

  float rho = max(P.w, 1e-3);

  // Adaptive smoothing, as in smoothed-particle hydrodynamics: a mass element
  // in a void spreads its mass over a large kernel, one in a cluster core over
  // a tiny one. Without this, sparsely sampled voids look like grain instead of
  // like emptiness, and cluster cores smear instead of concentrating.
  float hWorld = uSpacing * uKernelScale * clamp(pow(rho, -0.3333333), 0.30, 4.0);
  float sizePx = 2.0 * hWorld * uPixelsPerRadian / max(dist, 1e-5);
  float drawn = clamp(sizePx, 1.0, uMaxSize);
  gl_PointSize = drawn;

  // What is being drawn is projected mass density: each particle spreads its
  // mass over the area of its kernel, so surface brightness is column density
  // and does not fall off with distance. Where the splat has to be drawn larger
  // or smaller than its true footprint, the peak is rescaled to keep the total
  // mass it represents unchanged.
  // Never above one: when the camera is inside a particle's own smoothing
  // kernel the true footprint exceeds anything that can be rasterised, and
  // amplifying to compensate would turn a single particle into a white screen.
  float footprint = min(sizePx / drawn, 1.0);
  float columnDensity = pow(rho, 0.6666667) / (uKernelScale * uKernelScale);
  float fade = 1.0 - smoothstep(uFadeNear, uFadeFar, dist);

  float logd = log(rho) * 0.4342944819;
  vec3 col = webColor(logd * uDensityGamma);

  // Virialised regions run hot: high peculiar speed shifts the splat toward
  // blue-white, which separates cluster cores from the cold filaments feeding them.
  float heat = clamp(V.w * uVelocityScale, 0.0, 1.0);
  col = mix(col, mix(col, vec3(0.72, 0.86, 1.0), 0.78), heat * heat);

  vColor = col;
  vIntensity = uBrightness * columnDensity * footprint * footprint * fade * uStrideCompensation;
  if (vIntensity < 1e-6) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // cull
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vColor;
in float vIntensity;
out vec4 fragColor;

void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  // Gaussian core with a wider, fainter skirt: reads as a soft glow rather
  // than a disc, and the skirt is what makes overlapping splats fuse into
  // continuous filaments instead of visible beads.
  float g = exp(-r2 * 4.0) + 0.18 * exp(-r2 * 1.1);
  fragColor = vec4(vColor * (vIntensity * g), 1.0);
}`;

export class CosmicWebRenderer {
  constructor(ctx) {
    this.ctx = ctx;
    this.gl = ctx.gl;
    this.prog = ctx.program(VS, FS, 'cosmic-web');
    this.vao = ctx.gl.createVertexArray();

    this.settings = {
      kernelScale: 1.15,      // smoothing length in mean interparticle spacings
      maxSize: 48.0,
      // With the column-density normalisation below, the mean surface
      // brightness of one box depth is exactly brightness / kernelScale^2, so
      // this number is calibrated rather than guessed.
      brightness: 0.05,
      velocityScale: 0.9,
      densityGamma: 1.0,
      fadeNear: 1.1,          // in units of the box size
      fadeFar: 3.0,
      tileRadius: 2,
      pointBudget: 7_000_000,
    };

    this._tmpM = m4();
    this._tile = v3();
    this._stats = { tiles: 0, points: 0 };
  }

  get stats() { return this._stats; }

  /**
   * @param {ParticleMesh} pm
   * @param {Camera} camera
   * @param {number} boxWorldSize  physical size of the periodic box in world units
   */
  render(pm, camera, boxWorldSize) {
    const gl = this.gl;
    const s = this.settings;

    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);          // additive: order independent

    // Projection times the rotation-only view, since positions arrive
    // camera-relative.
    const viewRel = m4();
    camera.viewNoTranslation(viewRel);
    m4mul(this._tmpM, camera.proj, viewRel);

    const layout = pm.layout;
    // Apparent size must follow the field of view and the framebuffer height,
    // or the same scene reads differently at every window size.
    const viewportH = this.gl.drawingBufferHeight;
    const pixelsPerRadian = (viewportH * 0.5) / Math.tan(camera.fov * 0.5);
    const perSide = Math.cbrt(pm.particleCount);
    const spacing = boxWorldSize / perSide;

    const prog = this.prog.use()
      .set('uViewProjRel', this._tmpM)
      .set('uCameraPos', camera.position)
      .set('uTexSize', new Float32Array([layout.width, layout.height]))
      .set('uBoxSize', boxWorldSize)
      .set('uPixelsPerRadian', pixelsPerRadian)
      .set('uSpacing', spacing)
      .set('uKernelScale', s.kernelScale)
      .set('uMaxSize', s.maxSize)
      // Column density is independent of how finely the volume is sampled, so
      // the per-particle contribution scales inversely with particles per side.
      .set('uBrightness', s.brightness / perSide)
      .set('uVelocityScale', s.velocityScale)
      .set('uDensityGamma', s.densityGamma)
      .set('uFadeNear', s.fadeNear * boxWorldSize)
      .set('uFadeFar', s.fadeFar * boxWorldSize);
    prog.tex('uPos', pm.positionTexture).tex('uVel', pm.velocityTexture);

    gl.bindVertexArray(this.vao);

    // Which copy of the box the camera currently occupies.
    const cx = Math.floor(camera.position[0] / boxWorldSize);
    const cy = Math.floor(camera.position[1] / boxWorldSize);
    const cz = Math.floor(camera.position[2] / boxWorldSize);

    const R = s.tileRadius;
    const cullRadius = boxWorldSize * 0.8661;      // half-diagonal of the box
    const maxDist = s.fadeFar * boxWorldSize;
    const visible = [];
    for (let iz = -R; iz <= R; iz++) {
      for (let iy = -R; iy <= R; iy++) {
        for (let ix = -R; ix <= R; ix++) {
          const ox = (cx + ix) * boxWorldSize;
          const oy = (cy + iy) * boxWorldSize;
          const oz = (cz + iz) * boxWorldSize;
          const mx = ox + boxWorldSize * 0.5, my = oy + boxWorldSize * 0.5, mz = oz + boxWorldSize * 0.5;
          const d = Math.hypot(mx - camera.position[0], my - camera.position[1], mz - camera.position[2]);
          if (d - cullRadius > maxDist) continue;
          if (!sphereInFrustum(camera.frustum, mx, my, mz, cullRadius)) continue;
          visible.push({ ox, oy, oz, d });
        }
      }
    }
    // Nearest first, so the point budget is spent where it is most visible.
    visible.sort((a, b) => a.d - b.d);

    const total = pm.particleCount;
    let spent = 0;
    let drawn = 0;
    for (const tile of visible) {
      // Distant copies of the box are drawn from a sub-lattice of the particle
      // set. Every particle carries the same mass, so a regular subsample is an
      // unbiased estimate of the same density field, just noisier - and at that
      // distance the noise is far below a pixel.
      let stride = 1;
      const remaining = s.pointBudget - spent;
      if (remaining <= 0) break;
      const relD = tile.d / boxWorldSize;
      if (relD > 0.9) stride = 2;
      if (relD > 1.6) stride = 4;
      if (relD > 2.2) stride = 8;
      let count = Math.floor(total / stride);
      if (count > remaining) {
        stride = Math.max(stride, Math.ceil(total / remaining));
        count = Math.floor(total / stride);
      }
      if (count < 1) continue;

      v3set(this._tile, tile.ox, tile.oy, tile.oz);
      // Drawing every n-th particle removes (n-1)/n of the mass; brightening
      // the survivors keeps a distant copy of the box at the same total
      // luminosity as a near one.
      prog.set('uTileOffset', this._tile).set('uStride', stride)
        .set('uStrideCompensation', stride);
      gl.drawArrays(gl.POINTS, 0, count);
      spent += count;
      drawn++;
      this.ctx.drawCalls++;
    }

    this._stats.tiles = drawn;
    this._stats.points = spent;

    gl.disable(gl.BLEND);
    gl.depthMask(true);
  }
}

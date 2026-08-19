// The night sky, generated from where you are standing in a galaxy.
//
// Rather than scattering random dots on a sphere, this samples actual stellar
// positions from the host galaxy's density profile - an exponential disk plus a
// bulge - places the observer inside it, and works out what each star looks like
// from there: apparent brightness from its luminosity and distance, colour from
// its temperature, and reddening from the dust it shines through.
//
// The Milky Way's band across the sky is not drawn. It appears because the
// observer is inside a disk, and looking along the disk means looking through
// far more stars than looking out of it.

import { RNG } from '../core/rng.js';
import { mainSequence, blackbodyRGB } from './color.js';
import { mainSequenceLifetimeGyr } from '../universe/system.js';
import { m4, m4mul } from '../core/math.js';

const VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aDir;      // unit direction on the sky
layout(location = 1) in vec4 aColour;   // rgb, and apparent magnitude scale in w
uniform mat4 uViewProjRel;
uniform float uPixelScale;
uniform float uBrightness;
out vec3 vColour;
out float vIntensity;

void main() {
  // Placed far away and never translated: the sky does not move with the ship.
  gl_Position = uViewProjRel * vec4(aDir * 1.0e7, 1.0);
  float flux = aColour.w;      // already compressed to [0,1] by the generator
  // Bright stars are drawn slightly larger, which is how the eye and every
  // camera read them; spreading the same flux over more pixels means the peak
  // has to come down to match, hence the division by the area.
  float size = clamp(1.0 + log(1.0 + flux * 60.0) * 0.75, 1.0, 7.0);
  gl_PointSize = size;
  vColour = aColour.rgb;
  // uBrightness carries the area factor: with flux normalised so the brightest
  // star in the sky is 1, a bare flux/size^2 would peak near 1/49 and vanish.
  vIntensity = uBrightness * flux * 26.0 / (size * size);
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vColour;
in float vIntensity;
out vec4 fragColor;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  float g = exp(-r2 * 3.4) + 0.12 * exp(-r2 * 0.9);
  fragColor = vec4(vColour * (vIntensity * g), 1.0);
}`;

export class Starfield {
  constructor(ctx) {
    this.ctx = ctx;
    this.gl = ctx.gl;
    this.prog = ctx.program(VS, FS, 'starfield');
    this.vao = ctx.gl.createVertexArray();
    this.dirBuf = ctx.gl.createBuffer();
    this.colBuf = ctx.gl.createBuffer();
    this.count = 0;
    this.brightness = 1.6;
    this._vp = m4();
  }

  /**
   * Builds the sky seen from a point inside a galaxy.
   * @param {object} o
   * @param {number} o.seed
   * @param {number} o.count            stars to generate
   * @param {number} o.diskScaleKpc     disk scale length
   * @param {number} o.scaleHeightKpc
   * @param {number} o.observerRadiusKpc  where in the disk the observer sits
   * @param {number} o.bulgeFraction
   * @param {number} o.galaxyAgeGyr
   */
  build(o = {}) {
    const {
      seed = 1, count = 60000,
      diskScaleKpc = 2.6, scaleHeightKpc = 0.3,
      observerRadiusKpc = 8.2, bulgeFraction = 0.15, bulgeScaleKpc = 0.7,
      galaxyAgeGyr = 10, dustScaleHeightKpc = 0.12, dustOpacityPerKpc = 0.22,
    } = o;

    const rng = new RNG(seed >>> 0);
    const dirs = new Float32Array(count * 3);
    const cols = new Float32Array(count * 4);

    // Observer sits on the +x axis of the disk, in the midplane.
    const ox = observerRadiusKpc, oy = 0, oz = 0;
    let kept = 0;
    let maxFlux = 0;

    for (let i = 0; i < count; i++) {
      let x, y, z;
      if (rng.f() < bulgeFraction) {
        // Bulge: a roughly exponential spheroid at the centre.
        const r = -bulgeScaleKpc * Math.log(1 - rng.f() * 0.999);
        const d = rng.onSphere([0, 0, 0]);
        x = d[0] * r; y = d[1] * r; z = d[2] * r;
      } else {
        // Disk: exponential in radius, exponential in height.
        // Inverse-transform the radial profile r e^{-r/h} by rejection.
        let r;
        for (let t = 0; t < 40; t++) {
          r = -diskScaleKpc * Math.log(1 - rng.f() * 0.999) - diskScaleKpc * Math.log(1 - rng.f() * 0.999);
          if (r < diskScaleKpc * 8) break;
        }
        const th = rng.f() * Math.PI * 2;
        const h = -scaleHeightKpc * Math.log(1 - rng.f() * 0.999) * (rng.f() < 0.5 ? 1 : -1);
        x = r * Math.cos(th); y = h; z = r * Math.sin(th);
      }

      const dx = x - ox, dy = y - oy, dz = z - oz;
      const dist = Math.hypot(dx, dy, dz);
      if (dist < 1e-4) continue;

      // Stellar mass from the present-day population.
      let m = 0;
      for (let t = 0; t < 24; t++) {
        const cand = rng.powerLaw(0.1, 40, -2.3);
        if (rng.f() < Math.min(1, mainSequenceLifetimeGyr(cand) / galaxyAgeGyr)) { m = cand; break; }
      }
      if (m === 0) m = 0.3;
      const ms = mainSequence(m);

      // Extinction: integrate the dust disk along the line of sight. A thin
      // dusty layer in the midplane is what cuts the dark rifts through the
      // Milky Way, so a star seen through the plane is both dimmer and redder.
      const midplaneFraction = Math.exp(-Math.abs((y + oy) * 0.5) / dustScaleHeightKpc);
      const tau = dustOpacityPerKpc * dist * midplaneFraction;
      const extinction = Math.exp(-tau);

      const flux = ms.luminosity / (dist * dist) * extinction;
      if (flux < 2e-6) continue;

      const c = blackbodyRGB(ms.temperature);
      // Reddening: dust removes blue light preferentially.
      const red = Math.exp(-tau * 0.55);
      const blue = Math.exp(-tau * 1.6);

      const j = kept;
      dirs[j * 3] = dx / dist; dirs[j * 3 + 1] = dy / dist; dirs[j * 3 + 2] = dz / dist;
      cols[j * 4] = c[0]; cols[j * 4 + 1] = c[1] * red; cols[j * 4 + 2] = c[2] * blue;
      cols[j * 4 + 3] = flux;
      if (flux > maxFlux) maxFlux = flux;
      kept++;
    }

    // Normalise so the brightest star sits at a predictable level whatever the
    // galaxy; the eye adapts, and so should the exposure.
    const norm = maxFlux > 0 ? 1 / Math.pow(maxFlux, 0.55) : 1;
    for (let i = 0; i < kept; i++) {
      cols[i * 4 + 3] = Math.pow(cols[i * 4 + 3], 0.55) * norm;
    }

    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dirBuf);
    gl.bufferData(gl.ARRAY_BUFFER, dirs.subarray(0, kept * 3), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, cols.subarray(0, kept * 4), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    this.count = kept;
    this.stats = { generated: count, kept, maxFlux };
    return this.stats;
  }

  render(camera) {
    if (!this.count) return;
    const gl = this.gl;
    const viewRel = m4();
    camera.viewNoTranslation(viewRel);
    m4mul(this._vp, camera.proj, viewRel);

    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    this.prog.use().set('uViewProjRel', this._vp).set('uBrightness', this.brightness);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
    gl.depthMask(true);
    this.ctx.drawCalls++;
  }

  dispose() {
    this.gl.deleteBuffer(this.dirBuf);
    this.gl.deleteBuffer(this.colBuf);
    this.gl.deleteVertexArray(this.vao);
  }
}
